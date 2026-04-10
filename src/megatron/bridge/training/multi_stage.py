# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-stage pretraining with per-stage dataset blend configurations.

Enables training with different data mix ratios at different phases of training,
e.g., for continual pre-training workflows where the domain data proportion
changes over the training horizon.

Phase transitions happen in-loop: at boundary iterations the data iterators
are rebuilt with the new blend while the model, optimizer, scheduler, and
W&B run continue uninterrupted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StageBlendEntry:
    """A single dataset entry within a stage's blend configuration.

    Attributes:
        weight: Sampling weight for this dataset (relative, not necessarily summing to 1).
        path: Absolute path to the Megatron-format dataset prefix (without .bin/.idx).
    """

    weight: float
    path: str


@dataclass
class StageConfig:
    """Configuration for a single training stage.

    Attributes:
        name: Human-readable name for logging and checkpoint identification.
        duration_in_token: Number of tokens to train in this stage.
        blend: List of (weight, path) entries defining the data mix for this stage.
        lr: Optional per-stage learning rate override.
    """

    name: str
    duration_in_token: int
    blend: list[StageBlendEntry] = field(default_factory=list)
    lr: Optional[float] = None

    def to_blend_list(self) -> list:
        """Convert blend entries to the flat list format for get_blend_from_list().

        Returns:
            Flat list of [weight1, path1, weight2, path2, ...] as strings.
        """
        blend_list = []
        for entry in self.blend:
            blend_list.append(str(entry.weight))
            blend_list.append(entry.path)
        return blend_list


@dataclass
class MultiStageConfig:
    """Configuration for multi-stage pretraining.

    Attributes:
        stages: Ordered list of training stages to execute sequentially.
    """

    stages: list[StageConfig] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total token budget across all stages."""
        return sum(s.duration_in_token for s in self.stages)

    def cumulative_iters(self, seq_length: int, global_batch_size: int) -> list[int]:
        """Calculate cumulative train_iters boundaries for each stage.

        Args:
            seq_length: Sequence length per sample.
            global_batch_size: Global batch size (samples per iteration).

        Returns:
            List of cumulative iteration counts, one per stage.
        """
        tokens_per_iter = seq_length * global_batch_size
        cumulative = []
        total = 0
        for stage in self.stages:
            total += int(stage.duration_in_token) // tokens_per_iter
            cumulative.append(total)
        return cumulative

    def transition_steps(self, seq_length: int, global_batch_size: int) -> list[int]:
        """Step numbers where phase transitions occur.

        Returns boundaries between consecutive stages (excludes the final
        boundary which is the end of training).

        Args:
            seq_length: Sequence length per sample.
            global_batch_size: Global batch size (samples per iteration).

        Returns:
            List of iteration numbers where transitions happen.
        """
        cumulative = self.cumulative_iters(seq_length, global_batch_size)
        return cumulative[:-1]

    def get_phase_index_at_step(self, step: int, seq_length: int, global_batch_size: int) -> int:
        """Return 0-based phase index for a given training step.

        Args:
            step: Current training iteration.
            seq_length: Sequence length per sample.
            global_batch_size: Global batch size.

        Returns:
            Index into self.stages for the phase containing this step.
        """
        cumulative = self.cumulative_iters(seq_length, global_batch_size)
        for i, boundary in enumerate(cumulative):
            if step < boundary:
                return i
        return len(self.stages) - 1

    def get_phase_samples(self, phase_index: int, seq_length: int, global_batch_size: int) -> int:
        """Return total training samples for a specific phase.

        Args:
            phase_index: 0-based phase index.
            seq_length: Sequence length per sample.
            global_batch_size: Global batch size.

        Returns:
            Number of training samples in this phase.
        """
        tokens_per_iter = seq_length * global_batch_size
        phase_iters = int(self.stages[phase_index].duration_in_token) // tokens_per_iter
        return phase_iters * global_batch_size

    def consumed_samples_in_phase(self, step: int, seq_length: int, global_batch_size: int) -> int:
        """Samples consumed within the current phase (relative to phase start).

        Args:
            step: Current training iteration.
            seq_length: Sequence length per sample.
            global_batch_size: Global batch size.

        Returns:
            Number of samples consumed since the start of the current phase.
        """
        cumulative = self.cumulative_iters(seq_length, global_batch_size)
        phase_idx = self.get_phase_index_at_step(step, seq_length, global_batch_size)
        phase_start = cumulative[phase_idx - 1] if phase_idx > 0 else 0
        return (step - phase_start) * global_batch_size

    @classmethod
    def from_dict(cls, data: dict) -> "MultiStageConfig":
        """Parse a MultiStageConfig from a dictionary (e.g., loaded from YAML).

        Expected format:
            {"stages": [
                {"name": "...", "duration_in_token": N,
                 "train_data": [{"weight": W, "path": "P"}, ...]},
                ...
            ]}

        Args:
            data: Dictionary with a "stages" key.

        Returns:
            MultiStageConfig instance.
        """
        stages = []
        for stage_dict in data.get("stages", []):
            blend_entries = []
            for entry in stage_dict.get("train_data", []):
                blend_entries.append(
                    StageBlendEntry(weight=float(entry["weight"]), path=str(entry["path"]))
                )
            stages.append(
                StageConfig(
                    name=stage_dict.get("name", f"stage_{len(stages)+1}"),
                    duration_in_token=int(float(stage_dict["duration_in_token"])),
                    blend=blend_entries,
                    lr=stage_dict.get("lr"),
                )
            )
        return cls(stages=stages)
