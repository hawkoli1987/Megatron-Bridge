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

"""Unit tests for multi-stage pretraining configuration and logic."""

import pytest

from megatron.core.datasets.utils import get_blend_from_list

from megatron.bridge.training.multi_stage import (
    MultiStageConfig,
    StageBlendEntry,
    StageConfig,
)

SEQ = 4096
GBS = 1024
TOKENS_PER_ITER = SEQ * GBS  # 4_194_304


# ---------------------------------------------------------------------------
# StageConfig tests
# ---------------------------------------------------------------------------


class TestStageConfig:
    def test_to_blend_list(self):
        stage = StageConfig(
            name="test",
            duration_in_token=5_000_000_000,
            blend=[
                StageBlendEntry(weight=0.1, path="/data/dataset_a"),
                StageBlendEntry(weight=0.9, path="/data/dataset_b"),
            ],
        )
        result = stage.to_blend_list()
        assert result == ["0.1", "/data/dataset_a", "0.9", "/data/dataset_b"]

    def test_to_blend_list_empty(self):
        stage = StageConfig(name="empty", duration_in_token=1000, blend=[])
        assert stage.to_blend_list() == []

    def test_to_blend_list_single_dataset(self):
        stage = StageConfig(
            name="single",
            duration_in_token=1000,
            blend=[StageBlendEntry(weight=1.0, path="/data/only")],
        )
        assert stage.to_blend_list() == ["1.0", "/data/only"]

    def test_blend_list_parsed_by_get_blend_from_list(self):
        stage = StageConfig(
            name="test",
            duration_in_token=1000,
            blend=[
                StageBlendEntry(weight=0.3, path="/data/a"),
                StageBlendEntry(weight=0.7, path="/data/b"),
            ],
        )
        raw = stage.to_blend_list()
        parsed = get_blend_from_list(raw)
        assert parsed is not None
        prefixes, weights = parsed
        assert prefixes == ["/data/a", "/data/b"]
        assert weights == [0.3, 0.7]


# ---------------------------------------------------------------------------
# MultiStageConfig tests
# ---------------------------------------------------------------------------


class TestMultiStageConfig:
    @pytest.fixture
    def two_stage_config(self):
        return MultiStageConfig(
            stages=[
                StageConfig(
                    name="stage1",
                    duration_in_token=5_000_000_000,
                    blend=[
                        StageBlendEntry(weight=0.1, path="/data/a"),
                        StageBlendEntry(weight=0.9, path="/data/b"),
                    ],
                ),
                StageConfig(
                    name="stage2",
                    duration_in_token=5_000_000_000,
                    blend=[
                        StageBlendEntry(weight=0.9, path="/data/a"),
                        StageBlendEntry(weight=0.1, path="/data/b"),
                    ],
                ),
            ]
        )

    def test_total_tokens(self, two_stage_config):
        assert two_stage_config.total_tokens == 10_000_000_000

    def test_cumulative_iters(self, two_stage_config):
        result = two_stage_config.cumulative_iters(seq_length=SEQ, global_batch_size=GBS)
        assert len(result) == 2
        expected_per_stage = 5_000_000_000 // TOKENS_PER_ITER  # 1192
        assert result[0] == expected_per_stage
        assert result[1] == 2 * expected_per_stage

    def test_cumulative_iters_single_stage(self):
        config = MultiStageConfig(
            stages=[StageConfig(name="only", duration_in_token=1_000_000_000, blend=[])],
        )
        result = config.cumulative_iters(seq_length=2048, global_batch_size=512)
        assert result == [1_000_000_000 // (2048 * 512)]

    # --- Phase-aware query methods ---

    def test_transition_steps(self, two_stage_config):
        """Transition steps should be the boundary between stages (not the end)."""
        transitions = two_stage_config.transition_steps(SEQ, GBS)
        cumulative = two_stage_config.cumulative_iters(SEQ, GBS)
        assert transitions == [cumulative[0]]  # only the first boundary
        assert len(transitions) == 1

    def test_transition_steps_three_stages(self):
        config = MultiStageConfig(
            stages=[
                StageConfig(name="s1", duration_in_token=4_194_304_000, blend=[]),
                StageConfig(name="s2", duration_in_token=4_194_304_000, blend=[]),
                StageConfig(name="s3", duration_in_token=4_194_304_000, blend=[]),
            ]
        )
        transitions = config.transition_steps(SEQ, GBS)
        assert len(transitions) == 2
        cumulative = config.cumulative_iters(SEQ, GBS)
        assert transitions == cumulative[:2]

    def test_get_phase_index_at_step(self, two_stage_config):
        boundary = two_stage_config.cumulative_iters(SEQ, GBS)[0]
        # Before boundary → phase 0
        assert two_stage_config.get_phase_index_at_step(0, SEQ, GBS) == 0
        assert two_stage_config.get_phase_index_at_step(boundary - 1, SEQ, GBS) == 0
        # At and after boundary → phase 1
        assert two_stage_config.get_phase_index_at_step(boundary, SEQ, GBS) == 1
        assert two_stage_config.get_phase_index_at_step(boundary + 100, SEQ, GBS) == 1

    def test_get_phase_samples(self, two_stage_config):
        samples = two_stage_config.get_phase_samples(0, SEQ, GBS)
        expected_iters = 5_000_000_000 // TOKENS_PER_ITER
        assert samples == expected_iters * GBS

    def test_consumed_samples_in_phase(self, two_stage_config):
        boundary = two_stage_config.cumulative_iters(SEQ, GBS)[0]
        # At step 0: consumed 0 in phase 0
        assert two_stage_config.consumed_samples_in_phase(0, SEQ, GBS) == 0
        # At step 10: consumed 10*GBS in phase 0
        assert two_stage_config.consumed_samples_in_phase(10, SEQ, GBS) == 10 * GBS
        # At boundary (phase 1 start): consumed 0 in phase 1
        assert two_stage_config.consumed_samples_in_phase(boundary, SEQ, GBS) == 0
        # At boundary+5: consumed 5*GBS in phase 1
        assert two_stage_config.consumed_samples_in_phase(boundary + 5, SEQ, GBS) == 5 * GBS

    # --- Parsing tests ---

    def test_from_dict(self):
        data = {
            "stages": [
                {
                    "name": "warmup",
                    "duration_in_token": 5e9,
                    "train_data": [
                        {"weight": 0.1, "path": "/data/domain"},
                        {"weight": 0.9, "path": "/data/general"},
                    ],
                },
                {
                    "name": "focus",
                    "duration_in_token": 5e9,
                    "train_data": [
                        {"weight": 0.9, "path": "/data/domain"},
                        {"weight": 0.1, "path": "/data/general"},
                    ],
                    "lr": 5e-5,
                },
            ]
        }
        config = MultiStageConfig.from_dict(data)
        assert len(config.stages) == 2
        assert config.stages[0].name == "warmup"
        assert config.stages[0].duration_in_token == 5_000_000_000
        assert config.stages[0].blend[0].weight == 0.1
        assert config.stages[0].lr is None
        assert config.stages[1].lr == 5e-5

    def test_from_dict_auto_names(self):
        data = {
            "stages": [
                {"duration_in_token": 1e9, "train_data": []},
                {"duration_in_token": 2e9, "train_data": []},
            ]
        }
        config = MultiStageConfig.from_dict(data)
        assert config.stages[0].name == "stage_1"
        assert config.stages[1].name == "stage_2"

    def test_from_dict_scientific_notation(self):
        data = {
            "stages": [
                {
                    "name": "test",
                    "duration_in_token": 5.0e9,
                    "train_data": [{"weight": 1.0, "path": "/data/x"}],
                }
            ]
        }
        config = MultiStageConfig.from_dict(data)
        assert config.stages[0].duration_in_token == 5_000_000_000
