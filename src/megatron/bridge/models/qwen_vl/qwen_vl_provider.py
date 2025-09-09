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

from dataclasses import dataclass, field
from typing import List

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models import (
    Qwen2ModelProvider,
)

from .modeling_qwen25_vl import Qwen25VLModel


@dataclass
class Qwen25VLVisionConfig:
    """Dataclass wrapper for Qwen2_5_VLVisionConfig to enable proper serialization."""
    
    # Vision model configuration
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 3584
    hidden_act: str = "silu"
    image_size: int = 448
    in_channels: int = 3
    intermediate_size: int = 3420
    layer_norm_eps: float = 1e-6
    num_heads: int = 16
    num_positions: int = 577
    output_dim: int = 3584
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: int = 4
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: List[int] = field(default_factory=lambda: [7, 15, 23, 31])
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Handle any additional kwargs that weren't explicitly defined."""
        # This allows for flexibility in case there are additional config parameters
        pass
    
    def to_hf_config(self) -> Qwen2_5_VLVisionConfig:
        """Convert to the original HuggingFace transformers config object."""
        config = Qwen2_5_VLVisionConfig()
        
        # Copy all attributes to the transformers config
        for key, value in self.__dict__.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_hf_config(cls, config: Qwen2_5_VLVisionConfig, **kwargs) -> "Qwen25VLVisionConfig":
        """Create from a HuggingFace transformers config object with support for additional kwargs."""
        # Extract all relevant attributes from the transformers config
        config_kwargs = {}
        for field_info in cls.__dataclass_fields__.values():
            field_name = field_info.name
            if hasattr(config, field_name):
                config_kwargs[field_name] = getattr(config, field_name)
        
        # Override with any additional kwargs
        config_kwargs.update(kwargs)
        
        return cls(**config_kwargs)


# =============================================================================
# Qwen 2.5 VL Model Providers
# =============================================================================


@dataclass
class Qwen25VLModelProvider(Qwen2ModelProvider):
    """
    Base model provider for Qwen 2.5 VL Models.
    """

    # Language configuration inherited from Qwen25ModelProvider3B
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen25VLVisionConfig = field(default_factory=Qwen25VLVisionConfig)

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def hf_vision_config(self) -> Qwen2_5_VLVisionConfig:
        """Get the vision config as a HuggingFace transformers config object."""
        return self.vision_config.to_hf_config()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen25VLModel:
        model = Qwen25VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        
        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
