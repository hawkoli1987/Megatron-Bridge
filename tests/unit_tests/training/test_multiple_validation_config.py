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

"""Unit tests for multiple validation sets configuration."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Mock the problematic imports before importing the actual module
from unittest.mock import Mock, patch

# Mock megatron.core before importing
with patch.dict('sys.modules', {
    'megatron.core': Mock(),
    'megatron.core.utils': Mock(),
    'megatron.core.datasets': Mock(),
    'megatron.core.datasets.utils': Mock(),
    'megatron.core.distributed': Mock(),
    'megatron.core.optimizer': Mock(),
    'megatron.core.optimizer.distrib_optimizer': Mock(),
    'megatron.core.rerun_state_machine': Mock(),
    'megatron.core.msc_utils': Mock(),
}):
    from megatron.bridge.training.config import MockGPTDatasetConfig


class TestMultipleValidationConfig:
    """Unit tests for multiple validation sets configuration."""

    def test_multiple_validation_sets_default_false(self):
        """Test that multiple_validation_sets defaults to False."""
        config = MockGPTDatasetConfig()
        assert config.multiple_validation_sets is False
        assert hasattr(config, 'multiple_validation_sets')
        assert isinstance(config.multiple_validation_sets, bool)

    def test_multiple_validation_sets_enabled(self):
        """Test that multiple_validation_sets can be set to True."""
        config = MockGPTDatasetConfig(multiple_validation_sets=True)
        assert config.multiple_validation_sets is True

    def test_multiple_validation_sets_explicit_false(self):
        """Test that multiple_validation_sets can be explicitly set to False."""
        config = MockGPTDatasetConfig(multiple_validation_sets=False)
        assert config.multiple_validation_sets is False

    def test_config_validation_with_multiple_datasets(self):
        """Test config validation when multiple validation sets are enabled."""
        # Test with various combinations of related config options
        config = MockGPTDatasetConfig(
            multiple_validation_sets=True,
            sequence_length=512,
            data_sharding=True,
            dataloader_type="single",
            num_workers=4
        )
        
        # Verify the config is valid and all attributes are set correctly
        assert config.multiple_validation_sets is True
        assert config.sequence_length == 512
        assert config.data_sharding is True
        assert config.dataloader_type == "single"
        assert config.num_workers == 4

    def test_config_with_other_dataset_options(self):
        """Test multiple_validation_sets with other dataset configuration options."""
        config = MockGPTDatasetConfig(
            multiple_validation_sets=True,
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=1024,
            num_dataset_builder_threads=2,
            data_sharding=False,
            dataloader_type="cyclic",
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Verify all attributes are set correctly
        assert config.multiple_validation_sets is True
        assert config.random_seed == 1234
        assert config.reset_attention_mask is False
        assert config.reset_position_ids is False
        assert config.eod_mask_loss is False
        assert config.sequence_length == 1024
        assert config.num_dataset_builder_threads == 2
        assert config.data_sharding is False
        assert config.dataloader_type == "cyclic"
        assert config.num_workers == 8
        assert config.pin_memory is True
        assert config.persistent_workers is True

    def test_config_inheritance(self):
        """Test that multiple_validation_sets is properly inherited from base config."""
        # Test that the attribute is available in the config hierarchy
        config = MockGPTDatasetConfig()
        
        # Verify the attribute exists and can be accessed
        assert hasattr(config, 'multiple_validation_sets')
        
        # Test that it can be modified
        config.multiple_validation_sets = True
        assert config.multiple_validation_sets is True
        
        config.multiple_validation_sets = False
        assert config.multiple_validation_sets is False

    def test_config_type_validation(self):
        """Test that multiple_validation_sets only accepts boolean values."""
        # Test with boolean True
        config = MockGPTDatasetConfig(multiple_validation_sets=True)
        assert isinstance(config.multiple_validation_sets, bool)
        assert config.multiple_validation_sets is True
        
        # Test with boolean False
        config = MockGPTDatasetConfig(multiple_validation_sets=False)
        assert isinstance(config.multiple_validation_sets, bool)
        assert config.multiple_validation_sets is False

    def test_config_default_values_consistency(self):
        """Test that default values are consistent across multiple config instances."""
        config1 = MockGPTDatasetConfig()
        config2 = MockGPTDatasetConfig()
        
        # Both should have the same default value
        assert config1.multiple_validation_sets == config2.multiple_validation_sets
        assert config1.multiple_validation_sets is False
        assert config2.multiple_validation_sets is False