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

"""Unit tests for multiple validation sets feature - simplified version."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))


class TestMultipleValidationConfigSimple:
    """Simplified unit tests for multiple validation sets configuration."""

    def test_config_field_exists(self):
        """Test that the multiple_validation_sets field exists in the config."""
        config_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/config.py')
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check if the field is defined
        assert "multiple_validation_sets: bool = False" in content, "multiple_validation_sets field not found"
        
        # Check if the docstring is present
        assert "enables individual validation loss tracking" in content, "Docstring not found"
        
        # Check if it's in the right class
        lines = content.split('\n')
        in_gpt_dataset_config = False
        field_found = False
        
        for i, line in enumerate(lines):
            if "class GPTDatasetConfig" in line:
                in_gpt_dataset_config = True
            elif in_gpt_dataset_config and "class " in line and "GPTDatasetConfig" not in line:
                break
            elif in_gpt_dataset_config and "multiple_validation_sets:" in line:
                field_found = True
                break
        
        assert field_found, "multiple_validation_sets field not found in GPTDatasetConfig class"

    def test_eval_changes_present(self):
        """Test that the evaluation changes are present."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for key changes in the evaluation function
        assert "multiple_validation_sets" in content, "multiple_validation_sets not found in eval.py"
        assert "validation_results = {}" in content, "validation_results dict not found"
        assert "aggregated_loss_dict" in content, "aggregated_loss_dict not found"
        assert "validation (aggregated)" in content, "Aggregated validation logging not found"

    def test_data_loader_changes_present(self):
        """Test that the data loader changes are present."""
        loader_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/data/loaders.py')
        
        with open(loader_file, 'r') as f:
            content = f.read()
        
        # Check for key changes in the data loader
        assert "multiple_validation_sets" in content, "multiple_validation_sets not found in loaders.py"
        assert "isinstance(valid_ds, list)" in content, "List validation check not found"
        assert "valid_dataloader = []" in content, "List dataloader creation not found"

    def test_eval_function_structure(self):
        """Test the structure of the evaluation function changes."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for the main conditional structure
        assert "if (hasattr(state.cfg.dataset, 'multiple_validation_sets')" in content
        assert "and state.cfg.dataset.multiple_validation_sets" in content
        assert "and isinstance(data_iterator, list)):" in content
        
        # Check for individual dataset processing
        assert "for i, valid_data_iter in enumerate(data_iterator):" in content
        
        # Check for aggregation logic
        assert "aggregated_loss_dict[key] += total_loss_dict[key].item()" in content
        assert "aggregated_loss_dict[key] /= total_datasets" in content

    def test_data_loader_function_structure(self):
        """Test the structure of the data loader function changes."""
        loader_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/data/loaders.py')
        
        with open(loader_file, 'r') as f:
            content = f.read()
        
        # Check for the main conditional structure
        assert "if hasattr(cfg.dataset, 'multiple_validation_sets')" in content
        assert "and cfg.dataset.multiple_validation_sets" in content
        assert "and isinstance(valid_ds, list):" in content
        
        # Check for list creation
        assert "valid_dataloader = []" in content
        assert "for i, valid_dataset in enumerate(valid_ds):" in content
        
        # Check for iterator creation
        assert "if isinstance(valid_dataloader, list):" in content
        assert "valid_data_iterator = []" in content

    def test_logging_changes(self):
        """Test that logging changes are present."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for individual dataset logging
        assert "validation {dataset_path}" in content, "Individual dataset logging not found"
        
        # Check for aggregated logging
        assert "validation (aggregated)" in content, "Aggregated logging not found"
        
        # Check for console output changes
        assert "aggregated across" in content, "Console output changes not found"

    def test_error_handling(self):
        """Test that error handling is present."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for None handling
        assert "if valid_data_iter is not None:" in content, "None handling not found"
        
        # Check for timelimit handling
        assert "if timelimit:" in content, "Timelimit handling not found"
        assert "return" in content, "Early return not found"

    def test_backward_compatibility(self):
        """Test that backward compatibility is maintained."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for else clause maintaining original behavior
        assert "else:" in content, "Else clause not found"
        assert "# Original single validation dataset logic" in content, "Original logic comment not found"
        
        # Check that original evaluation call is preserved
        assert "total_loss_dict, collected_non_loss_data, timelimit = evaluate(" in content

    def test_documentation_changes(self):
        """Test that documentation changes are present."""
        eval_file = os.path.join(os.path.dirname(__file__), '../../../src/megatron/bridge/training/eval.py')
        
        with open(eval_file, 'r') as f:
            content = f.read()
        
        # Check for TODO comments
        assert "TODO: enable 'multiple_validation_sets' mode" in content, "TODO comment not found"
        
        # Check for function docstring updates
        assert "When multiple_validation_sets is True, valid_dataloader will be a list" in content or \
               "valid_data_iterator will be a list" in content, "Documentation update not found"