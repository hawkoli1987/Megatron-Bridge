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

"""Unit tests for multiple validation sets evaluation functions."""

from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import math

from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.config import MockGPTDatasetConfig


class TestMultipleValidationEvaluation:
    """Unit tests for multiple validation sets evaluation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_forward_step_func = Mock()
        self.mock_process_non_loss_data_func = Mock()
        
        # Mock state
        self.mock_state = Mock()
        self.mock_state.train_state = Mock()
        self.mock_state.train_state.step = 100
        self.mock_state.train_state.consumed_train_samples = 1000
        self.mock_state.train_state.start_time = 0.0
        self.mock_state.start_time = 0.0
        
        # Mock config
        self.mock_state.cfg = Mock()
        self.mock_state.cfg.logger = Mock()
        self.mock_state.cfg.logger.log_validation_ppl_to_tensorboard = True
        
        # Mock wandb logger
        self.mock_state.wandb_logger = Mock()
        
        # Mock writer
        self.mock_writer = Mock()

    def create_mock_loss_dict(self, loss_value=2.0):
        """Create a mock loss dictionary."""
        loss_tensor = torch.tensor(loss_value)
        return {"loss": loss_tensor}

    def create_mock_data_iterator(self, num_batches=3):
        """Create a mock data iterator."""
        mock_iterator = Mock()
        mock_iterator.__iter__ = Mock(return_value=iter(range(num_batches)))
        mock_iterator.__next__ = Mock(return_value={"tokens": torch.tensor([[1, 2, 3]])})
        return mock_iterator

    def create_mock_data_iterator_list(self, num_datasets=3, num_batches=3):
        """Create a list of mock data iterators."""
        return [self.create_mock_data_iterator(num_batches) for _ in range(num_datasets)]

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_evaluate_single_dataset_unchanged(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test that single dataset evaluation behavior is unchanged."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=False)
        data_iterator = self.create_mock_data_iterator()
        loss_dict = self.create_mock_loss_dict(2.5)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify
        mock_evaluate.assert_called_once()
        mock_print.assert_called()
        
        # Verify writer calls for single dataset
        assert self.mock_writer.add_scalar.call_count >= 2  # loss and loss vs samples
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation" in call[0] for call in calls)
        assert any("loss validation vs samples" in call[0] for call in calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_evaluate_multiple_datasets_basic(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test basic evaluation of multiple validation datasets."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1", "train2"],  # train datasets
            [["val1", "val2", "val3"], [0.5, 0.3, 0.2]],  # validation datasets with weights
            ["test1"]  # test datasets
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=3)
        loss_dict_1 = self.create_mock_loss_dict(2.0)
        loss_dict_2 = self.create_mock_loss_dict(3.0)
        loss_dict_3 = self.create_mock_loss_dict(4.0)
        
        mock_evaluate.side_effect = [
            (loss_dict_1, {}, False),
            (loss_dict_2, {}, False),
            (loss_dict_3, {}, False)
        ]
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify
        assert mock_evaluate.call_count == 3  # Called for each dataset
        
        # Verify writer calls for individual datasets
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation val1" in call[0] for call in calls)
        assert any("loss validation val2" in call[0] for call in calls)
        assert any("loss validation val3" in call[0] for call in calls)
        
        # Verify aggregated logging
        assert any("loss validation (aggregated)" in call[0] for call in calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_results_aggregation(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test loss aggregation across multiple validation datasets."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2", "val3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=3)
        
        # Different loss values for each dataset
        loss_dict_1 = self.create_mock_loss_dict(2.0)  # val1
        loss_dict_2 = self.create_mock_loss_dict(4.0)  # val2
        loss_dict_3 = self.create_mock_loss_dict(6.0)  # val3
        
        mock_evaluate.side_effect = [
            (loss_dict_1, {}, False),
            (loss_dict_2, {}, False),
            (loss_dict_3, {}, False)
        ]
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify aggregated loss calculation: (2.0 + 4.0 + 6.0) / 3 = 4.0
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        aggregated_calls = [call for call in calls if "aggregated" in call[0]]
        assert len(aggregated_calls) > 0
        
        # Check that aggregated value is logged
        for call in self.mock_writer.add_scalar.call_args_list:
            if "aggregated" in call[0][0]:
                assert call[0][2] == 4.0  # Aggregated loss value

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_dataset_identification(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test proper identification and naming of validation datasets."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val_dataset_1", "val_dataset_2", "val_dataset_3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=3)
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify dataset names are used correctly
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation val_dataset_1" in call[0] for call in calls)
        assert any("loss validation val_dataset_2" in call[0] for call in calls)
        assert any("loss validation val_dataset_3" in call[0] for call in calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_dataset_identification_fallback(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test fallback dataset naming when blend_per_split is not available."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = None  # No blend_per_split
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=2)
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify fallback naming is used
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation val_dataset_0" in call[0] for call in calls)
        assert any("loss validation val_dataset_1" in call[0] for call in calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_logging_separate_datasets(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test individual dataset logging to TensorBoard and Weights & Biases."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2"], [0.5, 0.5]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=2)
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify TensorBoard logging for individual datasets
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation val1" in call[0] for call in calls)
        assert any("loss validation val2" in call[0] for call in calls)
        assert any("loss validation val1 ppl" in call[0] for call in calls)
        assert any("loss validation val2 ppl" in call[0] for call in calls)
        
        # Verify Weights & Biases logging
        wandb_calls = [call[0] for call in self.mock_state.wandb_logger.log.call_args_list]
        assert any("loss validation val1" in str(call) for call in wandb_calls)
        assert any("loss validation val2" in str(call) for call in wandb_calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_logging_aggregated(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test aggregated metrics logging."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2"], [0.5, 0.5]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=2)
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify aggregated logging
        calls = [call[0] for call in self.mock_writer.add_scalar.call_args_list]
        assert any("loss validation (aggregated)" in call[0] for call in calls)
        assert any("loss validation (aggregated) ppl" in call[0] for call in calls)
        
        # Verify Weights & Biases aggregated logging
        wandb_calls = [call[0] for call in self.mock_state.wandb_logger.log.call_args_list]
        assert any("loss validation (aggregated)" in str(call) for call in wandb_calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_console_output_format(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test console output format for multiple validation datasets."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2", "val3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=3)
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify console output format
        print_calls = [call[0] for call in mock_print.call_args_list]
        assert any("aggregated across 3 datasets" in call for call in print_calls)
        assert any("loss value:" in call for call in print_calls)
        assert any("loss PPL:" in call for call in print_calls)

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_timelimit_handling(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test timelimit handling during multiple dataset evaluation."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2", "val3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=3)
        loss_dict = self.create_mock_loss_dict(2.0)
        
        # First dataset succeeds, second hits timelimit
        mock_evaluate.side_effect = [
            (loss_dict, {}, False),  # val1 - success
            (loss_dict, {}, True),   # val2 - timelimit hit
        ]
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify only first two datasets were processed (timelimit hit on second)
        assert mock_evaluate.call_count == 2

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_non_loss_data_processing(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test processing of non-loss data with multiple validation datasets."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2"], [0.5, 0.5]], ["test1"]
        ]
        
        data_iterator = self.create_mock_data_iterator_list(num_datasets=2)
        loss_dict = self.create_mock_loss_dict(2.0)
        non_loss_data = {"accuracy": 0.95}
        mock_evaluate.return_value = (loss_dict, non_loss_data, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify non-loss data function is called only once with first dataset's data
        self.mock_process_non_loss_data_func.assert_called_once_with(
            non_loss_data, self.mock_state.train_state.step, self.mock_writer
        )

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_empty_datasets_handling(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test handling of empty validation dataset list."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2", "val3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        data_iterator = []  # Empty list
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute - should handle empty list gracefully
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify no evaluation calls were made
        mock_evaluate.assert_not_called()

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.training.eval.print_rank_last')
    @patch('megatron.bridge.training.eval.is_last_rank', return_value=True)
    def test_validation_with_none_datasets(self, mock_is_last_rank, mock_print, mock_evaluate):
        """Test handling when some validation datasets are None."""
        # Setup
        self.mock_state.cfg.dataset = MockGPTDatasetConfig(multiple_validation_sets=True)
        self.mock_state.cfg.dataset.blend_per_split = [
            ["train1"], [["val1", "val2", "val3"], [0.5, 0.3, 0.2]], ["test1"]
        ]
        
        # Create iterator list with None elements
        data_iterator = [
            self.create_mock_data_iterator(),
            None,  # None dataset
            self.create_mock_data_iterator()
        ]
        
        loss_dict = self.create_mock_loss_dict(2.0)
        mock_evaluate.return_value = (loss_dict, {}, False)
        
        # Execute
        evaluate_and_print_results(
            self.mock_state,
            "test",
            self.mock_forward_step_func,
            data_iterator,
            self.mock_model,
            self.mock_process_non_loss_data_func,
            {},
            True,
            None,
            self.mock_writer
        )
        
        # Verify only non-None datasets were processed
        assert mock_evaluate.call_count == 2  # Only 2 non-None datasets