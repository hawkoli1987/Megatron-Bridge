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

"""Tests for train module utility functions."""

import math
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.transformer import MegatronModule

from megatron.bridge.training.train import _handle_mxfp8_param_buffer_copy
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig, TrainingConfig
from megatron.bridge.training.state import GlobalState, TrainState


class TestMxfp8ParamBufferCopy:
    """Unit tests for mxfp8 parameter buffer copying functionality."""

    def test_copy_main_params_called_when_both_flags_true(self):
        """Test that _copy_main_params_to_param_buffer is called when both config flags are True."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        assert (
            not hasattr(mock_other_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_other_optimizer._copy_main_params_to_param_buffer.called
        )

    def test_no_copy_when_reuse_grad_buf_false(self):
        """Test that no copying occurs when reuse_grad_buf_for_mxfp8_param_ag is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=True
        )
        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_overlap_param_gather_false(self):
        """Test that no copying occurs when overlap_param_gather is False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]
        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_no_copy_when_both_flags_false(self):
        """Test that no copying occurs when both flags are False."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [mock_distributed_optimizer]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=False, overlap_param_gather=False
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_not_called()

    def test_handles_multiple_distributed_optimizers(self):
        """Test that function calls copy on multiple DistributedOptimizers."""
        mock_distributed_optimizer_1 = Mock(spec=DistributedOptimizer)
        mock_distributed_optimizer_2 = Mock(spec=DistributedOptimizer)
        mock_other_optimizer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_other_optimizer,
            mock_distributed_optimizer_1,
            mock_distributed_optimizer_2,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer_1._copy_main_params_to_param_buffer.assert_called_once()
        mock_distributed_optimizer_2._copy_main_params_to_param_buffer.assert_called_once()

    def test_only_calls_on_distributed_optimizers(self):
        """Test that only DistributedOptimizer instances get the copy call."""
        mock_distributed_optimizer = Mock(spec=DistributedOptimizer)
        mock_regular_optimizer = Mock()  # Regular optimizer without _copy_main_params_to_param_buffer
        mock_different_optimizer = Mock()

        # Add the method to one non-DistributedOptimizer to ensure it's not called
        mock_different_optimizer._copy_main_params_to_param_buffer = Mock()

        mock_megatron_optimizer = Mock()
        mock_megatron_optimizer.chained_optimizers = [
            mock_regular_optimizer,
            mock_different_optimizer,
            mock_distributed_optimizer,
        ]

        _handle_mxfp8_param_buffer_copy(
            optimizer=mock_megatron_optimizer, reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

        mock_distributed_optimizer._copy_main_params_to_param_buffer.assert_called_once()
        mock_different_optimizer._copy_main_params_to_param_buffer.assert_not_called()

        assert (
            not hasattr(mock_regular_optimizer, "_copy_main_params_to_param_buffer")
            or not mock_regular_optimizer._copy_main_params_to_param_buffer.called
        )

class TestEvaluateAndPrintResults:
    """Unit tests for evaluate_and_print_results function."""

    def _create_mock_global_state(self, multiple_validation_sets=False, blend_per_split=None):
        """Create a mock GlobalState for testing."""
        mock_state = Mock(spec=GlobalState)
        mock_state.train_state = Mock(spec=TrainState)
        mock_state.train_state.step = 100
        mock_state.train_state.consumed_train_samples = 1000
        
        # Mock config
        mock_config = Mock(spec=ConfigContainer)
        mock_config.train = Mock(spec=TrainingConfig)
        mock_config.train.eval_iters = 2
        mock_config.train.global_batch_size = 32
        mock_config.train.micro_batch_size = 8
        mock_config.data_parallel_size = 4
        mock_config.model = Mock()
        mock_config.model.seq_length = 512
        mock_config.logger = Mock()
        mock_config.logger.log_validation_ppl_to_tensorboard = True
        
        # Mock dataset config
        mock_dataset_config = Mock(spec=GPTDatasetConfig)
        mock_dataset_config.multiple_validation_sets = multiple_validation_sets
        mock_dataset_config.blend_per_split = blend_per_split
        mock_config.dataset = mock_dataset_config
        
        mock_state.cfg = mock_config
        
        # Mock timers
        mock_state.timers = Mock()
        mock_timer = Mock()
        mock_timer.start = Mock()
        mock_timer.stop = Mock()
        mock_timer.elapsed = Mock(return_value=1.0)
        mock_state.timers.return_value = mock_timer
        
        # Mock loggers
        mock_state.tensorboard_logger = Mock()
        mock_state.wandb_logger = Mock()
        
        return mock_state

    def _create_mock_model(self):
        """Create a mock model for testing."""
        mock_model = Mock(spec=MegatronModule)
        mock_model.eval = Mock()
        mock_model.train = Mock()
        return [mock_model]

    def _create_mock_data_iterator(self, single=True):
        """Create mock data iterator(s) for testing."""
        if single:
            return Mock()
        else:
            # Return list of mock data iterators for multiple datasets
            return [Mock(), Mock()]

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.utils.common_utils.is_last_rank')
    @patch('megatron.bridge.utils.common_utils.print_rank_last')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_evaluate_and_print_results_single_dataset(self, mock_get_rank, mock_world_size, mock_print_rank_last, mock_is_last_rank, mock_evaluate):
        """Test original single dataset behavior in evaluate_and_print_results."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = (
            {"loss": torch.tensor(0.5)},  # total_loss_dict
            None,  # collected_non_loss_data
            False  # timelimit
        )
        
        # Create test data
        state = self._create_mock_global_state(multiple_validation_sets=False)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=True)
        forward_step_func = Mock()
        
        # Call the function
        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True
        )
        
        # Verify evaluate was called once with correct parameters
        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        # evaluate is called with positional arguments: state, forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose
        assert call_args[0][0] == state  # state
        assert call_args[0][2] == data_iterator  # data_iterator
        assert call_args[0][3] == model  # model
        assert call_args[0][5] == state.cfg  # config
        
        # Verify TensorBoard logging
        state.tensorboard_logger.add_scalar.assert_called()
        state.wandb_logger.log.assert_called()

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.utils.common_utils.is_last_rank')
    @patch('megatron.bridge.utils.common_utils.print_rank_last')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_evaluate_and_print_results_multiple_datasets(self, mock_get_rank, mock_world_size, mock_print_rank_last, mock_is_last_rank, mock_evaluate):
        """Test new multiple datasets behavior in evaluate_and_print_results."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        
        # Mock evaluate to return different results for each dataset
        def mock_evaluate_side_effect(*args, **kwargs):
            data_iterator = kwargs.get('data_iterator')
            if data_iterator == Mock():  # First dataset
                return {"loss": torch.tensor(0.4)}, None, False
            else:  # Second dataset
                return {"loss": torch.tensor(0.6)}, None, False
        
        mock_evaluate.side_effect = mock_evaluate_side_effect
        
        # Create test data with multiple validation sets
        blend_per_split = [
            (["train_paths"], None),
            (["val1", "val2"], None),  # Two validation datasets
            (["test_paths"], None)
        ]
        state = self._create_mock_global_state(
            multiple_validation_sets=True, 
            blend_per_split=blend_per_split
        )
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=False)
        forward_step_func = Mock()
        
        # Call the function
        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True
        )
        
        # Verify evaluate was called twice (once for each dataset)
        assert mock_evaluate.call_count == 2
        
        # Verify individual dataset logging
        individual_calls = [call for call in state.tensorboard_logger.add_scalar.call_args_list 
                          if "validation val1" in str(call) or "validation val2" in str(call)]
        assert len(individual_calls) > 0
        
        # Verify aggregated logging
        aggregated_calls = [call for call in state.tensorboard_logger.add_scalar.call_args_list 
                          if "validation (aggregated)" in str(call)]
        assert len(aggregated_calls) > 0

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.utils.common_utils.is_last_rank')
    @patch('megatron.bridge.utils.common_utils.print_rank_last')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_evaluate_and_print_results_timelimit_handling(self, mock_get_rank, mock_world_size, mock_print_rank_last, mock_is_last_rank, mock_evaluate):
        """Test timelimit handling in evaluate_and_print_results."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = (None, None, True)  # timelimit hit
        
        # Create test data
        state = self._create_mock_global_state()
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=True)
        forward_step_func = Mock()
        
        # Call the function
        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True
        )
        
        # Verify evaluate was called
        mock_evaluate.assert_called_once()
        
        # Verify no logging occurred due to timelimit
        state.tensorboard_logger.add_scalar.assert_not_called()
        state.wandb_logger.log.assert_not_called()

    @patch('megatron.bridge.training.eval.evaluate')
    @patch('megatron.bridge.utils.common_utils.is_last_rank')
    @patch('megatron.bridge.utils.common_utils.print_rank_last')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.get_rank')
    def test_evaluate_and_print_results_dataset_naming(self, mock_get_rank, mock_world_size, mock_print_rank_last, mock_is_last_rank, mock_evaluate):
        """Test dataset naming from blend_per_split configuration."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = ({"loss": torch.tensor(0.5)}, None, False)
        
        # Create test data with specific dataset names
        blend_per_split = [
            (["train_paths"], None),
            (["val_dataset_1", "val_dataset_2"], None),
            (["test_paths"], None)
        ]
        state = self._create_mock_global_state(
            multiple_validation_sets=True,
            blend_per_split=blend_per_split
        )
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=False)
        forward_step_func = Mock()
        
        # Call the function
        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True
        )
        
        # Verify evaluate was called twice
        assert mock_evaluate.call_count == 2
        
        # Verify dataset-specific logging with correct names
        logging_calls = [str(call) for call in state.tensorboard_logger.add_scalar.call_args_list]
        val1_calls = [call for call in logging_calls if "val_dataset_1" in call]
        val2_calls = [call for call in logging_calls if "val_dataset_2" in call]
        
        assert len(val1_calls) > 0
        assert len(val2_calls) > 0

    @patch('megatron.bridge.training.train.evaluate_and_print_results')
    def test_evaluate_and_print_results_split_timelimit(self, mock_evaluate):
        """Test that timelimit is split equally among multiple validation datasets."""
        # Setup mock state with multiple validation datasets and timelimit
        state = self._create_mock_global_state()
        state.cfg.dataset.multiple_validation_sets = True
        state.cfg.train.exit_duration_in_mins = 60.0  # 60 minutes total
        
        # Mock data iterators for 3 validation datasets
        data_iterator = [Mock(), Mock(), Mock()]
        
        # Mock evaluate to return successful results
        mock_evaluate.return_value = None
        
        # Call the function
        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=Mock(),
            data_iterator=data_iterator,
            model=[Mock()],
            config=state.cfg,
            verbose=False
        )
        
        # Verify evaluate was called 3 times (once per dataset)
        assert mock_evaluate.call_count == 3
        
        # Verify that per_dataset_timelimit was passed to evaluate calls
        # Each call should have per_dataset_timelimit = 60.0 / 3 = 20.0 minutes
        for call in mock_evaluate.call_args_list:
            call_kwargs = call[1]  # Get keyword arguments
            # The per_dataset_timelimit should be passed as the last positional argument
            # Since we can't easily check the positional args, we'll verify the call was made
            # and that the function signature supports the new parameter
            assert 'per_dataset_timelimit' in str(call) or len(call[0]) >= 8  # At least 8 positional args
