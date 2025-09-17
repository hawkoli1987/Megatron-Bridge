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

"""Tests for evaluation functions."""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
from megatron.core.transformer import MegatronModule
from megatron.core.rerun_state_machine import RerunMode

from megatron.bridge.training.eval import evaluate
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig, TrainingConfig
from megatron.bridge.training.state import GlobalState, TrainState


class TestEvaluate:
    """Unit tests for evaluate function."""

    def _create_mock_global_state(self, eval_iters=2, exit_duration_in_mins=None):
        """Create a mock GlobalState for testing."""
        mock_state = Mock(spec=GlobalState)
        mock_state.train_state = Mock(spec=TrainState)
        mock_state.train_state.step = 100
        mock_state.train_state.consumed_valid_samples = 0
        mock_state.train_state.start_time = time.time()
        
        # Mock config
        mock_config = Mock(spec=ConfigContainer)
        mock_config.train = Mock(spec=TrainingConfig)
        mock_config.train.eval_iters = eval_iters
        mock_config.train.global_batch_size = 32
        mock_config.train.micro_batch_size = 8
        mock_config.train.data_parallel_size = 4
        mock_config.train.exit_duration_in_mins = exit_duration_in_mins
        mock_config.train.empty_unused_memory_level = 0
        mock_config.model = Mock()
        mock_config.model.seq_length = 512
        
        # Add data_parallel_size at the config level (not just train level)
        mock_config.data_parallel_size = 4
        mock_config.train.tensorboard_logging = True
        mock_config.train.wandb_logging = True
        
        mock_state.cfg = mock_config
        
        # Mock timers
        mock_state.timers = Mock()
        mock_timer = Mock()
        mock_timer.start = Mock()
        mock_timer.stop = Mock()
        mock_timer.elapsed = Mock(return_value=1.0)
        mock_state.timers.return_value = mock_timer
        
        return mock_state

    def _create_mock_model(self):
        """Create a mock model for testing."""
        mock_model = Mock(spec=MegatronModule)
        mock_model.eval = Mock()
        mock_model.train = Mock()
        return [mock_model]

    def _create_mock_data_iterator(self):
        """Create mock data iterator for testing."""
        return Mock()

    @patch('megatron.bridge.training.eval.get_forward_backward_func')
    @patch('megatron.bridge.training.eval.get_rerun_state_machine')
    @patch('megatron.bridge.training.eval.fault_tolerance')
    @patch('megatron.bridge.training.eval.parallel_state')
    @patch('megatron.bridge.training.eval.torch.distributed')
    @patch('megatron.bridge.training.eval.check_forward_step_func_num_args')
    @patch('megatron.bridge.training.eval.maybe_inject_state')
    def test_evaluate_single_dataset(self, mock_inject_state, mock_check_args, mock_dist, mock_parallel_state, 
                                   mock_fault_tolerance, mock_rerun_state_machine, mock_forward_backward_func):
        """Test basic evaluation with single validation dataset."""
        # Setup mocks
        mock_check_args.return_value = 3
        mock_inject_state.return_value = Mock()
        
        # Mock rerun state machine
        mock_rerun_sm = Mock()
        mock_rerun_sm.get_mode.return_value = "DISABLED"
        mock_rerun_state_machine.return_value = mock_rerun_sm
        
        # Mock fault tolerance
        mock_fault_tolerance.on_eval_step_start = Mock()
        mock_fault_tolerance.on_eval_step_end = Mock()
        
        # Mock parallel state
        mock_parallel_state.is_pipeline_last_stage.return_value = True
        mock_parallel_state.get_data_parallel_group.return_value = Mock()
        
        # Mock distributed operations
        mock_dist.all_reduce = Mock()
        
        # Mock forward backward function
        def mock_forward_backward_func_impl(*args, **kwargs):
            return [
                {"loss": torch.tensor([0.5, 1.0], device="cuda")},  # [numerator, denominator]
                {"loss": torch.tensor([0.3, 1.0], device="cuda")}
            ]
        mock_forward_backward_func.return_value = mock_forward_backward_func_impl
        
        # Create test data
        state = self._create_mock_global_state(eval_iters=2)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator()
        forward_step_func = Mock()
        config = state.cfg
        
        # Call the function
        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            state=state,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            process_non_loss_data_func=None,
            config=config,
            verbose=False
        )
        
        # Verify results
        assert timelimit is False
        assert collected_non_loss_data is None
        assert "loss" in total_loss_dict
        assert total_loss_dict["loss"].item() == pytest.approx(0.4, rel=1e-6)  # (0.5 + 0.3) / 2
        
        # Verify model was set to eval mode
        model[0].eval.assert_called()
        
        # Verify rerun state machine was disabled and restored
        mock_rerun_sm.set_mode.assert_any_call(RerunMode.DISABLED)
        mock_rerun_sm.set_mode.assert_any_call("DISABLED")
        
        # Verify fault tolerance was called
        mock_fault_tolerance.on_eval_step_start.assert_called()
        mock_fault_tolerance.on_eval_step_end.assert_called()

    @patch('megatron.bridge.training.eval.get_forward_backward_func')
    @patch('megatron.bridge.training.eval.get_rerun_state_machine')
    @patch('megatron.bridge.training.eval.fault_tolerance')
    @patch('megatron.bridge.training.eval.parallel_state')
    @patch('megatron.bridge.training.eval.torch.distributed')
    @patch('megatron.bridge.training.eval.check_forward_step_func_num_args')
    @patch('megatron.bridge.training.eval.maybe_inject_state')
    def test_evaluate_multiple_datasets(self, mock_inject_state, mock_check_args, mock_dist, mock_parallel_state,
                                      mock_fault_tolerance, mock_rerun_state_machine, mock_forward_backward_func):
        """Test evaluation with multiple validation datasets."""
        # Setup mocks
        mock_check_args.return_value = 3
        mock_inject_state.return_value = Mock()
        
        # Mock rerun state machine
        mock_rerun_sm = Mock()
        mock_rerun_sm.get_mode.return_value = "DISABLED"
        mock_rerun_state_machine.return_value = mock_rerun_sm
        
        # Mock fault tolerance
        mock_fault_tolerance.on_eval_step_start = Mock()
        mock_fault_tolerance.on_eval_step_end = Mock()
        
        # Mock parallel state
        mock_parallel_state.is_pipeline_last_stage.return_value = True
        mock_parallel_state.get_data_parallel_group.return_value = Mock()
        
        # Mock distributed operations
        mock_dist.all_reduce = Mock()
        
        # Mock forward backward function
        def mock_forward_backward_func_impl(*args, **kwargs):
            return [
                {"loss": torch.tensor([0.5, 1.0], device="cuda")},
                {"loss": torch.tensor([0.3, 1.0], device="cuda")}
            ]
        mock_forward_backward_func.return_value = mock_forward_backward_func_impl
        
        # Create test data
        state = self._create_mock_global_state(eval_iters=2)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator()
        forward_step_func = Mock()
        config = state.cfg
        
        # Call the function (same as single dataset, but this tests the core evaluate function)
        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            state=state,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            process_non_loss_data_func=None,
            config=config,
            verbose=False
        )
        
        # Verify results
        assert timelimit is False
        assert collected_non_loss_data is None
        assert "loss" in total_loss_dict
        assert total_loss_dict["loss"].item() == pytest.approx(0.4, rel=1e-6)

    @patch('megatron.bridge.training.eval.get_forward_backward_func')
    @patch('megatron.bridge.training.eval.get_rerun_state_machine')
    @patch('megatron.bridge.training.eval.fault_tolerance')
    @patch('megatron.bridge.training.eval.parallel_state')
    @patch('megatron.bridge.training.eval.torch.distributed')
    @patch('megatron.bridge.training.eval.check_forward_step_func_num_args')
    @patch('megatron.bridge.training.eval.maybe_inject_state')
    def test_evaluate_timelimit_handling(self, mock_inject_state, mock_check_args, mock_dist, mock_parallel_state,
                                       mock_fault_tolerance, mock_rerun_state_machine, mock_forward_backward_func):
        """Test time limit functionality during evaluation."""
        # Setup mocks
        mock_check_args.return_value = 3
        mock_inject_state.return_value = Mock()
        
        # Mock rerun state machine
        mock_rerun_sm = Mock()
        mock_rerun_sm.get_mode.return_value = "DISABLED"
        mock_rerun_state_machine.return_value = mock_rerun_sm
        
        # Mock fault tolerance
        mock_fault_tolerance.on_eval_step_start = Mock()
        mock_fault_tolerance.on_eval_step_end = Mock()
        
        # Mock parallel state
        mock_parallel_state.is_pipeline_last_stage.return_value = True
        mock_parallel_state.get_data_parallel_group.return_value = Mock()
        
        # Mock distributed operations
        mock_dist.all_reduce = Mock()
        
        # Mock forward backward function to simulate timelimit
        def mock_forward_backward_func_impl(*args, **kwargs):
            # Simulate that we're past the timelimit by modifying the state
            # We need to access the state from the global scope since it's not passed as an argument
            # Let's modify the state before calling the function
            return [
                {"loss": torch.tensor([0.5, 1.0], device="cuda")}
            ]
        mock_forward_backward_func.return_value = mock_forward_backward_func_impl
        
        # Create test data with short exit duration
        state = self._create_mock_global_state(eval_iters=10, exit_duration_in_mins=0.001)  # Very short duration
        # Simulate that we're past the timelimit by modifying the start time
        state.train_state.start_time = time.time() - (0.002 * 60)  # 0.002 minutes ago
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator()
        forward_step_func = Mock()
        config = state.cfg
        
        # Call the function
        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            state=state,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            process_non_loss_data_func=None,
            config=config,
            verbose=False
        )
        
        # Verify timelimit was hit
        assert timelimit is True
        assert total_loss_dict is None
        assert collected_non_loss_data is None

    @patch('megatron.bridge.training.eval.get_forward_backward_func')
    @patch('megatron.bridge.training.eval.get_rerun_state_machine')
    @patch('megatron.bridge.training.eval.fault_tolerance')
    @patch('megatron.bridge.training.eval.parallel_state')
    @patch('megatron.bridge.training.eval.torch.distributed')
    @patch('megatron.bridge.training.eval.check_forward_step_func_num_args')
    @patch('megatron.bridge.training.eval.maybe_inject_state')
    def test_evaluate_non_loss_data_collection(self, mock_inject_state, mock_check_args, mock_dist, mock_parallel_state,
                                             mock_fault_tolerance, mock_rerun_state_machine, mock_forward_backward_func):
        """Test non-loss data collection during evaluation."""
        # Setup mocks
        mock_check_args.return_value = 3
        mock_inject_state.return_value = Mock()
        
        # Mock rerun state machine
        mock_rerun_sm = Mock()
        mock_rerun_sm.get_mode.return_value = "DISABLED"
        mock_rerun_state_machine.return_value = mock_rerun_sm
        
        # Mock fault tolerance
        mock_fault_tolerance.on_eval_step_start = Mock()
        mock_fault_tolerance.on_eval_step_end = Mock()
        
        # Mock parallel state
        mock_parallel_state.is_pipeline_last_stage.return_value = True
        mock_parallel_state.get_data_parallel_group.return_value = Mock()
        
        # Mock distributed operations
        mock_dist.all_reduce = Mock()
        
        # Mock forward backward function
        def mock_forward_backward_func_impl(*args, **kwargs):
            return [
                {"loss": torch.tensor([0.5, 1.0], device="cuda")}
            ]
        mock_forward_backward_func.return_value = mock_forward_backward_func_impl
        
        # Create test data
        state = self._create_mock_global_state(eval_iters=1)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator()
        forward_step_func = Mock()
        config = state.cfg
        
        # Mock non-loss data function
        non_loss_data_func = Mock(return_value={"accuracy": 0.95})
        
        # Call the function
        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            state=state,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            process_non_loss_data_func=None,
            config=config,
            verbose=False,
            non_loss_data_func=non_loss_data_func
        )
        
        # Verify results
        assert timelimit is False
        assert collected_non_loss_data == {"accuracy": 0.95}
        assert "loss" in total_loss_dict
        assert total_loss_dict["loss"].item() == pytest.approx(0.5, rel=1e-6)
        
        # Verify non-loss data function was called
        non_loss_data_func.assert_called_once_with(model)

    @patch('megatron.bridge.training.eval.get_forward_backward_func')
    @patch('megatron.bridge.training.eval.get_rerun_state_machine')
    @patch('megatron.bridge.training.eval.fault_tolerance')
    @patch('megatron.bridge.training.eval.parallel_state')
    @patch('megatron.bridge.training.eval.torch.distributed')
    @patch('megatron.bridge.training.eval.check_forward_step_func_num_args')
    @patch('megatron.bridge.training.eval.maybe_inject_state')
    def test_evaluate_memory_management(self, mock_inject_state, mock_check_args, mock_dist, mock_parallel_state,
                                      mock_fault_tolerance, mock_rerun_state_machine, mock_forward_backward_func):
        """Test memory management during evaluation."""
        # Setup mocks
        mock_check_args.return_value = 3
        mock_inject_state.return_value = Mock()
        
        # Mock rerun state machine
        mock_rerun_sm = Mock()
        mock_rerun_sm.get_mode.return_value = "DISABLED"
        mock_rerun_state_machine.return_value = mock_rerun_sm
        
        # Mock fault tolerance
        mock_fault_tolerance.on_eval_step_start = Mock()
        mock_fault_tolerance.on_eval_step_end = Mock()
        
        # Mock parallel state
        mock_parallel_state.is_pipeline_last_stage.return_value = True
        mock_parallel_state.get_data_parallel_group.return_value = Mock()
        
        # Mock distributed operations
        mock_dist.all_reduce = Mock()
        
        # Mock forward backward function
        def mock_forward_backward_func_impl(*args, **kwargs):
            return [
                {"loss": torch.tensor([0.5, 1.0], device="cuda")}
            ]
        mock_forward_backward_func.return_value = mock_forward_backward_func_impl
        
        # Create test data with memory management enabled
        state = self._create_mock_global_state(eval_iters=1)
        state.cfg.train.empty_unused_memory_level = 1  # Enable memory cleanup
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator()
        forward_step_func = Mock()
        config = state.cfg
        
        # Mock torch.cuda.empty_cache
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            # Call the function
            total_loss_dict, collected_non_loss_data, timelimit = evaluate(
                state=state,
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                process_non_loss_data_func=None,
                config=config,
                verbose=False
            )
            
            # Verify memory cleanup was called
            mock_empty_cache.assert_called()
        
        # Verify results
        assert timelimit is False
        assert collected_non_loss_data is None
        assert "loss" in total_loss_dict
