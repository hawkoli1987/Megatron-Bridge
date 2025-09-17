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

"""Unit tests for multiple validation sets data loaders."""

from unittest.mock import Mock, patch, MagicMock
import pytest
import torch

from megatron.bridge.training.config import MockGPTDatasetConfig, TrainingConfig
from megatron.bridge.training.train import TrainState
from megatron.bridge.data.loaders import build_train_valid_test_data_loaders, build_train_valid_test_data_iterators


class TestMultipleValidationDataLoaders:
    """Unit tests for multiple validation sets data loaders."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_train_dataset = Mock()
        self.mock_valid_dataset_1 = Mock()
        self.mock_valid_dataset_2 = Mock()
        self.mock_valid_dataset_3 = Mock()
        self.mock_test_dataset = Mock()
        
        # Mock train state
        self.mock_train_state = Mock(spec=TrainState)
        self.mock_train_state.consumed_valid_samples = 0

    def create_mock_config(self, multiple_validation_sets=False, skip_train=False):
        """Create a mock configuration for testing."""
        config = Mock()
        config.dataset = MockGPTDatasetConfig(
            multiple_validation_sets=multiple_validation_sets,
            dataloader_type="single",
            micro_batch_size=2,
            num_workers=1,
            data_sharding=True,
            pin_memory=False,
            persistent_workers=False
        )
        config.train = Mock()
        config.train.skip_train = skip_train
        config.train.micro_batch_size = 2
        return config

    def create_mock_datasets_provider(self, multiple_validation_sets=False):
        """Create a mock datasets provider function."""
        def mock_provider(cfg, train_state):
            if multiple_validation_sets:
                valid_datasets = [
                    self.mock_valid_dataset_1,
                    self.mock_valid_dataset_2,
                    self.mock_valid_dataset_3
                ]
            else:
                valid_datasets = self.mock_valid_dataset_1
            
            return (
                self.mock_train_dataset,
                valid_datasets,
                self.mock_test_dataset
            )
        return mock_provider

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_single_validation_dataloader_unchanged(self, mock_build_loader):
        """Test that single validation dataset behavior is unchanged."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=False)
        mock_provider = self.create_mock_datasets_provider(multiple_validation_sets=False)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider
        )
        
        # Verify
        assert valid_dl is not None
        assert not isinstance(valid_dl, list)  # Should be single dataloader, not list
        assert valid_dl == mock_dataloader
        
        # Verify build_pretraining_data_loader was called correctly
        mock_build_loader.assert_called_once()
        call_args = mock_build_loader.call_args
        assert call_args[0][0] == self.mock_valid_dataset_1  # dataset
        assert call_args[0][1] == 0  # consumed_valid_samples
        assert call_args[1]['dataloader_type'] == "single"

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_multiple_validation_dataloaders_creation(self, mock_build_loader):
        """Test creation of multiple validation dataloaders when enabled."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        mock_provider = self.create_mock_datasets_provider(multiple_validation_sets=True)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider
        )
        
        # Verify
        assert valid_dl is not None
        assert isinstance(valid_dl, list)  # Should be list of dataloaders
        assert len(valid_dl) == 3  # Should have 3 dataloaders
        assert all(dl == mock_dataloader for dl in valid_dl)  # All should be the same mock
        
        # Verify build_pretraining_data_loader was called for each dataset
        assert mock_build_loader.call_count == 3

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_validation_dataloader_with_none_elements(self, mock_build_loader):
        """Test handling of None elements in validation dataset list."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        def mock_provider_with_none(cfg, train_state):
            valid_datasets = [
                self.mock_valid_dataset_1,
                None,  # None element
                self.mock_valid_dataset_3
            ]
            return (self.mock_train_dataset, valid_datasets, self.mock_test_dataset)
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider_with_none
        )
        
        # Verify
        assert valid_dl is not None
        assert isinstance(valid_dl, list)
        assert len(valid_dl) == 3
        assert valid_dl[0] == mock_dataloader  # First dataset -> dataloader
        assert valid_dl[1] is None  # None dataset -> None dataloader
        assert valid_dl[2] == mock_dataloader  # Third dataset -> dataloader
        
        # Verify build_pretraining_data_loader was called only for non-None datasets
        assert mock_build_loader.call_count == 2

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_validation_dataloader_skip_train_mode(self, mock_build_loader):
        """Test dataloader creation in skip_train mode with multiple validation sets."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True, skip_train=True)
        mock_provider = self.create_mock_datasets_provider(multiple_validation_sets=True)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider
        )
        
        # Verify
        assert isinstance(valid_dl, list)
        assert len(valid_dl) == 3
        
        # Verify all calls used skip_train parameters
        for call in mock_build_loader.call_args_list:
            call_args = call[0]
            call_kwargs = call[1]
            assert call_args[1] == 0  # consumed_valid_samples should be 0
            assert call_kwargs['dataloader_type'] == "single"

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_validation_dataloader_iterator_creation_single(self, mock_build_loader):
        """Test creation of data iterators for single validation dataset."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=False)
        mock_provider = self.create_mock_datasets_provider(multiple_validation_sets=False)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        with patch('megatron.bridge.data.loaders._get_iterator') as mock_get_iterator:
            mock_iterator = Mock()
            mock_get_iterator.return_value = mock_iterator
            
            # Execute
            train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
                config, self.mock_train_state, mock_provider
            )
            
            # Verify
            assert valid_iter is not None
            assert not isinstance(valid_iter, list)  # Should be single iterator, not list
            assert valid_iter == mock_iterator
            
            # Verify _get_iterator was called
            mock_get_iterator.assert_called_once_with("cyclic", mock_dataloader)

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_validation_dataloader_iterator_creation_multiple(self, mock_build_loader):
        """Test creation of data iterators for multiple validation datasets."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        mock_provider = self.create_mock_datasets_provider(multiple_validation_sets=True)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        with patch('megatron.bridge.data.loaders._get_iterator') as mock_get_iterator:
            mock_iterator = Mock()
            mock_get_iterator.return_value = mock_iterator
            
            # Execute
            train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
                config, self.mock_train_state, mock_provider
            )
            
            # Verify
            assert valid_iter is not None
            assert isinstance(valid_iter, list)  # Should be list of iterators
            assert len(valid_iter) == 3
            assert all(iter == mock_iterator for iter in valid_iter)
            
            # Verify _get_iterator was called for each dataloader
            assert mock_get_iterator.call_count == 3

    @patch('megatron.bridge.data.loaders.build_pretraining_data_loader')
    def test_validation_dataloader_iterator_with_none_elements(self, mock_build_loader):
        """Test iterator creation with None dataloaders."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        mock_dataloader = Mock()
        mock_build_loader.return_value = mock_dataloader
        
        def mock_provider_with_none(cfg, train_state):
            valid_datasets = [self.mock_valid_dataset_1, None, self.mock_valid_dataset_3]
            return (self.mock_train_dataset, valid_datasets, self.mock_test_dataset)
        
        with patch('megatron.bridge.data.loaders._get_iterator') as mock_get_iterator:
            mock_iterator = Mock()
            mock_get_iterator.return_value = mock_iterator
            
            # Execute
            train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
                config, self.mock_train_state, mock_provider_with_none
            )
            
            # Verify
            assert isinstance(valid_iter, list)
            assert len(valid_iter) == 3
            assert valid_iter[0] == mock_iterator  # First dataloader -> iterator
            assert valid_iter[1] is None  # None dataloader -> None iterator
            assert valid_iter[2] == mock_iterator  # Third dataloader -> iterator
            
            # Verify _get_iterator was called only for non-None dataloaders
            assert mock_get_iterator.call_count == 2

    def test_validation_dataloader_empty_list(self):
        """Test handling of empty validation dataset list."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        
        def mock_provider_empty(cfg, train_state):
            return (self.mock_train_dataset, [], self.mock_test_dataset)
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider_empty
        )
        
        # Verify
        assert valid_dl is not None
        assert isinstance(valid_dl, list)
        assert len(valid_dl) == 0

    def test_validation_dataloader_all_none(self):
        """Test handling when all validation datasets are None."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        
        def mock_provider_all_none(cfg, train_state):
            return (self.mock_train_dataset, [None, None, None], self.mock_test_dataset)
        
        # Execute
        train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
            config, self.mock_train_state, mock_provider_all_none
        )
        
        # Verify
        assert valid_dl is not None
        assert isinstance(valid_dl, list)
        assert len(valid_dl) == 3
        assert all(dl is None for dl in valid_dl)

    def test_validation_dataloader_single_dataset_in_list(self):
        """Test handling when only one dataset is provided in the list."""
        # Setup
        config = self.create_mock_config(multiple_validation_sets=True)
        
        def mock_provider_single_in_list(cfg, train_state):
            return (self.mock_train_dataset, [self.mock_valid_dataset_1], self.mock_test_dataset)
        
        with patch('megatron.bridge.data.loaders.build_pretraining_data_loader') as mock_build_loader:
            mock_dataloader = Mock()
            mock_build_loader.return_value = mock_dataloader
            
            # Execute
            train_dl, valid_dl, test_dl = build_train_valid_test_data_loaders(
                config, self.mock_train_state, mock_provider_single_in_list
            )
            
            # Verify
            assert valid_dl is not None
            assert isinstance(valid_dl, list)
            assert len(valid_dl) == 1
            assert valid_dl[0] == mock_dataloader