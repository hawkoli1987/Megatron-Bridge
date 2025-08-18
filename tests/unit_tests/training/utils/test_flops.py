#!/usr/bin/env python3
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

"""Tests for flop_utils module."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

from megatron.bridge.training.utils.flop_utils import num_floating_point_operations
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.models import GPTModelProvider, T5ModelProvider
from megatron.bridge.training.tokenizers.config import TokenizerConfig


class TestFlops:
    #@pytest.mark.parametrize("model_provider", [GPTModelProvider, T5ModelProvider])
    def test_llama31_8b_flops(self, model_provider):
        import megatron.bridge.recipes.llama31_8b as llama31_8b
        cfg = llama31_8b.get_config()

        num_ops = num_floating_point_operations(cfg, batch_size=1)
        print(num_ops)
        assert num_ops == 144*1e12