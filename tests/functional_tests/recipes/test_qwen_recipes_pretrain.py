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

"""Functional smoke tests for Qwen recipe configurations."""

import pytest

from megatron.bridge.recipes.qwen import (
    qwen2_500m_pretrain_config as qwen2_500m_config,
)
from megatron.bridge.recipes.qwen import (
    qwen25_500m_pretrain_config as qwen25_500m_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


QWEN_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides)
    (qwen2_500m_config, "qwen2_500m", {}),
    (qwen25_500m_config, "qwen25_500m", {}),
]


class TestQwenRecipes:
    """Test class for Qwen recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides", QWEN_PRETRAIN_RECIPES)
    def test_qwen_pretrain_recipes(self, config_func, recipe_name, parallelism_overrides, tmp_path):
        """Functional test for Qwen recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(config_func, recipe_name, tmp_path, **parallelism_overrides)
