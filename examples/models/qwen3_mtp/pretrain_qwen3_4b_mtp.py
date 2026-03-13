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

"""
Qwen3-4B MTP Retrofit Pre-training Script.

Two-phase training to retrofit Multi-Token Prediction onto Qwen3-4B:

  Phase 1 — MTP Warmup (backbone frozen):
    torchrun --nproc_per_node=8 pretrain_qwen3_4b_mtp.py \
      --config-file conf/qwen3_4b_mtp_warmup.yaml \
      --hf-pretrained-checkpoint Qwen/Qwen3-4B \
      --freeze-non-mtp

  Phase 2 — Joint Training (all params):
    torchrun --nproc_per_node=8 pretrain_qwen3_4b_mtp.py \
      --config-file conf/qwen3_4b_mtp_joint.yaml \
      --pretrained-checkpoint /path/to/warmup/checkpoint

Supports YAML config files and Hydra-style CLI overrides (key=value).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen.qwen3 import qwen3_4b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).parent.resolve()


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Qwen3-4B MTP retrofit training with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML OmegaConf override file (e.g. conf/qwen3_4b_mtp_warmup.yaml)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to Megatron checkpoint directory (e.g. warmup checkpoint for Phase 2).",
    )
    parser.add_argument(
        "--hf-pretrained-checkpoint",
        type=str,
        default=None,
        help=(
            "HuggingFace model name or path for initial weight loading (Phase 1). "
            "MTP layers are left randomly initialized."
        ),
    )
    parser.add_argument(
        "--freeze-non-mtp",
        action="store_true",
        help="Freeze all non-MTP parameters (used during Phase 1 warmup).",
    )
    parser.add_argument(
        "--mtp-num-layers",
        type=int,
        default=1,
        help="Number of MTP prediction layers (default: 1).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=None,
        help="Tokenized dataset prefix path(s) for Megatron data loading.",
    )
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = qwen3_4b_pretrain_config()

    # MTP configuration
    cfg.model.mtp_enabled = True
    cfg.model.mtp_num_layers = args.mtp_num_layers

    if args.freeze_non_mtp:
        cfg.model.freeze_non_mtp = True

    if args.hf_pretrained_checkpoint:
        cfg.checkpoint.hf_pretrained_checkpoint = args.hf_pretrained_checkpoint

    if args.pretrained_checkpoint:
        cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint

    if args.data_path:
        cfg.dataset.blend = (args.data_path, None)
        cfg.dataset.split = "9999,8,2"

    # Convert to OmegaConf for YAML/CLI merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides)

    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides, excluded_fields)

    # Remove CLI overrides from sys.argv so Megatron-Core's internal parser
    # doesn't misinterpret them (e.g., as data paths).
    sys.argv = [sys.argv[0]]

    if get_rank_safe() == 0:
        cfg.print_yaml()

    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
