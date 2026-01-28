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
Megatron-HuggingFace Checkpoint Conversion Example

This script demonstrates how to convert models between HuggingFace and Megatron formats
using the AutoBridge import_ckpt and export_ckpt methods.

Features:
- Import HuggingFace models to Megatron checkpoint format
- Export Megatron checkpoints to HuggingFace format
- Support for various model architectures (GPT, Llama, etc.)
- Configurable model and conversion settings

Usage examples:
  # Import a HuggingFace model to Megatron format
  python examples/conversion/convert_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b

  # Export a Megatron checkpoint to HuggingFace format
  python examples/conversion/convert_checkpoints.py export \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b \
    --hf-path ./exports/llama3_2_1b_hf

  # Import with custom settings
  python examples/conversion/convert_checkpoints.py import \
    --hf-model ./local_model \
    --megatron-path ./checkpoints/custom_model \
    --torch-dtype bfloat16 \
    --device-map auto

  # Export without progress bar (useful for scripting)
  python examples/conversion/convert_checkpoints.py export \
    --hf-model ./local_model \
    --megatron-path ./checkpoints/custom_model \
    --hf-path ./exports/custom_model_hf \
    --no-progress
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from megatron.bridge import AutoBridge
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from megatron.bridge.models.qwen.qwen3_mtp_bridge import Qwen3MTPBridge
except ImportError:
    print("Qwen3MTPBridge not found. MTP conversion will fail.")


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """
    Import a HuggingFace model and save it as a Megatron checkpoint.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        torch_dtype: Model precision ("float32", "float16", "bfloat16")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        trust_remote_code: Allow custom model code execution
    """
    print(f"🔄 Starting import: {hf_model} -> {megatron_path}")

    # Prepare kwargs
    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print(f"   Using device_map: {device_map}")

    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
        print(f"   Trust remote code: {trust_remote_code}")

    # Import using the convenience method
    print(f"📥 Loading HuggingFace model: {hf_model}")
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        **kwargs,
    )

    print(f"✅ Successfully imported model to: {megatron_path}")

    # Verify the checkpoint was created
    checkpoint_path = Path(megatron_path)
    if checkpoint_path.exists():
        print("📁 Checkpoint structure:")
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")


# Modify the following function as such
def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    trust_remote_code: bool = False,
) -> None:
    print(f"🔄 Starting export: {megatron_path} -> {hf_path}")
    # Validate megatron checkpoint exists
    checkpoint_path = validate_path(megatron_path, must_exist=True)
    # Custom Model Loading & Registration
    dummy_model = None
    if trust_remote_code:
        print(f"📥 Loading target HF model config from: {hf_model}")
        try:
            # Load config
            config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
            with torch.device("meta"):
                dummy_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model_class = type(dummy_model)
            print(f"✅ Detected Custom Class: {model_class.__name__}")
            # Register the bridge
            if "Qwen3MTPBridge" in globals():
                Qwen3MTPBridge.register_for_custom_class(model_class)
                print(f"🔗 Registered Qwen3MTPBridge for {model_class.__name__}")
        except Exception as e:
            print(f"⚠️ Failed to pre-load custom class: {e}")
    # Create the Bridge
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=trust_remote_code)
    # Monkey-Patch for New Architecture by injecting keys from dummy_model
    if dummy_model is not None:
        try:
            print("🛠️   Patching bridge (ignoring missing weights on disk)...")
            bridge.hf_pretrained.state.source.get_all_keys = lambda: list(dummy_model.state_dict().keys())
        except Exception as e:
            print(f"⚠️ Failed to patch state source: {e}")
    # Run Export
    print("📤 Exporting to HuggingFace format...")
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
    )
    print(f"✅ Successfully exported model to: {hf_path}")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand (HF -> Megatron)
    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to Megatron checkpoint format")
    import_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    import_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved"
    )
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    import_parser.add_argument("--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")')
    import_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")

    # Export subcommand (Megatron -> HF)
    export_parser = subparsers.add_parser("export", help="Export Megatron checkpoint to HuggingFace format")
    export_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    export_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint is stored"
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace model will be saved"
    )
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar during export")
    export_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
