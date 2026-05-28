"""Split exported NemotronH-MTP HF checkpoint into target_view + draft_view
for vLLM speculative decoding.

target_view: backbone only (model_type=nemotron_h, NemotronHForCausalLM).
             Served by vLLM's native nemotron_h.py model class.

draft_view:  MTP heads only (model_type=nemotron_h_mtp, NemotronHMTPModel).
             Served by vLLM's nemotron_h_mtp.py model class. Requires the
             patched vllm clone on PYTHONPATH so the '-' (MLP) pattern char
             is supported (see /mnt/weka/aisg/source_files/vllm_yuli).

Usage:
    python setup_views.py --source-dir /mnt/weka/aisg/ckpt/mb/mtp/nemotron3_nano_4b_mtp/exported
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def setup_target_view(source_dir: Path, target_dir: Path) -> None:
    """target_view = backbone only (no MTP layers)."""
    print(f"Setting up target view: {target_dir}")

    for st_file in source_dir.glob("*.safetensors"):
        state_dict = load_file(st_file)
        clean_dict = {k: v for k, v in state_dict.items() if "mtp" not in k.lower()}
        if clean_dict:
            save_file(clean_dict, target_dir / st_file.name)
            print(
                f"  Saved {len(clean_dict)} weights "
                f"(removed {len(state_dict) - len(clean_dict)} MTP weights)"
            )

    # Copy non-weight files (configs, tokenizer, modeling files)
    for file in source_dir.glob("*"):
        if file.suffix == ".safetensors" or file.is_dir():
            continue
        if file.name in ("target_view", "draft_view"):
            continue
        shutil.copy(file, target_dir / file.name)

    # Strip MTP markers; preserve auto_map so AutoConfig can use the snapshot's
    # `-`-aware NemotronHConfig (upstream transformers 5.3.0 doesn't parse '-').
    config_path = target_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["model_type"] = "nemotron_h"
    config["architectures"] = ["NemotronHForCausalLM"]
    # Keep mtp_* fields on the TARGET config: vLLM's NemotronHMultiTokenPredictor
    # reads `vllm_config.model_config.hf_config` (the TARGET's hf_config) when
    # constructing the draft. So mtp_hybrid_override_pattern + num_nextn_predict_layers
    # must be present on the target's hf_config or speculator construction fails.
    config["num_nextn_predict_layers"] = config.get("mtp_num_layers", 1)
    config["auto_map"] = {
        "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def setup_draft_view(source_dir: Path, draft_dir: Path) -> None:
    """draft_view = MTP heads only, served as NemotronHMTPModel."""
    print(f"Setting up draft view: {draft_dir}")

    # Symlink everything except config.json (we rewrite that)
    for file in source_dir.glob("*"):
        if file.is_dir() or file.name == "config.json":
            continue
        if file.name in ("target_view", "draft_view"):
            continue
        os.symlink(file.absolute(), draft_dir / file.name)

    with open(source_dir / "config.json") as f:
        config = json.load(f)

    # vLLM expects num_nextn_predict_layers == 1 for NemotronHMTPModel
    config["model_type"] = "nemotron_h_mtp"
    config["architectures"] = ["NemotronHMTPModel"]
    config["num_nextn_predict_layers"] = 1
    config["mtp_num_layers"] = config.get("mtp_num_layers", 1)
    config["mtp_hybrid_override_pattern"] = config.get(
        "mtp_hybrid_override_pattern", "*-"
    )
    # Point AutoConfig at our NemotronHMTPConfig subclass which explicitly
    # stores mtp_hybrid_override_pattern; the snapshot's NemotronHConfig
    # accepts `-` in patterns but doesn't materialize the MTP-specific fields.
    config["auto_map"] = {
        "AutoConfig": "modeling_nemotron_3_nano_4b_mtp.NemotronHMTPConfig",
    }

    with open(draft_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup target_view + draft_view")
    parser.add_argument("--source-dir", type=str, required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    target_dir = source_dir / "target_view"
    draft_dir = source_dir / "draft_view"

    for d in (target_dir, draft_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    setup_target_view(source_dir, target_dir)
    setup_draft_view(source_dir, draft_dir)

    print()
    print("Setup complete.")
    print(f"  Target: {target_dir}")
    print(f"  Draft:  {draft_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
