import os
import json
import shutil
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file


def setup_target_view(source_dir: Path, target_dir: Path) -> None:
    """Create target view by removing MTP weights."""
    print(f"Setting up target view: {target_dir}")

    # Remove MTP weights
    for st_file in source_dir.glob("*.safetensors"):
        state_dict = load_file(st_file)
        clean_dict = {k: v for k, v in state_dict.items() if "mtp" not in k.lower()}
        if clean_dict:
            save_file(clean_dict, target_dir / st_file.name)
            print(f"  Saved {len(clean_dict)} weights (removed {len(state_dict) - len(clean_dict)} MTP weights)")

    # Copy non-weight files
    for file in source_dir.glob("*"):
        if file.suffix == ".safetensors" or file.is_dir():
            continue
        if file.name in ["target_view", "draft_view"]:
            continue
        shutil.copy(file, target_dir / file.name)

    # Update config for standard Qwen3
    config_path = target_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["model_type"] = "qwen3"
    config["architectures"] = ["Qwen3ForCausalLM"]
    config.pop("mtp_num_layers", None)
    config.pop("auto_map", None)
    if config.get("rope_scaling") is None:
        config.pop("rope_scaling", None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def setup_draft_view(source_dir: Path, draft_dir: Path, scripts_dir: Path) -> None:
    """Create draft view for MTP speculative decoding."""
    print(f"Setting up draft view: {draft_dir}")

    # Symlink weight files to share with source
    for file in source_dir.glob("*"):
        if file.is_dir() or file.name == "config.json":
            continue
        if file.name in ["target_view", "draft_view"]:
            continue
        os.symlink(file.absolute(), draft_dir / file.name)

    # Create MTP-specific config
    with open(source_dir / "config.json") as f:
        config = json.load(f)

    config["model_type"] = "mtp"
    config["architectures"] = ["Qwen3MTPModel"]
    config.setdefault("mtp_num_layers", 1)
    config["n_predict"] = config.get("mtp_num_layers", 1)
    config["num_nextn_predict_layers"] = config.get("mtp_num_layers", 1)
    config["auto_map"] = {
        "AutoConfig": "qwen3_mtp_config.Qwen3MTPConfig",
        "AutoModel": "vllm_qwen3_mtp.Qwen3MTPModel",
        "AutoModelForCausalLM": "vllm_qwen3_mtp.Qwen3MTPModel"
    }
    if config.get("rope_scaling") is None:
        config.pop("rope_scaling", None)

    with open(draft_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy Python modules to draft directory for trust_remote_code
    for pyfile in ["qwen3_mtp_config.py", "vllm_qwen3_mtp.py"]:
        src = scripts_dir / pyfile
        if src.exists():
            shutil.copy(src, draft_dir / pyfile)
            print(f"  Copied {pyfile}")


def main():
    parser = argparse.ArgumentParser(description="Setup model views for MTP inference")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Path to converted HuggingFace model")
    parser.add_argument("--scripts-dir", type=str, required=True,
                        help="Path to directory containing Python scripts")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    scripts_dir = Path(args.scripts_dir)

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    target_dir = source_dir / "target_view"
    draft_dir = source_dir / "draft_view"

    # Clean and create directories
    for d in [target_dir, draft_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    setup_target_view(source_dir, target_dir)
    setup_draft_view(source_dir, draft_dir, scripts_dir)

    print()
    print("Setup complete.")
    print(f"  Target: {target_dir}")
    print(f"  Draft: {draft_dir}")

    return 0


if __name__ == "__main__":
    exit(main())