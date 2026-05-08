"""
Training wrapper for QLoRA fine-tuning with mlx-lm.

Reads a YAML config, verifies data files, and runs mlx_lm.lora.

Usage:
    python scripts/train.py --config configs/light.yaml
    python scripts/train.py --config configs/quality.yaml
    python scripts/train.py --config configs/light.yaml --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description="QLoRA training wrapper")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML training config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Validate required config keys
    required_keys = ["model", "data", "adapter_path"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        print(f"ERROR: Config missing required keys: {', '.join(missing)}")
        sys.exit(1)

    # Resolve data path relative to finetune/ directory
    finetune_dir = Path(__file__).resolve().parent.parent
    data_dir = finetune_dir / config["data"]
    adapter_path = finetune_dir / config["adapter_path"]

    # Verify data files
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists():
        print(f"ERROR: Training data not found: {train_file}")
        print("Run prepare_data.py first.")
        sys.exit(1)

    train_count = count_lines(train_file)
    valid_count = count_lines(valid_file) if valid_file.exists() else 0
    if not valid_file.exists():
        print(
            "WARNING: No valid.jsonl found; training will proceed without validation."
        )
    print(f"Data: {train_count} train, {valid_count} valid samples")
    print(f"Model: {config['model']}")
    print(f"Adapters: {adapter_path}")

    # Build mlx_lm lora command
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        config["model"],
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
    ]

    # Add training params from config
    param_map = {
        "iters": "--iters",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "seed": "--seed",
        "num_layers": "--num-layers",
        "max_seq_length": "--max-seq-length",
        "val_batches": "--val-batches",
        "steps_per_eval": "--steps-per-eval",
        "steps_per_report": "--steps-per-report",
        "save_every": "--save-every",
    }

    for key, flag in param_map.items():
        if key in config:
            cmd.extend([flag, str(config[key])])

    # LoRA parameters: pass config file via -c if lora_rank is specified
    # mlx-lm reads lora_parameters from the YAML config
    if "lora_rank" in config:
        lora_config = {
            "lora_parameters": {
                "rank": config["lora_rank"],
                "dropout": config.get("lora_dropout", 0.0),
                "scale": config.get("lora_scale", 20.0),
            }
        }
        lora_config_path = adapter_path / "lora_params.yaml"
        adapter_path.mkdir(parents=True, exist_ok=True)
        with open(lora_config_path, "w") as lf:
            yaml.dump(lora_config, lf)
        cmd.extend(["-c", str(lora_config_path)])

    # Boolean flags
    if config.get("mask_prompt"):
        cmd.append("--mask-prompt")
    if config.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")

    print(f"\nCommand: {' '.join(cmd)}\n")

    if args.dry_run:
        print("(dry run, not executing)")
        return

    # Create adapter output directory
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Run training
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
