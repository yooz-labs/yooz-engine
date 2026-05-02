"""
Fuse LoRA adapters into the base model.

Merges the trained adapter weights into the base model to produce
a standalone fine-tuned model. Runs a sanity check after fusion.

Usage:
    python scripts/fuse_model.py --config configs/light.yaml --save-path outputs/light/fused
    python scripts/fuse_model.py --config configs/quality.yaml --save-path outputs/quality/fused
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into base model")
    parser.add_argument(
        "--config", type=Path, required=True, help="Training config YAML"
    )
    parser.add_argument(
        "--save-path", type=Path, required=True, help="Output directory for fused model"
    )
    parser.add_argument(
        "--de-quantize",
        action="store_true",
        help="De-quantize the model (produces larger full-precision model)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    finetune_dir = Path(__file__).resolve().parent.parent
    adapter_path = finetune_dir / config["adapter_path"]
    model_id = config["model"]

    if not adapter_path.exists():
        print(f"ERROR: Adapter not found: {adapter_path}")
        print("Train the model first.")
        sys.exit(1)

    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {args.save_path}")

    # Build fuse command
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(args.save_path),
    ]

    if args.de_quantize:
        cmd.append("--de-quantize")
        print("De-quantizing (full precision output)")

    print(f"\nCommand: {' '.join(cmd)}\n")

    args.save_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ERROR: Fusion failed")
        sys.exit(result.returncode)

    # Report model size
    total_size = sum(f.stat().st_size for f in args.save_path.rglob("*") if f.is_file())
    print(f"\nFused model size: {total_size / (1024 * 1024):.0f} MB")

    # Sanity check: try loading and generating
    print("\nRunning sanity check...")
    import mlx_lm

    model, tokenizer = mlx_lm.load(str(args.save_path))

    # Use the actual training system prompt for the sanity check
    system_prompt = (
        "Fix grammar, capitalize properly, and convert spoken numbers to digits. "
        'Return the fixed text as JSON: {"result": "corrected text"}.'
    )
    test_inputs = [
        "the meeting is tomorrow at three pm",
        "we need about fifty units ready",
        "i think we should prepare for the presentation",
    ]

    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.1)
    for text in test_inputs:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=128, sampler=sampler
        )
        try:
            data = json.loads(response.strip())
            result_text = data.get("result", response)
        except json.JSONDecodeError:
            result_text = response.strip()

        print(f"  Input:  {text}")
        print(f"  Output: {result_text}")
        print()

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
