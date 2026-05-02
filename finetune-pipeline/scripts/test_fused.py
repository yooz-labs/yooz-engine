"""Quick test of fused model against test set using best config (few_shot, t=0.0)."""

import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import mlx_lm
from mlx_lm.sample_utils import make_sampler

# Import prompts from experiment.py
sys.path.insert(0, str(Path(__file__).parent))
from experiment import PROMPTS, detect_mode, parse_result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test fused model")
    parser.add_argument("--fused-path", type=Path, required=True)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--model-type", choices=["light", "quality"], required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-variant",
        default="few_shot",
        choices=["few_shot", "zero_shot", "minimal"],
    )
    args = parser.parse_args()

    print(f"Loading fused model: {args.fused_path}")
    model, tokenizer = mlx_lm.load(str(args.fused_path))
    sampler = make_sampler(temp=args.temperature)

    # Load test data
    samples = []
    with open(args.test_set) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    prompt_variant = PROMPTS[args.model_type][args.prompt_variant]
    print(f"Model type: {args.model_type}")
    print(f"Prompt: {args.prompt_variant}, Temperature: {args.temperature}")
    print(f"Test samples: {len(samples)}")
    print()

    json_ok = 0
    exact = 0
    total_sim = 0.0
    total_lat = 0.0
    mode_stats = {
        "standard": {"n": 0, "json": 0, "exact": 0, "sim": 0.0},
        "full": {"n": 0, "json": 0, "exact": 0, "sim": 0.0},
    }

    for i, s in enumerate(samples):
        msgs = s["messages"]
        user_input = msgs[1]["content"]
        expected_raw = msgs[2]["content"]
        mode = detect_mode(msgs[0]["content"])

        expected = parse_result(expected_raw)
        if expected is None:
            continue

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt_variant[mode]},
                {"role": "user", "content": user_input},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        start = time.perf_counter()
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler
        )
        lat = (time.perf_counter() - start) * 1000
        total_lat += lat

        parsed = parse_result(response)
        is_json = parsed is not None
        if is_json:
            json_ok += 1

        output = parsed or response.strip()
        if output == expected:
            exact += 1

        sim = SequenceMatcher(None, output.lower(), expected.lower()).ratio()
        total_sim += sim

        mode_stats[mode]["n"] += 1
        mode_stats[mode]["json"] += int(is_json)
        mode_stats[mode]["exact"] += int(output == expected)
        mode_stats[mode]["sim"] += sim

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}...")

    n = len(samples)
    print(f"\nFused Model Results ({args.prompt_variant}, t={args.temperature}, n={n})")
    print(f"  JSON compliance: {json_ok / n * 100:.1f}%")
    print(f"  Exact match:     {exact / n * 100:.1f}%")
    print(f"  Avg similarity:  {total_sim / n:.4f}")
    print(f"  Avg latency:     {total_lat / n:.0f}ms")

    for mode in ("standard", "full"):
        ms = mode_stats[mode]
        if ms["n"] > 0:
            print(
                f"  {mode}: JSON {ms['json'] / ms['n'] * 100:.1f}% | "
                f"Exact {ms['exact'] / ms['n'] * 100:.1f}% | "
                f"Sim {ms['sim'] / ms['n']:.4f} | n={ms['n']}"
            )


if __name__ == "__main__":
    main()
