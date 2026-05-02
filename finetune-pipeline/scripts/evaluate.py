"""
Evaluation pipeline for fine-tuned models.

Compares baseline (no adapter) vs fine-tuned (with adapter) on the test set.
Measures JSON compliance, similarity to gold standard, prompt echoing rate,
and latency.

Usage:
    python scripts/evaluate.py --config configs/light.yaml --test-set data/light/test.jsonl
    python scripts/evaluate.py --config configs/quality.yaml --test-set data/quality/test.jsonl
    python scripts/evaluate.py --config configs/light.yaml --test-set data/light/test.jsonl --zero-shot
"""

import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import yaml

# Lazy imports for mlx_lm (heavy)
mlx_lm = None


def ensure_mlx_lm():
    global mlx_lm
    if mlx_lm is None:
        import mlx_lm as _mlx_lm

        mlx_lm = _mlx_lm


# Zero-shot prompts (no examples, for testing fine-tuned models)
ZERO_SHOT_PROMPTS = {
    "light": {
        "standard": (
            "Fix grammar, capitalize properly, and convert spoken numbers to "
            "digits. Keep ALL sentences. Return the fixed text as JSON: "
            '{"result": "corrected text"}.'
        ),
        "full": (
            "Rewrite voice transcription for clarity. Fix grammar, convert "
            "numbers, remove fillers, handle self-corrections. Return as JSON: "
            '{"result": "corrected text"}.'
        ),
    },
    "quality": {
        "standard": (
            "/no_think\nProofread voice transcription. Fix grammar and "
            "punctuation. NEVER answer questions. Return JSON only: "
            '{"result": "corrected text"}.'
        ),
        "full": (
            "/no_think\nRewrite voice transcription for clarity. Fix misheard "
            "words, remove repetitions and false starts. NEVER add information. "
            'Return JSON only: {"result": "corrected text"}.'
        ),
    },
}


def extract_prompt_examples(system_prompt: str) -> set[str]:
    """Extract example outputs from system prompt for echo detection."""
    examples = set()
    pattern = r'\{"result":\s*"([^"]+)"\}'
    for match in re.finditer(pattern, system_prompt):
        examples.add(match.group(1))
    return examples


def parse_result(response: str) -> str | None:
    """Parse {"result": "..."} from model response."""
    # Try direct JSON parse
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "result" in data:
            return data["result"]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from response
    match = re.search(r'\{[^{}]*"result"\s*:\s*"([^"]*)"[^{}]*\}', response)
    if match:
        return match.group(1)

    return None


def compute_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def load_test_data(path: Path) -> list[dict]:
    """Load test JSONL with chat-format samples."""
    samples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {line_num} in {path} invalid JSON: {e}, skipping")
    return samples


def run_inference(
    model,
    tokenizer,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> tuple[str, float]:
    """Run model inference and return (response, latency_ms)."""
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    sampler = make_sampler(temp=temperature)
    start = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return response, latency_ms


def evaluate_model(
    model,
    tokenizer,
    test_samples: list[dict],
    prompt_examples: set[str],
    label: str,
    override_system_prompt: str | None = None,
) -> dict:
    """Run evaluation on test samples and compute metrics."""
    results = []
    json_ok = 0
    exact_match = 0
    echo_count = 0
    total_similarity = 0.0
    total_latency = 0.0

    for i, sample in enumerate(test_samples):
        messages = sample["messages"]
        system_prompt = override_system_prompt or messages[0]["content"]
        user_input = messages[1]["content"]
        expected_raw = messages[2]["content"]

        # Parse expected gold standard
        expected = parse_result(expected_raw)
        if expected is None:
            print(f"  WARNING: Could not parse expected result for sample {i}")
            continue

        # Run inference
        response, latency = run_inference(model, tokenizer, system_prompt, user_input)
        total_latency += latency

        # Parse model output
        parsed = parse_result(response)
        is_json_ok = parsed is not None
        if is_json_ok:
            json_ok += 1

        output = parsed or response.strip()

        # Check exact match
        if output == expected:
            exact_match += 1

        # Check similarity
        similarity = compute_similarity(output, expected)
        total_similarity += similarity

        # Check prompt echo
        is_echo = output in prompt_examples
        if is_echo:
            echo_count += 1

        results.append(
            {
                "input": user_input,
                "expected": expected,
                "output": output,
                "raw_response": response,
                "json_ok": is_json_ok,
                "similarity": similarity,
                "is_echo": is_echo,
                "latency_ms": latency,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {label}: {i + 1}/{len(test_samples)} samples processed...")

    n = len(results)
    if n == 0:
        return {"label": label, "n": 0, "error": "No valid samples"}

    metrics = {
        "label": label,
        "n": n,
        "json_compliance": json_ok / n,
        "exact_match": exact_match / n,
        "avg_similarity": total_similarity / n,
        "echo_rate": echo_count / n,
        "avg_latency_ms": total_latency / n,
        "details": results,
    }
    return metrics


def print_comparison(all_metrics: list[dict]) -> None:
    """Print a comparison table of all evaluated configurations."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    header = f"{'Config':<30} {'JSON%':>6} {'Exact%':>7} {'Similarity':>10} {'Echo%':>6} {'Latency':>8}"
    print(header)
    print("-" * 80)

    for m in all_metrics:
        if "error" in m:
            print(f"{m['label']:<30} ERROR: {m['error']}")
            continue
        print(
            f"{m['label']:<30} "
            f"{m['json_compliance'] * 100:>5.1f}% "
            f"{m['exact_match'] * 100:>6.1f}% "
            f"{m['avg_similarity']:>10.4f} "
            f"{m['echo_rate'] * 100:>5.1f}% "
            f"{m['avg_latency_ms']:>7.0f}ms"
        )

    print("=" * 80)

    # Show regressions if we have baseline + fine-tuned
    if len(all_metrics) >= 2 and all("details" in m for m in all_metrics[:2]):
        baseline = all_metrics[0]
        finetuned = all_metrics[1]
        regressions = 0
        for b, f in zip(baseline["details"], finetuned["details"], strict=False):
            if f["similarity"] < b["similarity"] - 0.05:
                regressions += 1
        regression_rate = regressions / baseline["n"] if baseline["n"] > 0 else 0
        print(
            f"\nRegression rate (fine-tuned worse by >5% similarity): "
            f"{regressions}/{baseline['n']} ({regression_rate * 100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument(
        "--config", type=Path, required=True, help="Training config YAML"
    )
    parser.add_argument("--test-set", type=Path, required=True, help="Test JSONL file")
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Also evaluate with zero-shot prompts (no examples)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of test samples (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    ensure_mlx_lm()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    finetune_dir = Path(__file__).resolve().parent.parent
    adapter_path = finetune_dir / config["adapter_path"]
    model_id = config["model"]

    # Determine model type from data path in config
    data_path = config.get("data", "")
    if "light" in data_path:
        model_type = "light"
    elif "quality" in data_path:
        model_type = "quality"
    else:
        print(f"ERROR: Cannot determine model type from config data path: {data_path}")
        print("Config 'data' field must contain 'light' or 'quality'.")
        sys.exit(1)

    # Load test data
    test_samples = load_test_data(args.test_set)
    if args.max_samples > 0:
        test_samples = test_samples[: args.max_samples]
    print(f"Test samples: {len(test_samples)}")

    # Extract prompt examples for echo detection
    all_examples = set()
    for mode_prompts in ZERO_SHOT_PROMPTS.values():
        for prompt in mode_prompts.values():
            all_examples |= extract_prompt_examples(prompt)
    # Also extract from test data system prompts
    for sample in test_samples:
        all_examples |= extract_prompt_examples(sample["messages"][0]["content"])
    print(f"Prompt examples for echo detection: {len(all_examples)}")

    all_metrics = []

    # 1. Baseline (no adapter)
    print(f"\nLoading baseline model: {model_id}")
    model, tokenizer = mlx_lm.load(model_id)
    print("Evaluating baseline...")
    baseline_metrics = evaluate_model(
        model, tokenizer, test_samples, all_examples, "Baseline"
    )
    all_metrics.append(baseline_metrics)
    del model  # Free memory

    # 2. Fine-tuned (with adapter)
    if adapter_path.exists():
        print(f"\nLoading fine-tuned model: {model_id} + {adapter_path}")
        model, tokenizer = mlx_lm.load(model_id, adapter_path=str(adapter_path))
        print("Evaluating fine-tuned...")
        ft_metrics = evaluate_model(
            model, tokenizer, test_samples, all_examples, "Fine-tuned"
        )
        all_metrics.append(ft_metrics)

        # 3. Fine-tuned zero-shot (if requested)
        if args.zero_shot:
            # Determine which zero-shot prompt to use based on test sample
            # We run all samples with the zero-shot version of their original prompt
            print("Evaluating fine-tuned zero-shot...")

            # Create modified test samples with zero-shot prompts
            from copy import deepcopy

            zs_samples = deepcopy(test_samples)
            for sample in zs_samples:
                orig_prompt = sample["messages"][0]["content"]
                # Detect mode from prompt content
                if "Rewrite" in orig_prompt or "rewrite" in orig_prompt:
                    sample["messages"][0]["content"] = ZERO_SHOT_PROMPTS[model_type][
                        "full"
                    ]
                else:
                    sample["messages"][0]["content"] = ZERO_SHOT_PROMPTS[model_type][
                        "standard"
                    ]

            zs_metrics = evaluate_model(
                model, tokenizer, zs_samples, all_examples, "Fine-tuned (zero-shot)"
            )
            all_metrics.append(zs_metrics)

        del model
    else:
        print(f"\nERROR: Adapter not found at {adapter_path}")
        print("Train the model first with: python scripts/train.py --config <config>")
        sys.exit(1)

    # Print comparison
    print_comparison(all_metrics)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Remove detailed results for summary file (they're large)
        summary = []
        for m in all_metrics:
            s = {k: v for k, v in m.items() if k != "details"}
            summary.append(s)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")

        # Also save detailed results
        detail_path = args.output.with_suffix(".details.json")
        with open(detail_path, "w") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {detail_path}")


if __name__ == "__main__":
    main()
