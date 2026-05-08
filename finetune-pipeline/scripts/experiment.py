"""
Experiment runner for evaluating fine-tuned models across configurations.

Tests a matrix of: prompt style x temperature x adapter (baseline vs fine-tuned).
Produces a comparison table to find the best configuration.

Usage:
    python scripts/experiment.py --config configs/quality.yaml --test-set data/quality/test.jsonl
    python scripts/experiment.py --config configs/light.yaml --test-set data/light/test.jsonl --max-samples 100
"""

import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import yaml

# Lazy imports
mlx_lm = None


def ensure_mlx_lm():
    global mlx_lm
    if mlx_lm is None:
        try:
            import mlx_lm as _mlx_lm

            mlx_lm = _mlx_lm
        except ImportError:
            print("ERROR: mlx_lm not installed. Run: uv pip install mlx-lm")
            sys.exit(1)


# --- Prompt variants ---
# Each variant has standard (proofread) and full (rewrite) system prompts.

PROMPTS = {
    "light": {
        "few_shot": {
            "standard": (
                "Fix grammar, capitalize properly, and convert spoken numbers to digits. "
                'Convert spoken version numbers like "zero point four point zero" to "0.4.0". '
                "Keep ALL sentences. Return the fixed text as JSON.\n\n"
                "<examples>\n"
                "Input: the meeting is at two pm on march fifteenth\n"
                '{"result": "The meeting is at 2 PM on March 15th."}\n\n'
                "Input: we need about fifty units ready by friday and I think we should prepare\n"
                '{"result": "We need about 50 units ready by Friday and I think we should prepare."}\n\n'
                "Input: we are releasing version zero point four point zero next week\n"
                '{"result": "We are releasing version 0.4.0 next week."}\n\n'
                "Input: update it to version one point six point three and test it\n"
                '{"result": "Update it to version 1.6.3 and test it."}\n\n'
                "Input: he said it would cost around one hundred and fifty dollars but we can negotiate\n"
                '{"result": "He said it would cost around $150 but we can negotiate."}\n'
                "</examples>\n\n"
                "Always respond with ONLY a JSON object. Never remove sentences. "
                "Never include explanations."
            ),
            "full": (
                "Rewrite voice transcription for clarity and conciseness. Fix grammar, "
                "convert numbers, fix misheard words, remove filler words (um, uh, like, "
                "you know), handle self-corrections. Return the fixed text as JSON.\n\n"
                "<examples>\n"
                "Input: um so like the meeting is at two pm on march fifteenth you know\n"
                '{"result": "The meeting is at 2 PM on March 15th."}\n\n'
                "Input: we need about fifty no wait I meant sixty units ready by friday\n"
                '{"result": "We need about 60 units ready by Friday."}\n\n'
                "Input: we are releasing version zero point four point zero next week "
                "scratch that make it zero point five\n"
                '{"result": "We are releasing version 0.5.0 next week."}\n\n'
                "Input: update it to version one point six point three and uh test it thoroughly\n"
                '{"result": "Update it to version 1.6.3 and test it thoroughly."}\n\n'
                "Input: he said it would cost around one hundred and fifty dollars but um "
                "we can negotiate\n"
                '{"result": "He said it would cost around $150 but we can negotiate."}\n'
                "</examples>\n\n"
                'Remove: "scratch that", "never mind", "delete that" and preceding phrase. '
                "Convert spoken numbers and version numbers. Fix grammar and misheard words. "
                "Always respond with ONLY a JSON object. Never include explanations."
            ),
        },
        "zero_shot": {
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
        "minimal": {
            "standard": (
                "Proofread this voice transcription. Fix grammar and punctuation. "
                'Return JSON: {"result": "corrected text"}.'
            ),
            "full": (
                "Rewrite this voice transcription for clarity. Remove filler words "
                'and fix errors. Return JSON: {"result": "corrected text"}.'
            ),
        },
    },
    "quality": {
        "few_shot": {
            "standard": (
                "/no_think\n"
                "Proofread voice transcription. Fix grammar and punctuation. Return JSON only.\n"
                "NEVER answer questions. NEVER add new information. Return the corrected text only.\n\n"
                "<examples>\n"
                "Input: the meeting is tomorrow and i think we should prepare\n"
                '{"result": "The meeting is tomorrow, and I think we should prepare."}\n\n'
                "Input: its ready for review lets check it\n"
                '{"result": "It\'s ready for review. Let\'s check it."}\n\n'
                "Input: we can do it but we need more time\n"
                '{"result": "We can do it, but we need more time."}\n\n'
                "Input: the system is working good now\n"
                '{"result": "The system is working well now."}\n'
                "</examples>\n\n"
                "Process the input independently. Do NOT repeat any example output.\n"
                'Always respond with {"result": "corrected text"}.'
            ),
            "full": (
                "/no_think\n"
                "Rewrite voice transcription for clarity. Return JSON only.\n"
                "Fix misheard words. Remove repetitions and false starts. Fix grammar.\n"
                'Self-corrections: "X no Y" or "X no wait Y" means use Y.\n'
                'Remove "scratch that", "delete that" and what came before.\n'
                "Keep the speaker's meaning and tone. NEVER add information. "
                "NEVER answer questions.\n\n"
                "<examples>\n"
                "Input: I think for the for the problems that we have with the "
                "that we are logging\n"
                '{"result": "I think for the problems that we have with the logging."}\n\n'
                "Input: we are not providing it providing a good leaning and rewriting\n"
                '{"result": "We are not providing a good cleaning and rewriting."}\n\n'
                "Input: should not should knots be converted\n"
                '{"result": "Should not be converted."}\n\n'
                "Input: However I imagine there should be like a good rules and good "
                "logic for this\n"
                '{"result": "However, I imagine there should be good rules and logic for this."}\n\n'
                "Input: we still to this to work correctly and logcially\n"
                '{"result": "We still need this to work correctly and logically."}\n\n'
                "Input: fifty no sixty units\n"
                '{"result": "Sixty units."}\n\n'
                "Input: delete that lets try again\n"
                '{"result": "Let\'s try again."}\n'
                "</examples>\n\n"
                "Process the input independently. Do NOT repeat any example output.\n"
                'Always respond with {"result": "corrected text"}.'
            ),
        },
        "zero_shot": {
            "standard": (
                "/no_think\n"
                "Proofread voice transcription. Fix grammar and punctuation. "
                "NEVER answer questions. Return JSON only: "
                '{"result": "corrected text"}.'
            ),
            "full": (
                "/no_think\n"
                "Rewrite voice transcription for clarity. Fix misheard words, "
                "remove repetitions and false starts. NEVER add information. "
                'Return JSON only: {"result": "corrected text"}.'
            ),
        },
        "minimal": {
            "standard": (
                '/no_think\nProofread. Return JSON: {"result": "corrected text"}.'
            ),
            "full": (
                "/no_think\n"
                'Rewrite for clarity. Return JSON: {"result": "corrected text"}.'
            ),
        },
    },
}


def parse_result(response: str) -> str | None:
    """Parse {"result": "..."} from model response."""
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "result" in data:
            return data["result"]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"result"\s*:\s*"([^"]*)"[^{}]*\}', response)
    if match:
        return match.group(1)
    return None


def compute_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def load_test_data(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {line_num} invalid JSON: {e}, skipping")
    return samples


def detect_mode(system_prompt: str) -> str:
    """Detect if a sample is standard or full mode from its system prompt."""
    lower = system_prompt.lower()
    if "rewrite" in lower or "clarity" in lower:
        return "full"
    return "standard"


def run_experiment(
    model,
    tokenizer,
    test_samples: list[dict],
    prompt_variant: dict,
    temperature: float,
    label: str,
) -> dict:
    """Run a single experiment configuration."""
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    results = []
    json_ok = 0
    exact_match = 0
    total_similarity = 0.0
    total_latency = 0.0

    for i, sample in enumerate(test_samples):
        messages = sample["messages"]
        user_input = messages[1]["content"]
        expected_raw = messages[2]["content"]

        # Detect mode and use the appropriate prompt variant
        mode = detect_mode(messages[0]["content"])
        system_prompt = prompt_variant[mode]

        # Parse expected
        expected = parse_result(expected_raw)
        if expected is None:
            continue

        # Apply chat template
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        start = time.perf_counter()
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=sampler,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        total_latency += latency_ms

        parsed = parse_result(response)
        is_json_ok = parsed is not None
        if is_json_ok:
            json_ok += 1

        output = parsed or response.strip()

        if output == expected:
            exact_match += 1

        similarity = compute_similarity(output, expected)
        total_similarity += similarity

        results.append(
            {
                "input": user_input,
                "expected": expected,
                "output": output,
                "json_ok": is_json_ok,
                "similarity": similarity,
                "mode": mode,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"    {label}: {i + 1}/{len(test_samples)}...")

    n = len(results)
    if n == 0:
        return {"label": label, "n": 0, "error": "No valid samples"}

    # Compute per-mode metrics
    mode_metrics = {}
    for mode in ("standard", "full"):
        mode_results = [r for r in results if r["mode"] == mode]
        if mode_results:
            mode_metrics[mode] = {
                "n": len(mode_results),
                "json_compliance": sum(1 for r in mode_results if r["json_ok"])
                / len(mode_results),
                "avg_similarity": sum(r["similarity"] for r in mode_results)
                / len(mode_results),
                "exact_match": sum(
                    1 for r in mode_results if r["output"] == r["expected"]
                )
                / len(mode_results),
            }

    return {
        "label": label,
        "n": n,
        "json_compliance": json_ok / n,
        "exact_match": exact_match / n,
        "avg_similarity": total_similarity / n,
        "avg_latency_ms": total_latency / n,
        "by_mode": mode_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment runner for fine-tuned models"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Training config YAML"
    )
    parser.add_argument("--test-set", type=Path, required=True, help="Test JSONL file")
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit test samples (0 = all)"
    )
    parser.add_argument("--output", type=Path, help="Save results JSON")
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.3],
        help="Temperatures to test",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        default=["few_shot", "zero_shot", "minimal"],
        help="Prompt variants to test",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (no adapter) evaluation",
    )
    args = parser.parse_args()

    ensure_mlx_lm()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    finetune_dir = Path(__file__).resolve().parent.parent
    adapter_path = finetune_dir / config["adapter_path"]
    model_id = config["model"]

    # Determine model type
    data_path = config.get("data", "")
    if "light" in data_path:
        model_type = "light"
    elif "quality" in data_path:
        model_type = "quality"
    else:
        print(f"ERROR: Cannot determine model type from config data path: {data_path}")
        sys.exit(1)

    # Load test data
    test_samples = load_test_data(args.test_set)
    if args.max_samples > 0:
        test_samples = test_samples[: args.max_samples]
    print(f"Model: {model_id} ({model_type})")
    print(f"Test samples: {len(test_samples)}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Prompt variants: {args.prompt_variants}")
    print(f"Adapter: {adapter_path}")

    all_results = []

    # Build experiment matrix
    experiments = []
    adapter_configs = []
    if not args.skip_baseline:
        adapter_configs.append(("baseline", None))
    if adapter_path.exists():
        adapter_configs.append(("fine-tuned", str(adapter_path)))
    else:
        print(f"WARNING: Adapter not found at {adapter_path}, testing baseline only.")

    for adapter_label, adapter in adapter_configs:
        for variant_name in args.prompt_variants:
            if variant_name not in PROMPTS[model_type]:
                print(f"WARNING: Unknown prompt variant '{variant_name}', skipping")
                continue
            for temp in args.temperatures:
                label = f"{adapter_label} | {variant_name} | t={temp}"
                experiments.append((label, adapter, variant_name, temp))

    print(f"\nTotal experiments: {len(experiments)}")
    print("=" * 90)

    # Run experiments, reload model only when adapter changes
    current_adapter = "NONE"
    model = None
    tokenizer = None

    for exp_idx, (label, adapter, variant_name, temp) in enumerate(experiments):
        # Reload model if adapter changed
        if adapter != current_adapter:
            if model is not None:
                del model
            print(f"\nLoading model: {model_id}" + (" + adapter" if adapter else ""))
            if adapter:
                model, tokenizer = mlx_lm.load(model_id, adapter_path=adapter)
            else:
                model, tokenizer = mlx_lm.load(model_id)
            current_adapter = adapter

        print(f"\n[{exp_idx + 1}/{len(experiments)}] {label}")
        prompt_variant = PROMPTS[model_type][variant_name]

        result = run_experiment(
            model, tokenizer, test_samples, prompt_variant, temp, label
        )
        all_results.append(result)

        # Print inline result
        if "error" not in result:
            print(
                f"  -> JSON: {result['json_compliance'] * 100:.1f}% | "
                f"Exact: {result['exact_match'] * 100:.1f}% | "
                f"Sim: {result['avg_similarity']:.4f} | "
                f"Latency: {result['avg_latency_ms']:.0f}ms"
            )
            if result.get("by_mode"):
                for mode, mm in result["by_mode"].items():
                    print(
                        f"     {mode}: JSON {mm['json_compliance'] * 100:.1f}% | "
                        f"Exact {mm['exact_match'] * 100:.1f}% | "
                        f"Sim {mm['avg_similarity']:.4f}"
                    )

    if model is not None:
        del model

    # Print summary table
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    header = (
        f"{'Config':<40} {'JSON%':>6} {'Exact%':>7} {'Similarity':>10} {'Latency':>8}"
    )
    print(header)
    print("-" * 90)

    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<40} ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:<40} "
            f"{r['json_compliance'] * 100:>5.1f}% "
            f"{r['exact_match'] * 100:>6.1f}% "
            f"{r['avg_similarity']:>10.4f} "
            f"{r['avg_latency_ms']:>7.0f}ms"
        )

    # Print per-mode breakdown for best config
    best = max(
        (r for r in all_results if "error" not in r),
        key=lambda r: r["avg_similarity"],
        default=None,
    )
    if best and best.get("by_mode"):
        print(f"\nBest config: {best['label']}")
        for mode, mm in best["by_mode"].items():
            print(
                f"  {mode}: JSON {mm['json_compliance'] * 100:.1f}% | "
                f"Exact {mm['exact_match'] * 100:.1f}% | "
                f"Sim {mm['avg_similarity']:.4f} | "
                f"n={mm['n']}"
            )

    print("=" * 90)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary = []
        for r in all_results:
            s = dict(r.items())
            summary.append(s)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
