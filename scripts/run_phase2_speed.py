"""Phase 2: Speed benchmark for issue #9.

Measures (per model):
  - tokens/sec generation throughput,
  - time-to-first-token (TTFT) from streaming generate,
  - peak resident set size (RSS) via psutil delta,
  - load time.

Reuses the same candidate list as run_phase1_quality.py so numbers line up.
Outputs:
  /Volumes/S1/yooz/research/issue-9/results/phase2_speed/{slug}.json
  /Volumes/S1/yooz/research/issue-9/results/phase2_speed/summary_table.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median, stdev

_HF_HOME = "/Volumes/S1/yooz/research/issue-9/models/hf_cache"
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_HF_HOME}/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", f"{_HF_HOME}/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse the model list from phase1.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase1_quality import (  # noqa: E402
    CANDIDATES,
    PROOFREAD_SYSTEM,
    REWRITE_SYSTEM,
    build_prompt,
)

RESULTS_DIR = Path("/Volumes/S1/yooz/research/issue-9/results/phase2_speed")

# Three representative inputs covering short/medium/long.
PROMPT_INPUTS = [
    ("short", "let us meet tomorrow at nine am"),
    ("medium", (
        "we are releasing version zero point four point zero next week and "
        "we need to make sure all the tests pass before then because the team "
        "is depending on this release for the demo"
    )),
    ("long", (
        "so um basically what i was trying to say is that you know when we when we "
        "look at the entire flow from the time the user opens the app to the moment "
        "they get the response we have a few bottlenecks like for example the model "
        "load time the request serialization and the network round trip but i think "
        "the biggest one right now is actually the model itself it just doesn't "
        "stream tokens fast enough so we should probably benchmark it across a few "
        "different sizes and quantization levels to see whats actually happening"
    )),
]


def measure_one(spec, mlx_lm, model, tokenizer, length_label: str, user: str,
                max_tokens: int, mode: str = "rewrite") -> dict:
    """Run one streaming generation and capture TTFT + tok/s."""
    from mlx_lm.sample_utils import make_sampler

    system = PROOFREAD_SYSTEM if mode == "proofread" else REWRITE_SYSTEM
    prompt = build_prompt(tokenizer, system, user, spec.family)

    sampler = make_sampler(temp=0.0)

    # Streaming generate.
    started = time.perf_counter()
    first_token_at: float | None = None
    n_tokens = 0
    text_buf = []
    for chunk in mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                        max_tokens=max_tokens, sampler=sampler):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        # GenerationResponse object exposes .text and .token
        n_tokens += 1
        text_buf.append(getattr(chunk, "text", "") or "")
    end = time.perf_counter()

    if first_token_at is None:
        ttft_ms = float("inf")
    else:
        ttft_ms = (first_token_at - started) * 1000.0

    decode_s = end - (first_token_at or started)
    tps = n_tokens / decode_s if decode_s > 0 else 0.0
    return {
        "length": length_label,
        "mode": mode,
        "n_tokens_generated": n_tokens,
        "total_ms": (end - started) * 1000.0,
        "ttft_ms": ttft_ms,
        "decode_seconds": decode_s,
        "tokens_per_second": tps,
        "output_preview": ("".join(text_buf))[:200],
    }


def evaluate_speed(spec, max_tokens: int, warmup: int, n_repeats: int) -> dict:
    import psutil
    import mlx_lm

    print(f"\n[{spec.slug}] loading {spec.repo}")

    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    t0 = time.perf_counter()
    try:
        if spec.family == "gemma4":
            from mlx_lm.utils import load_model, load_tokenizer, hf_repo_to_path
            path = hf_repo_to_path(spec.repo)
            model, _ = load_model(path, strict=False)
            tokenizer = load_tokenizer(path)
        else:
            model, tokenizer = mlx_lm.load(spec.repo)
    except Exception as e:
        return {
            "slug": spec.slug,
            "repo": spec.repo,
            "error": f"load failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    load_seconds = time.perf_counter() - t0
    rss_after_load = proc.memory_info().rss

    # Warmup runs (compile MLX kernels, fill caches).
    print(f"[{spec.slug}] loaded in {load_seconds:.1f}s, warmup x{warmup}")
    for _ in range(warmup):
        measure_one(spec, mlx_lm, model, tokenizer,
                    PROMPT_INPUTS[0][0], PROMPT_INPUTS[0][1], 32, "rewrite")

    rss_post_warmup = proc.memory_info().rss

    # Per-length measurements with repeats.
    measurements = []
    for length, user in PROMPT_INPUTS:
        for mode in ("proofread", "rewrite"):
            for _ in range(n_repeats):
                m = measure_one(spec, mlx_lm, model, tokenizer, length, user,
                                max_tokens, mode)
                measurements.append(m)
                rss = proc.memory_info().rss
                m["rss_bytes"] = rss

    rss_peak = max(m["rss_bytes"] for m in measurements)

    def stat(values):
        if not values:
            return None
        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "stdev": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    summary = {
        "slug": spec.slug,
        "repo": spec.repo,
        "family": spec.family,
        "role": spec.role,
        "load_seconds": load_seconds,
        "rss_baseline_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_warmup_bytes": rss_post_warmup,
        "rss_peak_bytes": rss_peak,
        "rss_load_delta_bytes": rss_after_load - rss_before,
        "rss_peak_delta_bytes": rss_peak - rss_before,
        "ttft_ms": stat([m["ttft_ms"] for m in measurements]),
        "tokens_per_second": stat([m["tokens_per_second"] for m in measurements]),
        "total_ms": stat([m["total_ms"] for m in measurements]),
        "by_length": {},
    }

    # Per-length breakdown.
    for length, _ in PROMPT_INPUTS:
        sub = [m for m in measurements if m["length"] == length]
        summary["by_length"][length] = {
            "ttft_ms": stat([m["ttft_ms"] for m in sub]),
            "tokens_per_second": stat([m["tokens_per_second"] for m in sub]),
        }

    out_path = RESULTS_DIR / f"{spec.slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "measurements": measurements}, f, indent=2)
    print(f"[{spec.slug}] saved {out_path}")

    del model
    del tokenizer
    gc.collect()
    return summary


def render_summary(summaries, out_md: Path) -> None:
    lines = ["# Phase 2 Speed Benchmark — issue #9\n"]
    lines.append("| Model | Load (s) | TTFT med (ms) | tok/s med | Peak RSS (MB) |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            lines.append(f"| {s['slug']} | ERR | ERR | ERR | ERR |")
            continue
        ttft = (s["ttft_ms"] or {}).get("median", float("nan"))
        tps = (s["tokens_per_second"] or {}).get("median", float("nan"))
        peak_mb = s["rss_peak_bytes"] / (1024 * 1024)
        lines.append(
            f"| {s['slug']} | {s['load_seconds']:.1f} | "
            f"{ttft:.0f} | {tps:.1f} | {peak_mb:.0f} |"
        )

    lines.append("\n## TTFT by input length\n")
    lines.append("| Model | short | medium | long |")
    lines.append("|---|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            continue
        row = [s["slug"]]
        for length in ("short", "medium", "long"):
            ttft = (s["by_length"][length]["ttft_ms"] or {}).get("median", float("nan"))
            row.append(f"{ttft:.0f}")
        lines.append("| " + " | ".join(row) + " |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--skip", nargs="+", default=None)
    args = p.parse_args()

    selected = CANDIDATES
    if args.only:
        selected = [s for s in selected if s.slug in set(args.only)]
    if args.skip:
        selected = [s for s in selected if s.slug not in set(args.skip)]

    summaries = []
    for spec in selected:
        try:
            s = evaluate_speed(spec, args.max_tokens, args.warmup, args.repeats)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[{spec.slug}] FATAL {e}")
            traceback.print_exc()
            s = {"slug": spec.slug, "repo": spec.repo, "role": spec.role, "error": str(e)}
        summaries.append(s)

    out_md = RESULTS_DIR / "summary_table.md"
    render_summary(summaries, out_md)
    print(f"\nSummary: {out_md}")
    (RESULTS_DIR / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
