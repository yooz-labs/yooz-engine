"""Phase 1: Quality benchmark for issue #9.

Compares Qwen3.5 + Gemma-4 candidates against Qwen2.5/Qwen3 baselines on the
gold-standard touchup test split. Two prompts per item (proofread + rewrite),
metrics: exact-match, CER (jiwer), semantic similarity (MiniLM).

Outputs per-model JSON to:
  /Volumes/S1/yooz/research/issue-9/results/phase1_quality/{slug}.json

And an aggregated comparison:
  /Volumes/S1/yooz/research/issue-9/results/phase1_quality/summary_table.md

Heavy I/O lives on S1; only the script + summary stay in the worktree.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# All caching on S1.
_HF_HOME = "/Volumes/S1/yooz/research/issue-9/models/hf_cache"
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_HF_HOME}/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", f"{_HF_HOME}/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESULTS_DIR = Path("/Volumes/S1/yooz/research/issue-9/results/phase1_quality")
LOGS_DIR = Path("/Volumes/S1/yooz/research/issue-9/logs")
DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "bench-tools" / "gold_standard.jsonl"

# Touchup prompts. Locked from bench-tools/prompts.py + finetune-pipeline/scripts/experiment.py
# but with placeholder text removed (early smoke runs showed Qwen3.5/Gemma echoing the
# literal "corrected text" placeholder back).
PROOFREAD_SYSTEM = (
    "You are a copy editor. Proofread the speech-to-text transcription provided by the "
    "user. Fix spelling, punctuation, capitalization, and obvious word-choice errors "
    "caused by the speech-to-text engine. Convert spoken numbers to digits where it "
    "improves clarity (e.g. \"nine am\" -> \"9 AM\"). Keep the speaker's voice; do NOT "
    "rephrase or remove filler words (um, uh, like). "
    "Output ONLY a single JSON object of the form {\"result\": \"<corrected>\"} where "
    "<corrected> is the actual corrected text. No explanation, no markdown."
)

REWRITE_SYSTEM = (
    "You are an editor. Rewrite the speech-to-text transcription provided by the user "
    "for clarity and readability. Fix grammar, spelling, and punctuation. Remove filler "
    "words (um, uh, like, you know, basically). Convert spoken numbers to digits. Split "
    "run-on sentences. Preserve the original meaning and intent. "
    "Output ONLY a single JSON object of the form {\"result\": \"<rewritten>\"} where "
    "<rewritten> is the actual rewritten text. No explanation, no markdown."
)

# Gemma-4 emits a structured `<|channel>thought ... <channel|>` reasoning channel by
# default. Forcing `<|channel>final` in the assistant prefix skips the thought channel
# and goes straight to the answer.
GEMMA4_FINAL_SUFFIX = "<|channel>final\n"

JSON_RE = re.compile(r'\{[^{}]*"result"\s*:\s*"((?:[^"\\]|\\.)*)"[^{}]*\}', re.DOTALL)


@dataclass
class ModelSpec:
    slug: str  # short, filename-safe
    repo: str  # HF id
    family: str  # qwen2 / qwen3 / qwen35 / gemma4
    role: str  # baseline | candidate
    no_think: bool = False  # prepend /no_think


# Notes:
# - mlx-community/Qwen3-1.7B-Ojus-4bit is gated/missing; we substitute the closest
#   public equivalent, mlx-community/Qwen3-1.7B-4bit, as the production-Quality
#   baseline. The engine's deployed "yoozQuality" was QLoRA-fine-tuned from this.
# - mlx-community/gemma-4-e2b-it-4bit ships the multimodal "language_model.*"
#   layout that mlx-lm 0.31.3 cannot load. We substitute the text-only stripped
#   checkpoint mlx-community/Gemma4-E2B-IT-Text-int4 (model_type=gemma4_text)
#   and load it with strict=False to drop spurious K/V projections in the last
#   20 KV-shared layers. The E4B text-only equivalent does not yet exist on
#   mlx-community; the multimodal e4b 4bit fails to load for the same reason
#   and is therefore skipped.
CANDIDATES = [
    ModelSpec("qwen2.5-0.5B-baseline", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "qwen2", "baseline", False),
    ModelSpec("qwen3-1.7B-baseline", "mlx-community/Qwen3-1.7B-4bit", "qwen3", "baseline", True),
    ModelSpec("qwen3.5-0.8B", "mlx-community/Qwen3.5-0.8B-MLX-4bit", "qwen35", "candidate", True),
    ModelSpec("qwen3.5-2B-optiq", "mlx-community/Qwen3.5-2B-OptiQ-4bit", "qwen35", "candidate", True),
    ModelSpec("gemma-4-e2b-text", "mlx-community/Gemma4-E2B-IT-Text-int4", "gemma4", "candidate", False),
]


def load_test_set(path: Path, limit: int | None) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]
    return samples


def parse_json_result(response: str) -> str | None:
    s = response.strip()
    # Strip common Qwen3 think tags + Gemma-4 channel markers.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"<\|channel>thought.*?<channel\|>", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"<\|channel>final\s*", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"<channel\|>|<turn\|>|<\|turn>", "", s).strip()
    # Try a clean json object.
    try:
        if s.startswith("{"):
            obj = json.loads(s)
            if isinstance(obj, dict) and "result" in obj:
                return str(obj["result"])
    except json.JSONDecodeError:
        pass
    m = JSON_RE.search(s)
    if m:
        # Unescape minimal JSON escapes.
        return m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
    return None


def cer(reference: str, hypothesis: str, jiwer_mod) -> float:
    """Character error rate via jiwer."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer_mod.cer(reference, hypothesis)


def build_prompt(tokenizer, system: str, user: str, family: str) -> str:
    """Family-aware chat template assembly.

    - qwen35 / qwen3: uses `enable_thinking=False` to skip the empty <think>...</think>.
    - gemma4: hand-rolled prompt with `<|channel>final\n` suffix to skip the thought
      channel (the tokenizer config has no chat_template).
    - qwen2 (Qwen2.5) and others: default chat template.
    """
    if family == "gemma4":
        # Gemma-4 has no chat_template in tokenizer_config.json. Build by hand.
        return (
            "<|turn>system\n" + system + "<turn|>"
            "<|turn>user\n" + user + "<turn|>"
            "<|turn>model\n" + GEMMA4_FINAL_SUFFIX
        )

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if family in ("qwen3", "qwen35"):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            # Fallback if tokenizer doesn't accept enable_thinking (older).
            pass
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate_model(
    spec: ModelSpec,
    samples: list[dict],
    sim_model,
    jiwer_mod,
    max_tokens: int,
) -> dict:
    """Run proofread + rewrite for every sample, compute metrics."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    print(f"\n[{spec.slug}] loading {spec.repo}")
    t0 = time.perf_counter()
    try:
        if spec.family == "gemma4":
            # gemma-4 text-only checkpoints ship k_proj/v_proj for the last 20
            # KV-shared layers that mlx-lm's Gemma4 arch doesn't allocate.
            # Loading non-strictly drops those extras cleanly.
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
    load_s = time.perf_counter() - t0
    print(f"[{spec.slug}] loaded in {load_s:.1f}s")

    sampler = make_sampler(temp=0.0)
    per_sample = []
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)

    n = len(samples)
    started = time.perf_counter()
    for i, sample in enumerate(samples):
        meta = sample.get("metadata", {})
        difficulty = meta.get("difficulty", "unknown")
        domain = meta.get("domain", "unknown")
        raw = sample["raw_transcription"]

        for mode, system, expected in (
            ("proofread", PROOFREAD_SYSTEM, sample["proofread"]),
            ("rewrite", REWRITE_SYSTEM, sample["rewrite"]),
        ):
            prompt = build_prompt(tokenizer, system, raw, spec.family)
            t = time.perf_counter()
            try:
                resp = mlx_lm.generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                )
            except Exception as e:
                resp = ""
                gen_err = f"{type(e).__name__}: {e}"
            else:
                gen_err = None
            latency_ms = (time.perf_counter() - t) * 1000

            parsed = parse_json_result(resp)
            output = parsed if parsed is not None else resp.strip()
            json_ok = parsed is not None

            ex = output == expected
            try:
                cer_v = cer(expected, output, jiwer_mod)
            except Exception:
                cer_v = 1.0

            row = {
                "id": sample["id"],
                "mode": mode,
                "difficulty": difficulty,
                "domain": domain,
                "expected": expected,
                "output": output,
                "json_ok": json_ok,
                "exact_match": ex,
                "cer": cer_v,
                "latency_ms": latency_ms,
                "gen_err": gen_err,
            }
            per_sample.append(row)
            by_difficulty[difficulty].append(row)
            by_domain[domain].append(row)
            by_mode[mode].append(row)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - started
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate else 0
            print(f"[{spec.slug}] {i + 1}/{n}  rate={rate:.2f} samples/s  eta={eta:.0f}s", flush=True)

    # Semantic similarity in batches at the end.
    print(f"[{spec.slug}] computing semantic similarity for {len(per_sample)} rows")
    expected_texts = [r["expected"] for r in per_sample]
    output_texts = [r["output"] for r in per_sample]
    emb_e = sim_model.encode(expected_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    emb_o = sim_model.encode(output_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    import numpy as np
    sims = (emb_e * emb_o).sum(axis=1)
    for row, sim in zip(per_sample, sims):
        row["semantic_similarity"] = float(sim)

    def aggregate(rows):
        if not rows:
            return None
        n_rows = len(rows)
        return {
            "n": n_rows,
            "exact_match": sum(1 for r in rows if r["exact_match"]) / n_rows,
            "json_ok": sum(1 for r in rows if r["json_ok"]) / n_rows,
            "avg_cer": sum(r["cer"] for r in rows) / n_rows,
            "avg_semantic_similarity": sum(r["semantic_similarity"] for r in rows) / n_rows,
            "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n_rows,
        }

    summary = {
        "slug": spec.slug,
        "repo": spec.repo,
        "family": spec.family,
        "role": spec.role,
        "no_think": spec.no_think,
        "n_samples": n,
        "load_seconds": load_s,
        "total": aggregate(per_sample),
        "by_mode": {k: aggregate(v) for k, v in by_mode.items()},
        "by_difficulty": {k: aggregate(v) for k, v in by_difficulty.items()},
        "by_domain": {k: aggregate(v) for k, v in by_domain.items()},
    }

    out_path = RESULTS_DIR / f"{spec.slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": per_sample}
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[{spec.slug}] saved {out_path}")

    del model
    gc.collect()
    return summary


def render_summary_table(summaries: list[dict], out_md: Path) -> None:
    lines = []
    lines.append("# Phase 1 Quality Benchmark — issue #9\n")
    lines.append("Touchup quality on gold-standard test split.\n")
    lines.append("Metrics aggregated across both proofread + rewrite prompts.\n")
    lines.append("")
    lines.append("| Model | Role | n | EM | JSON | Avg CER | Avg sim | Lat (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            lines.append(f"| {s['slug']} | {s.get('role','?')} | - | ERR | - | - | - | - |")
            continue
        t = s["total"] or {}
        lines.append(
            f"| {s['slug']} | {s['role']} | {t.get('n','-')} | "
            f"{t.get('exact_match', 0) * 100:.1f}% | "
            f"{t.get('json_ok', 0) * 100:.1f}% | "
            f"{t.get('avg_cer', 0):.4f} | "
            f"{t.get('avg_semantic_similarity', 0):.4f} | "
            f"{t.get('avg_latency_ms', 0):.0f} |"
        )

    lines.append("\n## Per mode\n")
    lines.append("| Model | Mode | EM | CER | Sim |")
    lines.append("|---|---|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            continue
        for mode, m in (s.get("by_mode") or {}).items():
            if not m:
                continue
            lines.append(
                f"| {s['slug']} | {mode} | {m['exact_match'] * 100:.1f}% | "
                f"{m['avg_cer']:.4f} | {m['avg_semantic_similarity']:.4f} |"
            )

    lines.append("\n## Per difficulty\n")
    lines.append("| Model | Difficulty | n | EM | CER | Sim |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            continue
        for diff, m in (s.get("by_difficulty") or {}).items():
            if not m:
                continue
            lines.append(
                f"| {s['slug']} | {diff} | {m['n']} | "
                f"{m['exact_match'] * 100:.1f}% | {m['avg_cer']:.4f} | {m['avg_semantic_similarity']:.4f} |"
            )

    lines.append("\n## Per domain\n")
    lines.append("| Model | Domain | n | EM | CER | Sim |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for s in summaries:
        if "error" in s:
            continue
        for dom, m in (s.get("by_domain") or {}).items():
            if not m:
                continue
            lines.append(
                f"| {s['slug']} | {dom} | {m['n']} | "
                f"{m['exact_match'] * 100:.1f}% | {m['avg_cer']:.4f} | {m['avg_semantic_similarity']:.4f} |"
            )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Gold-standard JSONL")
    p.add_argument("--limit", type=int, default=0, help="Cap test samples (0=all)")
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument(
        "--only", nargs="+", default=None,
        help="Restrict to these slugs (e.g. qwen2.5-0.5B-baseline)",
    )
    p.add_argument(
        "--skip", nargs="+", default=None,
        help="Skip these slugs",
    )
    args = p.parse_args()

    samples = load_test_set(args.gold, args.limit or None)
    print(f"Loaded {len(samples)} gold samples from {args.gold}")

    # One-time SBERT load.
    from sentence_transformers import SentenceTransformer
    import jiwer

    print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    selected = CANDIDATES
    if args.only:
        selected = [s for s in selected if s.slug in set(args.only)]
    if args.skip:
        selected = [s for s in selected if s.slug not in set(args.skip)]

    if not selected:
        print("No candidates matched filters.", file=sys.stderr)
        return 1

    summaries = []
    for spec in selected:
        try:
            summary = evaluate_model(spec, samples, sim_model, jiwer, args.max_tokens)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[{spec.slug}] FATAL: {e}")
            traceback.print_exc()
            summary = {"slug": spec.slug, "repo": spec.repo, "role": spec.role, "error": str(e)}
        summaries.append(summary)

    # Render summary even if some failed.
    out_md = RESULTS_DIR / "summary_table.md"
    render_summary_table(summaries, out_md)
    print(f"\nSummary table: {out_md}")
    # Also keep an aggregated json.
    (RESULTS_DIR / "all_summaries.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
