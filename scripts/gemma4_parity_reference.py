#!/usr/bin/env python3
"""Generate the Python mlx-lm reference for the Swift Gemma4 parity test (#184).

The Swift engine loads Gemma4 through the ``mlx-swift-lm`` fork pinned in
project.yml (yooz-labs/mlx-swift-lm), which shares the same MLX C++/Metal core
as Python ``mlx``. So greedy (argmax)
decoding on identical quantized weights is expected to match token-for-token.
This script captures that reference so the Swift side can assert parity offline.

Run with the exact version Infinite validated on:

    uv run --with mlx-lm==0.31.3 python scripts/gemma4_parity_reference.py

Writes ``YoozEngine/Infinite/results/gemma4_<tag>_parity_reference.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

# Fixed, deterministic instruction. Short answer with stable structure so a
# greedy decode is reproducible and easy to eyeball in the reference file.
PROMPT_TEXT = "List the first five prime numbers, separated by commas."

# Catalog rows the engine gates behind swiftRuntimeSupported (issue #184).
MODELS = {
    "e4b": "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
    "26b-a4b": "mlx-community/gemma-4-26b-a4b-it-4bit",
}


def build_reference(repo: str, max_tokens: int) -> dict:
    model, tokenizer = load(repo)

    messages = [{"role": "user", "content": PROMPT_TEXT}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if isinstance(prompt_ids, str):
        # Older tokenizers ignore tokenize=True and return the formatted text;
        # iterating that would write per-character ordinals as "token ids".
        raise RuntimeError(
            "apply_chat_template returned text, not token ids; this tokenizer "
            "build does not honor tokenize=True."
        )
    prompt_ids = [int(t) for t in prompt_ids]

    # First-step logits: most robust single-shot signal (no sampling/cascade).
    logits = model(mx.array(prompt_ids)[None])[:, -1, :]
    top5 = [int(t) for t in mx.argsort(-logits[0])[:5].tolist()]

    sampler = make_sampler(temp=0.0)  # argmax / greedy
    generated_ids: list[int] = []
    text = ""
    finish_reason = None
    for resp in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler
    ):
        generated_ids.append(int(resp.token))
        text += resp.text
        finish_reason = resp.finish_reason

    if not generated_ids:
        raise RuntimeError(
            f"stream_generate yielded no tokens for {repo!r}; check the model "
            "load and max_tokens before writing a degenerate reference."
        )

    return {
        "model_repo": repo,
        "prompt_text": PROMPT_TEXT,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "max_tokens": max_tokens,
        "first_step_top5_token_ids": top5,
        "generated_token_ids": generated_ids,
        "generated_text": text,
        "finish_reason": finish_reason,
        "sampler": "greedy/argmax (temp=0.0)",
        "reference_engine": "mlx-lm==0.31.3",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=sorted(MODELS) + ["all"], default="all"
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / (
        "YoozEngine/Infinite/results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = sorted(MODELS) if args.model == "all" else [args.model]
    for tag in tags:
        ref = build_reference(MODELS[tag], args.max_tokens)
        out_path = out_dir / f"gemma4_{tag}_parity_reference.json"
        out_path.write_text(json.dumps(ref, indent=2) + "\n")
        print(f"[{tag}] wrote {out_path}")
        print(f"[{tag}] prompt_tokens={ref['prompt_token_count']} "
              f"gen_tokens={len(ref['generated_token_ids'])}")
        print(f"[{tag}] text: {ref['generated_text']!r}")


if __name__ == "__main__":
    main()
