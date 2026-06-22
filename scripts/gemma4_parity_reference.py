#!/usr/bin/env python3
"""Generate the Python mlx-lm reference for the Swift Gemma4 parity test (#184).

The Swift engine loads Gemma4 through the ``mlx-swift-lm`` fork pinned in
project.yml (yooz-labs/mlx-swift-lm), which shares the same MLX C++/Metal core
as Python ``mlx``. So greedy (argmax)
decoding on identical quantized weights is expected to match token-for-token.
This script captures that reference so the Swift side can assert parity offline.

Run with the exact version Infinite validated on:

    uv run --with mlx-lm==0.31.3 python scripts/gemma4_parity_reference.py

The 12B ``gemma4_unified_text`` row has no implementation in mlx-lm (only
``gemma4`` / ``gemma4_text``); its only Python reference is mlx-vlm's text path,
so that row is generated through mlx-vlm instead. The two engines need different
``--with`` envs, so ``--model all`` covers only the mlx-lm rows; request the 12B
row explicitly:

    uv run --with mlx-vlm==0.6.3 python scripts/gemma4_parity_reference.py --model 12b

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
# `gemma4` / `gemma4_text` models reference through mlx-lm.
MODELS = {
    "e4b": "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
    "26b-a4b": "mlx-community/gemma-4-26b-a4b-it-4bit",
}

# `gemma4_unified_text` (the 12B family, #187) is absent from mlx-lm, so its
# reference comes from mlx-vlm's text path instead. Same prompt + greedy decode.
VLM_MODELS = {
    "12b": "mlx-community/gemma-4-12B-it-4bit",
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


def build_reference_vlm(repo: str, max_tokens: int) -> dict:
    """Reference for `gemma4_unified_text` via mlx-vlm's text path.

    mlx-lm has no `gemma4_unified_text`, so the 12B family is referenced through
    mlx-vlm. mlx-vlm reads the same HF tokenizer_config the Swift engine's
    swift-transformers stack does, so the chat template matches (unlike the
    mlx-lm-vs-swift-transformers preamble difference seen on the 26B row).
    """
    import mlx_vlm
    from mlx_vlm import load as vlm_load, stream_generate as vlm_stream
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = vlm_load(repo)
    formatted = apply_chat_template(
        processor, model.config, PROMPT_TEXT, num_images=0
    )

    generated_ids: list[int] = []
    text = ""
    finish_reason = None
    for resp in vlm_stream(
        model, processor, formatted, max_tokens=max_tokens, temperature=0.0
    ):
        text += resp.text
        if resp.token is not None:
            generated_ids.append(int(resp.token))
        finish_reason = resp.finish_reason

    if not generated_ids:
        raise RuntimeError(
            f"mlx-vlm stream_generate yielded no tokens for {repo!r}; check the "
            "model load and max_tokens before writing a degenerate reference."
        )
    if finish_reason is None:
        raise RuntimeError(
            f"mlx-vlm stream_generate completed without setting finish_reason for "
            f"{repo!r}; the fixture would record null. Check the mlx-vlm version's "
            "response-object contract before writing the reference."
        )

    return {
        "model_repo": repo,
        "prompt_text": PROMPT_TEXT,
        "prompt_formatted": formatted if isinstance(formatted, str) else None,
        "max_tokens": max_tokens,
        "generated_token_ids": generated_ids,
        "generated_text": text,
        "finish_reason": finish_reason,
        "sampler": "greedy/argmax (temperature=0.0)",
        "reference_engine": f"mlx-vlm=={mlx_vlm.__version__}",
    }


def main() -> None:
    all_tags = sorted(MODELS) + sorted(VLM_MODELS)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=all_tags + ["all"], default="all"
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / (
        "YoozEngine/Infinite/results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # `all` covers only the mlx-lm rows: the VLM rows need a different `--with`
    # env (mlx-vlm), so importing mlx_vlm under an mlx-lm-only run would crash
    # mid-batch. Request VLM rows (e.g. 12b) explicitly.
    tags = sorted(MODELS) if args.model == "all" else [args.model]
    for tag in tags:
        if tag in VLM_MODELS:
            ref = build_reference_vlm(VLM_MODELS[tag], args.max_tokens)
            gen_count = len(ref["generated_token_ids"])
        else:
            ref = build_reference(MODELS[tag], args.max_tokens)
            gen_count = len(ref["generated_token_ids"])
        out_path = out_dir / f"gemma4_{tag}_parity_reference.json"
        out_path.write_text(json.dumps(ref, indent=2) + "\n")
        print(f"[{tag}] wrote {out_path}")
        print(f"[{tag}] gen_tokens={gen_count} finish={ref['finish_reason']}")
        print(f"[{tag}] text: {ref['generated_text']!r}")


if __name__ == "__main__":
    main()
