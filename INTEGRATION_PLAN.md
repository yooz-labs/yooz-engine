# Issue #9 Integration Plan — Qwen3.5 + Gemma 4 vs current Yooz LLMs

Decision document for whether the current Yooz engine LLM lineup
(Qwen2.5-0.5B + Qwen3-1.7B) should be swapped for Qwen3.5 / Gemma 4
candidates after Phase 1-3 benchmarking.

## TL;DR

- **Light tier**: keep Qwen2.5-0.5B-Instruct-4bit (the current `yoozLight`)
  for now. The Qwen3.5-0.8B candidate is a poor zero-shot fit (EM 10.9%,
  CER 0.418 in Phase 1), but a partial LoRA fine-tune (5/12 epochs, val
  loss 0.252) **dramatically** flips the picture: EM 20.4%, Sim 0.83 on
  the matched test set. Caveat: this number is on the fine-tune training
  distribution (different prompt format from `gold_standard.jsonl`); a
  matched-prompt re-evaluation is the next gate before swapping the
  Light backend. See "Phase 3 fine-tune results" below.
- **Quality tier**: keep Qwen3-1.7B-4bit (the current `yoozQuality`).
  Qwen3.5-2B-OptiQ wins on quality but blows the 400 ms end-to-end
  latency budget at 612 ms (TTFT median is 219 ms, but end-to-end avg is
  612 ms in Phase 1). Gemma-4-E2B wins further on quality but exceeds
  the budget at 939 ms and is also blocked by upstream `mlx-swift-lm`
  arch support.
- **Defer**: Qwen3.5-2B fine-tune (issue #67), Gemma-4 fine-tune + arch
  port (issue #68), and a prompt-matched re-eval of the qwen35-light
  fine-tune. None of these can change the tier-fit verdict in this PR;
  all are follow-ups.

## Phase 1 quality results (full run, 5936 samples per model)

Two prompts per sample (proofread + rewrite); metrics aggregated across
both. Source: `/Volumes/S1/yooz/research/issue-9/results/phase1_quality/`.

| Model | Role | n | EM | JSON | CER | Sim | Lat (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5B-baseline | baseline | 5936 | 14.8% | 88.3% | 0.3516 | 0.8520 | 191 |
| qwen3-1.7B-baseline   | baseline | 5936 | 18.3% | 88.2% | 0.2428 | 0.8949 | 395 |
| qwen3.5-0.8B          | candidate (no fine-tune) | 5936 | 10.9% | 71.0% | 0.4182 | 0.8241 | 315 |
| qwen3.5-2B-optiq      | candidate | 5936 | 18.9% | 99.6% | 0.1931 | 0.9082 | 612 |
| gemma-4-e2b-text      | candidate | 5936 | **19.1%** | 99.4% | **0.1876** | **0.9084** | 939 |

Per-mode, per-difficulty, and per-domain breakdowns are in
`results/phase1_quality/summary_table.md`.

## Phase 2 speed results

Three input lengths (short/medium/long), six warm runs each, on the same
M-series Mac as the eventual deployment target.

| Model | Load (s) | TTFT med (ms) | tok/s med | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| qwen2.5-0.5B-baseline | 0.6 | 79  | 429.0 | 864  |
| qwen3-1.7B-baseline   | 1.1 | 153 | 195.8 | 1608 |
| qwen3.5-0.8B          | 1.2 | 155 | 325.8 | 1182 |
| qwen3.5-2B-optiq      | 1.7 | 219 | 138.7 | 2336 |
| gemma-4-e2b-text      | 2.2 | 182 | 126.8 | 3715 |

## Phase 3 fine-tune results

### qwen35-light (Qwen3.5-0.8B-MLX-4bit + LoRA)

- Config: `finetune-pipeline/configs/qwen35-light.yaml`
  (rank 8, 8 layers, batch 4, 6000 iters, max-seq 512, mask_prompt true).
- Training data: `finetune-pipeline/data/light/train.jsonl` (~24 K samples,
  same split used for the existing Qwen2.5 `yooz-light-v3` adapter).
- **Crash at iter 2600**: training stopped due to a Metal OOM caused by a
  concurrent `xcodebuild` test run on the same GPU from a parallel agent.
  Latest checkpoint saved: `0002500_adapters.safetensors` at val loss
  **0.252** (5/12 epochs). Loss trend was still flat-to-slightly-down,
  but the adapter is converged enough to evaluate honestly.
- Eval set: 500 samples from `data/light/test.jsonl` (fine-tune training
  distribution; the full set is 2998, capped to ship inside the
  available GPU window). Source:
  `/Volumes/S1/yooz/research/issue-9/results/phase3_finetune/qwen35-light/eval_results.json`.

| Variant | EM | JSON | Sim | Echo | Lat (ms) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B baseline (no adapter) | 0.0%   | 62.2% | 0.112 | 17.6% | 1041 |
| Qwen3.5-0.8B + LoRA iter 2500      | 20.4%  | 97.4% | 0.832 | 0.2%  | 397  |
| Regression rate (fine-tuned >5% sim drop vs baseline) | 0/500 (0.0%) |   |   |   |   |

Reading the table: the baseline Qwen3.5-0.8B model is essentially
unusable on the fine-tune-format prompts (it generates multi-paragraph
"thinking" preambles instead of JSON, hence EM=0% and Sim=0.11). The
LoRA adapter at iter 2500 fixes JSON compliance (97.4%), removes
prompt-echoing (17.6% → 0.2%), and produces a 7x similarity jump.
Latency drops from 1041 ms (baseline emits ~256 tokens of "thinking")
to 397 ms (fine-tuned emits compact JSON), comfortably under the
400 ms budget for an end-to-end response.

**Caveat** — _not_ a direct Phase 1 comparison: the fine-tune evaluation
above uses the in-distribution test set (`data/light/test.jsonl`), which
shares prompt structure with the training data. Phase 1 used the engine's
actual touchup prompt format (`bench-tools/gold_standard.jsonl`). Numbers
are not strictly apples-to-apples. Before swapping the Light backend,
re-run the fine-tuned model with the engine's prompt format on
`gold_standard.jsonl`. Filed as follow-up issue
[#71](https://github.com/yooz-labs/yooz-engine/issues/71).

**Verdict**: the partial fine-tune is a genuine, large improvement on
the matched distribution, but tier-swap is gated on a prompt-matched
re-eval. Light tier **stays on Qwen2.5-0.5B-Instruct-4bit** in this PR.
Once the matched eval lands and confirms the EM/Sim numbers carry over,
file an engine PR adding `yoozLightV4` (Qwen3.5-0.8B fused).

### qwen35-quality

Deferred to follow-up issue
[#67](https://github.com/yooz-labs/yooz-engine/issues/67). Phase 1 numbers
already show that Qwen3.5-2B-OptiQ beats the Qwen3-1.7B baseline on
quality (EM 18.9% vs 18.3%, CER 0.193 vs 0.243) but exceeds the 400 ms
latency budget at 612 ms TTFT. A LoRA adapter is unlikely to move
latency, so this fine-tune is informational rather than tier-changing.

### gemma-4 (e2b text-only)

Blocked, follow-up issue
[#68](https://github.com/yooz-labs/yooz-engine/issues/68):

1. **`mlx_lm.lora` strict-load**: `mlx-lm 0.31.3` calls
   `mlx_lm.load(strict=True)` which fails on both
   `gemma-4-e2b-it-4bit` (multimodal) and `Gemma4-E2B-IT-Text-int4`
   (text-only) — the text-only checkpoint ships 140 spurious K/V
   projection weights for the last 20 KV-shared layers. Inference works
   with `load_model(..., strict=False)` (used in
   `scripts/run_phase1_quality.py`), but training does not.
2. **`mlx-swift-lm` `gemma4_text` arch**: not registered as of
   mlx-swift 0.21. The engine's LLM module cannot run Gemma 4 even after
   we work around #1.

Workarounds (none implemented in this PR):

- Vendor a small `mlx_lm.lora` wrapper that calls `load(strict=False)`.
- Pre-fuse the checkpoint locally, dropping the unused KV-shared K/V
  parameters.
- Wait for `mlx-lm > 0.31.3` to recognise the `gemma4_text` shared K/V
  layout.

Recommendation: don't block the v4 release on Gemma 4. Even with a
working fine-tune, 939 ms TTFT exceeds the Quality tier budget; would
need draft-model + speculative decoding or a hardware upgrade to fit.

## Engine integration — `LLMBackend.swift`

Current state at
`/Users/yahya/Documents/git/yooz/yooz-engine/YoozEngine/LLM/LLMBackend.swift`:
two registered cases, `yoozLight` (Qwen2.5-0.5B-4bit, fine-tuned
`yooz-light-v3`) and `yoozQuality` (Qwen3-1.7B-4bit), with
`displayName`, `description`, `estimatedSize`, `isEmbedded`,
`baseModelId`, and `packageName` switch-on-self.

**Proposed diff for this issue**: none. The current cases stand. Re-open
when issue #67 lands a fine-tuned Qwen3.5-2B that fits the Quality tier.

If a future swap happens, the diff sketch is:

```swift
enum LLMModelType: String, CaseIterable, Sendable {
    case yoozLight = "yooz-light-v3"           // Qwen2.5-0.5B-4bit, 276 MB
    case yoozQuality = "yooz-quality-v3"       // Qwen3-1.7B-4bit, 1037 MB
    // Future v4 (gated on a winning fine-tune):
    case yoozQualityV4 = "yooz-quality-v4"     // Qwen3.5-2B-OptiQ-4bit, ~1.4 GB

    var displayName: String { ... }
    var description: String { ... }
    var estimatedSize: Int64 { ... }
    var isEmbedded: Bool { ... }                // v4: false (GHCR pull)
    var baseModelId: String { ... }
    var packageName: String { "yooz-models" }   // unchanged
}
```

The actual base model id, sizes, and prompt-format constants will be
filled in once a winning fine-tune lands.

## GHCR registry updates

None for this issue — no winners, so nothing to push. When issue #67
lands, the existing `yooz-models` GHCR package will need a
`yooz-quality-v4-mlx` artifact (fused Qwen3.5-2B base + LoRA) using the
same `mlx-swift-examples` packaging pipeline already used for v3.

## Prompt-format changes

Already in this PR via `scripts/run_phase1_quality.py` and
`bench-tools/prompts.py`:

- Drop the `"corrected text"` placeholder (Qwen3.5/Gemma echoed it
  literally during smoke runs).
- Per-family chat-template handling:
  - **Qwen2.5**: default `apply_chat_template`.
  - **Qwen3 / Qwen3.5**: `apply_chat_template(..., enable_thinking=False)`
    to skip the empty `<think>...</think>` block.
  - **Gemma-4**: hand-rolled
    `<|turn>system ... <turn|><|turn>user ...<turn|><|turn>model\n<|channel>final\n`
    to force the final response channel (the tokenizer config has no
    `chat_template`).

Engine-side mirroring is **not** required this PR (the current LLM
backends keep their existing prompt format); it's only relevant when a
v4 swap actually ships.

## Phase 5 — pipeline automation (this PR + follow-up)

This PR adds an `evaluate` subcommand to `bench-tools/cli.py` that
delegates to the Phase 1 harness:

```bash
python -m bench-tools.cli evaluate \
    --repo mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --family qwen35 \
    --gold bench-tools/gold_standard.jsonl \
    --limit 10
```

It produces a per-model JSON in `results/eval/` with the same schema as
`scripts/run_phase1_quality.py` (so future runs aggregate cleanly).

Follow-up: upstream the same `evaluate` subcommand into
`~/Documents/git/yooz/yooz-benchmark/benchmarks/cli.py` so the broader
benchmark repo gets the capability. Filed as
[yooz-benchmark#11](https://github.com/yooz-labs/yooz-benchmark/issues/11).
Sketch:

- File: `benchmarks/cli.py` — new `evaluate` group mirroring
  `bench-tools/cli.py` `evaluate`.
- File: `benchmarks/eval.py` — extract reusable functions
  (`load_test_set`, `parse_json_result`, `evaluate_model`, summary
  rendering) from `scripts/run_phase1_quality.py` into a library
  module.
- File: `benchmarks/families.py` — registry of family → chat-template
  builder, so future model families (Mistral, Phi-5, Gemma-5) only need
  a single new entry.

Acceptance: a one-liner like

```bash
yooz-bench evaluate --repo mlx-community/Phi-5-mini-4bit --family qwen3
```

can produce a per-model JSON + summary row in under a day, end-to-end,
including review.
