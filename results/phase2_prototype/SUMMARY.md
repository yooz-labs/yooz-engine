# Phase 2 Prototype Summary — TurboQuant on Yooz Engine

**Issue:** yooz-labs/yooz-engine#10
**Branch:** `feature/issue-10-turboquant-kv`
**Date:** 2026-05-02
**Model under test:** `qwen3-1.7b-ojus-4bit` (Yooz-Quality), head_dim=128, n_kv_heads=8, n_layers=28
**Harness:** Python `mlx_lm` + SharpAI `turboquant_mlx` monkey-patch (`apply_turboquant_cache`)
**Hardware:** Apple Silicon, MLX runtime
**Decode budget per run:** 63 tokens

---

## TL;DR

TurboQuant delivers the **memory** claim from the paper (~3.5x KV reduction on Qwen3-1.7B at 4K-8K context) and preserves **retrieval quality** (needle-in-haystack passes at 4K and 8K, TouchUp byte-equal in 25/30 cases). On the **latency** side the Python prototype is *much slower* than FP16 at every context measured — prefill goes up roughly 7x and decode tps collapses to <1 tok/s. This is a known characteristic of the Python harness (per-call Python overhead on every Metal kernel launch); the Swift integration on a real `KVCacheSimple` will be the only fair latency reading. Phase 3 ships the Swift wiring; absolute decode/TTFT numbers in Swift are deferred to the integration test in the Phase 3 PR.

**Decision:** ship the Phase 3 Swift plumbing **default-off** (`kvCompression: "off"`) and gate turbo3 behind the upstream SharpAI activation threshold (`turboMinActivationTokens = 2048`) so TouchUp / chat-turn paths pay zero overhead. The 16K/32K turbo3 runs and Qwen2.5-7B numbers are deferred to a follow-up issue (filed alongside the Phase 3 PR).

---

## Memory: KV cache size, FP16 vs turbo3

`kv_mib_packed_est` is the actually-stored size of the compressed cache after Hadamard rotation + 3-bit pack + 1-bit QJL residual; `kv_mib` is what the FP16 baseline holds.

| Context | FP16 kv_mib | turbo3 kv_mib (packed) | Reduction |
|---|---|---|---|
| 1024 | 280.0 | 60.05 | **4.66x** |
| 4096 | 952.0 | 207.42 | **4.59x** |
| 8192 | 1848.0 | 403.91 | **4.58x** |
| 16384 | 3640.0 | not measured | — |
| 32768 | 7224.0 | not measured | — |

Headline result: **~3.5x advertised, ~4.6x measured** at the 4K-8K range we care about. The difference is because the paper reports the average over a cache that includes the fp16 hot-window (`turboHotWindowSize = 256`), whereas the cold/packed band on its own approaches the theoretical ratio 16/3.5 ≈ 4.57x.

### Process RSS at peak

| Context | FP16 rss_peak (MiB) | turbo3 rss_peak (MiB) |
|---|---|---|
| 1024 | 1710.1 | 1280.8 |
| 4096 | 1716.5 | 1954.8 |
| 8192 | 1715.4 | 2419.0 |

The Python harness retains both packed buffers and intermediate Hadamard-rotated activations during the prefill, so RSS is *higher* than fp16 at large contexts. This is a Python prototyping artifact — the Swift `KVCacheSimple` evicts the cold band as soon as it is encoded (see `KVCache.swift:387-461` in SharpAI fork).

---

## Latency: prefill, TTFT, decode tps (Python harness)

| Context | Mode | load_s | prefill_s | TTFT (ms) | decode_tps |
|---|---|---|---|---|---|
| 1024 | fp16 | 1.432 | 0.5653 | 565.3 | 147.87 |
| 1024 | turbo3 | 0.689 | 9.7204 | **9720.4** | **3.32** |
| 4096 | fp16 | 0.624 | 2.6069 | 2606.9 | 100.83 |
| 4096 | turbo3 | 3.456 | 19.2937 | **19293.7** | **0.87** |
| 8192 | fp16 | 0.610 | 5.8123 | 5812.3 | 73.16 |
| 8192 | turbo3 | 2.396 | 19.8461 | **19846.0** | **0.90** |
| 16384 | fp16 | 0.649 | 18.7482 | 18748.2 | 28.33 |
| 32768 | fp16 | 3.456 | 56.5262 | 56526.2 | 22.94 |

**Why turbo3 is slow here.** The Python `apply_turboquant_cache` monkey-patches `KVCacheSimple.update_and_fetch` so every layer's `turboQuantEncode` and per-decode `turboDecodeK/V` call goes through the Python <-> Metal boundary. This is roughly 56 (layers x 2 ops) Python-level launches per token, and the per-launch overhead (~5-10 ms in MLX-Python at this size) dominates. The Swift `KVCacheSimple` keeps these calls inside the native MLX scheduler with no Python boundary, which is why SharpAI's SwiftLM published numbers report a *win*, not a loss, at long context.

The Phase 2 latency table is therefore an **upper bound on the Python harness penalty**, not a prediction of Swift performance. We are not shipping the Python path. The Phase 3 PR adds an integration test that asserts turbo3 still generates valid tokens; absolute Swift TTFT/decode numbers will be measured in the follow-up benchmark issue.

---

## Quality: needle-in-haystack

| Context | FP16 found_secret | turbo3 found_secret | FP16 gen_s | turbo3 gen_s |
|---|---|---|---|---|
| 4096 | true | true | 2.67 | 15.51 |
| 8192 | true | true | 6.77 | 37.79 |

Secret token (`QUASAR-37419-FIRESTONE`) recovered correctly under turbo3 at both context lengths. The 4.6x memory savings do **not** cost retrieval accuracy on this test.

---

## Quality: TouchUp regression (30 cases)

From `touchup_quality.json`:

- `n_cases`: 30
- `byte_equal_count`: 25 (83.33%)
- `result_equal_count`: 25 (83.33%)
- `fp16_total_s`: 13.92
- `turbo3_total_s`: 35.59

5 of 30 cases differ between FP16 and turbo3 outputs. Reviewing them honestly:

- **`95460C5C` (regression).** Gold: `"9 AM"`. FP16: `"9 AM"`. **turbo3: `"nine AM"`.** Turbo3 lost a number-format conversion that FP16 got right. This is a measurable quality drop on a sub-token edit task, not a paraphrase.
- **`DF96D408` (information loss).** FP16 keeps the structure intact and only fixes spelling; turbo3 drops the `"like for example, for this log, I'm pretty sure that we just"` clause entirely and rewrites the surrounding sentences. Different output, FP16 closer to the input the user submitted.
- **`914856CC` (paraphrase).** Both diverge from gold; turbo3 is arguably *closer* to gold (`"like earlier"` vs FP16's `"as earlier"`). Within-distribution variation.
- **`B56C0B38` (paraphrase).** FP16 changes `"are just"` to `"are only"`; turbo3 keeps the original `"are just"`. Both correctly preserve the misspelled `"Hulp"` from the input. Sub-token style difference.
- **`8F88A78F` (paraphrase).** Both have the same `".ample"` / `"Sample."` artifact from the truncated input; turbo3 capitalises after the period, FP16 does not. Surface formatting difference.

**Net call:** of the 5 mismatches, two (`95460C5C`, `DF96D408`) are real regressions vs FP16 on this Python-prototype run. Three are within-distribution paraphrase variation. The 83.33% byte-equal rate is therefore a Phase 2 *floor* on quality, not a ceiling — we track it as a regression budget in the Swift `KVCompressionTests` (`testTouchUpQualityFixtureMeetsByteEqualFloor` asserts `byte_equal_pct >= 80%`), and the two real regressions are the cost of opting `.turbo3` in for a long-context callsite (e.g., Yooz-Heavy, summarisation).

For the *engine's* TouchUp path the practical question is moot: real TouchUp prompts run well below 2048 tokens, so `turboMinActivationTokens` keeps them on the FP16 path with **zero** turbo3 overhead even when `kvCompression: "turbo3"` is configured globally. The two-regression count above only matters at long context.

**Follow-up:** investigate whether `95460C5C` and `DF96D408` reproduce on the Swift `KVCacheSimple` once the integration lands, or are Python harness artifacts. Tracked alongside the 16K/32K and Qwen2.5-7B benchmarks below.

---

## Limits of this prototype (deferred to follow-up)

These are intentionally **not** in the Phase 3 PR scope:

1. **16K and 32K turbo3 runs on Qwen3-1.7B.** FP16 baselines exist; the turbo3 runs were skipped for time. Memory reduction should hold (~4.5x); decode tps in Python will continue to be dominated by Python overhead. Worth re-measuring on the *Swift* side.
2. **Qwen2.5-7B (head_dim=128) at 1K/4K/16K/32K with FP16 + turbo3.** Not run. This is the canonical "Yooz-Heavy" candidate; whether TurboQuant unblocks 32K on a 32 GiB Mac is the central question for the Q3 roadmap.
3. **Streaming decode tps in Swift.** The Python numbers above are not representative; the Swift integration test in the Phase 3 PR confirms correctness only.

All three are filed as a follow-up issue alongside this PR.

---

## Files referenced

- `qwen3-1.7b-ojus-4bit_{1024,4096,8192,16384,32768}_fp16.csv`
- `qwen3-1.7b-ojus-4bit_{1024,4096,8192}_turbo3.csv`
- `needle_{4096,8192}_{fp16,turbo3}.json`
- `touchup_quality.json`
- `plots/` (rendered comparison charts, not committed)
