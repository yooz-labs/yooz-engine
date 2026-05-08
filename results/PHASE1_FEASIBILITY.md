# Phase 1 — TurboQuant Feasibility for Yooz Engine

**Issue:** yooz-labs/yooz-engine#10
**Branch:** `feature/issue-10-turboquant-kv`
**Date:** 2026-05-02
**Status:** Phase 1 complete; recommendation below.

---

## TL;DR

**Recommendation: Option 3 — depend on the SharpAI fork of `mlx-swift` + `mlx-swift-lm` as Swift Packages, behind a default-off `kvCompression` config flag.**

The TurboQuant integration is non-trivial: it requires a new C++ op (`turbo_encode_k/v`, `turbo_decode_k/v`), a hand-written Metal SDPA kernel that dequantizes 3-bit indices on the fly, and a `KVCacheSimple` extension that orchestrates hot-window eviction. SharpAI's fork already ships all three pieces under MIT, is actively maintained (last push today), and is wired into a real production app (SwiftLM). Re-implementing the C++/Metal layer in yooz-engine is a multi-week effort that will diverge from the upstream and is out of scope for issue #10.

The cost is one upstream change: swap our `mlx-swift` and `mlx-swift-lm` package URLs from `ml-explore/*` to `SharpAI/*`. Because SharpAI's fork is layered on top of upstream and only adds new symbols (no breaking changes to existing API surface — verified for `KVCacheSimple`, `MLXFast`, `KVCache` protocol), this swap is non-invasive for the rest of the engine.

If we later need to upstream the work or reduce supply-chain risk, Option 1 (vendor the SharpAI patches into our own fork) becomes a clean follow-up. Option 2 (subclass `KVCacheSimple` from the outside) does **not** work because the Metal SDPA kernel that consumes `polarKeys`/`polarValues` lives inside the C++/Metal layer of `mlx-swift` itself; it cannot be reached from a Swift subclass.

---

## What TurboQuant actually does

From arXiv 2504.19874 (Zandieh et al., *TurboQuant*, AISTATS / ICLR 2026) and the SharpAI implementation:

1. **L2-normalize** each `head_dim`-sized vector (D=128 or D=256) onto the unit sphere.
2. **Random Walsh-Hadamard rotation** (in-place, `O(d log d)`, deterministic seed=42): `D1 · FWHT · D2`. WHT redistributes outlier channels so the rotated vector is approximately Gaussian `N(0, 1/sqrt(d))`.
3. **Lloyd-Max 3-bit centroid lookup** (8 non-linear bins matched to N(0,1/d), constant table). Pack into `uint8` (3 bits × 128 = 48 bytes per K or V vector).
4. **K cache only:** add **1-bit QJL** (Quantized Johnson-Lindenstrauss) residual sign on top of the 3-bit indices. Storage: K = 4.25 b/dim, V = 3.125 b/dim, **average ≈ 3.6 b/dim ≈ 4.4× compression vs fp16** (or 3.5× counting the fp16 hot-window in a real workload).
5. **Metal SDPA dequant** runs inline in the attention kernel: unpack 3-bit index → centroid value from a `constant float[8]` table → inverse rotate → reconstruct.

What it is **not**:

- Not weight quantization. KV cache only. Forward-pass FLOPs are unchanged. A 7B with TurboQuant still needs ~4× more compute per token than a 1.7B (auto-memory note `reference_turboquant.md` already flags this).
- Not a free upgrade for short prompts. Cost-benefit only kicks in at long context. SharpAI gates compression behind `turboMinActivationTokens = 2048` and a `turboHotWindowSize = 256` fp16 hot window so short tool-use calls and chat turns pay nothing.

---

## What we already have in yooz-engine

The current LLM module (`YoozEngine/LLM/MLXLLMBackend.swift`) is unusually well-positioned for TurboQuant:

- It already manages the KV cache **directly**, not through `container.generate(input:parameters:)`. We construct `cache = context.model.newCache(parameters:)`, optionally restore `cache[i].state = savedState[i]`, then call the lower-level `MLXLMCommon.generate(input:cache:parameters:context:)`. This exact entry point is where SharpAI's `simple.turboQuantEnabled = true` flip happens.
- The `LLMBackend` actor protocol already takes a config-style configuration (`bundleIdentifier`, `modelType`). Adding a `kvCompression` enum is mechanical.
- `LLMModelType` is already enum-driven, so a new "Yooz-Heavy" 7B variant can be added cleanly later (out of scope here, but unblocked by this work).

## What we'd be picking up from SharpAI

Across the two forks the relevant new code is roughly:

| Layer | File(s) | Lines added (est) |
|---|---|---|
| C++ TurboQuant primitives (CPU reference) | `mlx-swift/Source/Cmlx/mlx/mlx/fast/turbo_quant.h` | ~400 |
| C++ MLX-array op layer (`turbo_encode_k/v`, `turbo_decode_k/v`) | `mlx-swift/Source/Cmlx/mlx/mlx/fast.{h,cpp}` | ~300 |
| Metal SDPA dequant kernel (3-bit unpack + centroid lookup + inverse WHT in the hot loop) | `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h` | ~250 |
| C ABI bridge | `mlx-swift/Source/Cmlx/include/mlx/c/fast.h` | ~40 |
| Swift binding (`MLXFast.turboQuantEncode/turboDecodeK/turboDecodeV`) | `mlx-swift/Source/MLX/MLXFast.swift` | ~70 |
| `KVCacheSimple` hot-window eviction + `polarKeys/polarValues` | `mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` | ~250 |
| `AttentionUtils` decode path | `mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift` | ~50 |
| C atomic telemetry aggregator | `mlx-swift/Source/Cmlx/mlx/mlx/core/moe_stream_op.cpp` (small) | ~80 |

Total ≈ 1.4k lines across two C++/Swift packages, ~half of which is Metal kernel code. This is the work we'd be re-implementing if we went any other route.

## License & supply chain

Both `SharpAI/mlx-swift` and `SharpAI/mlx-swift-lm` are MIT-licensed forks of the official Apple `ml-explore/*` repos (verified via `gh api`). Both are actively maintained — `SharpAI/mlx-swift-lm` was pushed today (2026-05-02). The `e707b7f` (mlx-swift) and `38d7ff2` (mlx-swift-lm) commits the SwiftLM submodule pins explicitly note "Restore downstream compatibility" and "Suppress verbose Metal compile logging," meaning the fork is being kept rebased on upstream rather than diverging.

Risk: SharpAI is a single-org fork. If they go inactive, the cost of taking over the patches is on us. Mitigation: vendor the fork into a `yooz-labs/mlx-swift-yooz` and `yooz-labs/mlx-swift-lm-yooz` if/when issue #10 ships and we want to lock the supply chain. The fork has clean turbo-only changes that we can rebase ourselves.

## Options compared

### Option 1 — Patch `mlx-swift{,-lm}` directly (yooz-labs forks)

Pros: total control, can drop telemetry we don't need, can pin to known-good commits.
Cons: takes on permanent maintenance burden. Doubles our review surface for every MLX upstream sync. Doesn't actually de-risk anything until SharpAI goes inactive.

### Option 2 — Side-load via a custom `KVCache` subclass

**Does not work.** The `KVCache` protocol exposes `.state` as `[MLXArray]`, but the Metal SDPA kernel that runs during attention reads `polarKeys` / `polarValues` directly through the C++ layer, bypassing the Swift cache abstraction. Dequantizing inside `KVCache.state` would reconstruct the full fp16 buffer back to memory — exactly the bytes we're trying to avoid keeping resident. The whole point of the fast-path is that the Metal kernel reads the packed buffer and dequantizes during the attention dot product. That requires changes to both the C++ op layer and the SDPA kernel, neither of which is reachable from outside the `mlx-swift` package.

### Option 3 — Depend on SharpAI's `mlx-swift{,-lm}` fork as Swift Packages **(recommended)**

Pros:
- Zero new Metal/C++ code in yooz-engine.
- The `KVCacheSimple` API stays identical; we get `simple.turboQuantEnabled = true` as a one-liner inside `MLXLLMBackend.generate(...)`.
- SharpAI ships well-tested defaults (hot window 256, activation threshold 2048, head-dim guards for unsupported models).
- Fork is upstream-compatible, so a future MLX upgrade in yooz-engine is a single package-pin bump.

Cons:
- We add a third-party org as a critical dependency. (Mitigated by MIT license + ability to vendor later.)
- Pin must move from `mlx-swift {0.21.2, ...}` to `SharpAI/mlx-swift {e707b7f, ...}`. Requires regression-testing the engine's existing LLM path (system-prompt KV caching).

### Option 4 — Lift just the Metal kernels into yooz-engine

Possible in theory (the Metal kernel and `turbo_quant.h` are self-contained), but the kernel **only works inside the `mlx-swift` SDPA op tree**. Calling it directly would require us to either (a) re-route all attention through our own kernel — which means re-implementing every model architecture that ships in `mlx-swift-lm` (Llama, Qwen, Mistral, Gemma, ...) — or (b) re-vendor the entire SDPA op subtree, which is ~80% of Option 1.

---

## Concrete plan if we go with Option 3

### Phase 2 (Python-first de-risk, MLX Python)

We still want a Python-side validation of the paper's headline numbers (3.5× compression, +22.8% decode at 32K, no quality regression on Yooz TouchUp prompts). The Python `helgklaizar/turboquant_mlx` repo is a drop-in `apply_turboquant_cache(model)` monkey-patch on `mlx_lm`. Plan:

1. Set `HF_HOME=/Volumes/S1/yooz/research/issue-10/models/hf_cache`.
2. `uv venv /Volumes/S1/yooz/research/issue-10/.venv && uv pip install -e external/turboquant_mlx mlx mlx-lm`.
3. Bench `Qwen3-1.7B-MLX-4bit` and `Qwen2.5-7B-Instruct-MLX-4bit` at 1K / 4K / 16K / 32K context, FP16 vs 3-bit vs 2-bit. Capture KV bytes, prefill TTFT, decode tokens/s.
4. Run yooz-benchmark TouchUp prompts through the 1.7B with TurboQuant on. Compare against gold standard for quality regression.
5. CSV+plots → `results/phase2_prototype/`.

Gate to Phase 3: 7B at 32K must fit in <12 GB and decode at >7 tok/s on M4 Pro. Quality drift on TouchUp gold-standard must be ≤0.5%.

### Phase 3 (Swift integration in yooz-engine)

1. Update `project.yml` package URLs:
   - `mlx-swift` → `https://github.com/SharpAI/mlx-swift.git`, pin commit `e707b7f1610e77a32f12e3e28910d6bd3d9ffef1`.
   - `mlx-swift-lm` → `https://github.com/SharpAI/mlx-swift-lm.git`, pin commit `38d7ff2840ab6b91a84b8f168c3cc2539f9356e1`.
   - These are the same commits SwiftLM ships against, so we inherit their test coverage.
2. Add `KVCompressionMode` enum to `LLMBackend.swift`:
   ```swift
   public enum KVCompressionMode: String, Sendable {
       case off
       case turbo3        // 3-bit PolarQuant + QJL on K, 3-bit on V (~3.6 b/dim avg)
       // case turbo2 — out of scope; SharpAI fork only ships 3-bit, and turbo2 has +9..23% PPL
   }
   ```
   Decision: drop `turbo2` from the issue scope. The SharpAI fork only ships 3-bit, and the Phase 1 Hybrid analysis (`docs/turboquant_hybrid_architecture.md`) shows 2-bit V2-affine has +9–23% PPL — not a tradeoff we want for a service that's the source of truth for TouchUp/Grammar.
3. Extend `MLXLLMBackend` to accept `kvCompression: KVCompressionMode` (default `.off`) and flip `simple.turboQuantEnabled` after `newCache(...)`. Touch points: `init`, the `container.perform { context in ... }` block in `generate(...)`, plus a single config plumbing edit through `EngineConfig`.
4. Add a compatibility guard: TurboQuant requires `head_dim ∈ {128, 256}` (or 512 split into 2×256). Yooz-Light (Qwen2.5-0.5B) and Yooz-Quality (Qwen3-1.7B) both use head_dim=128, so they're fine; future Yooz-Heavy (Qwen2.5-7B, head_dim=128) is fine. Document this in the model-loading path; fall back to fp16 with a logger warning.
5. Engine config:
   ```
   /v1/llm/generate request body: { ..., "kv_compression": "off" | "turbo3" }
   engine.config global default: kvCompression: "off"
   ```
6. Re-enable system-prompt KV cache compatibility — `KVCacheSimple.state` (in SharpAI fork) already decodes `polarKeys` back to fp16 when read, so our existing `cachedPromptKVState: [[MLXArray]]` snapshot/restore continues to work. Add a regression test for this exact path.

### Phase 4 (stretch — speculative decoding)

Out of scope for issue #10. SharpAI's `DFlashEngine` already implements 0.5B-draft + 7B-verify combined with TurboKV. Open as follow-up issue once Phase 3 is shipped.

---

## Open questions / risks

1. **Embedded Yooz-Light is 0.5B-instruct-4bit at 276 MB.** It's tiny. TurboKV does **nothing** for it (the activation threshold is 2048 tokens; TouchUp prompts are well under that). The user-visible feature is "now you can run a 7B (Yooz-Heavy)," not "your existing models get faster." Ship messaging needs to reflect that.
2. **Apple Intelligence path (`FoundationModelsBackend`) is unaffected.** TurboQuant only applies to MLX. macOS 26+ users on Apple Intelligence skip this path entirely.
3. **MLX upstream version drift.** Yooz-engine's `project.yml` currently pins `mlx-swift 0.21.2` and `mlx-swift-lm 2.30.3`. SharpAI's fork is rebased on a more recent upstream. We need to verify this doesn't break the existing `MLXLLMBackend.generate()` path. Plan: in Phase 3, keep the current pins as a fallback branch; bump in a separate PR after the integration tests pass.
4. **Telemetry.** SharpAI's `TurboKVTelemetry.logOnce` writes via `@_silgen_name("mlx_turbo_kv_record")` to a C atomic in `moe_stream_op.cpp`. We should expose that count via `/v1/health` so we can tell from the outside whether compression is active. Trivial, but worth budgeting.
5. **Build size.** SharpAI fork adds ~250 lines of Metal kernel code. Ship-impact on `Yooz Engine.app` should be sub-MB. Verify after Phase 3.

## Files referenced

- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/YoozEngine/LLM/MLXLLMBackend.swift` — current MLX backend
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/YoozEngine/LLM/LLMBackend.swift` — protocol + types
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/project.yml` — XcodeGen package pins
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/external/SwiftLM/docs/turboquant_hybrid_architecture.md` — design doc
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/external/SwiftLM/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift` — `turboQuantEnabled` integration
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/external/SwiftLM/mlx-swift/Source/Cmlx/mlx/mlx/fast/turbo_quant.h` — algorithm reference
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/external/SwiftLM/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h` — Metal SDPA dequant kernel
- `/Users/yahya/Documents/git/yooz/engine-issue-10-turbo/external/SwiftLM/Sources/MLXInferenceCore/InferenceEngine.swift:619` — flip-the-flag pattern
- `/Volumes/S1/yooz/research/issue-10/data/turboquant_paper.pdf` — paper

## Decision

Proceed to Phase 2 with the SharpAI dependency plan in mind. Begin Python validation while drafting the `project.yml` pin swap on a side branch. Block Phase 3 on Phase 2 numbers landing within paper claims.
