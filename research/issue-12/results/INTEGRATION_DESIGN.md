# Phase 3 — Integration design

## Single-model verdict

> **Can Qwen3-ASR-1.7B replace BOTH Parakeet TDT and FastConformer with
> a single model at acceptable latency?**

**Not as a default replacement.** The 4.6x latency multiplier on warm
inference (0.07 s -> 0.31 s on a 5 s clip) is too risky to drop on
top of Whisper / Notes voice-keyboard UX. Instead, ship Qwen3-ASR-1.7B
as an opt-in **"Multilingual (Preview)"** backend, default Parakeet
TDT for everyone except Arabic, and let the preview earn the default
slot through telemetry over the 4-week graduation window described
below.

Backing numbers (Phase 1 + Phase 2, all on this Mac, 16 kHz mono):

| Language | Model | WER | warm latency on 5 s clip | RTFX |
| --- | --- | ---: | ---: | ---: |
| English | Parakeet TDT 0.6B | 0.069 | 0.07 s | 75x |
| English | Qwen3-ASR-1.7B-8bit | **0.063** | 0.31 s | 16x |
| Arabic | Qwen3-ASR-1.7B-8bit | **0.067** | ~0.5 s on real utterances | ~16x |
| Persian | Qwen3-ASR-1.7B-8bit | 0.283 | ~0.9 s on real utterances | ~16x |
| Hebrew | Qwen3-ASR-1.7B-8bit | 0.828 | ~1.0 s | not viable |

The 240 ms gap on a 5 s utterance is not a UX-killer for push-to-talk
dictation, but it is meaningful for streaming chunks (sub-1 s) and
for users who type one short word at a time. Streaming-mode WER is
also unmeasured (Phase 1 used offline `generate()`; the upstream
`stream_transcribe` path needs its own benchmark before any default
flip). Treat streaming as the open risk.

## Rollout shape

The three product surfaces this rollout touches (Whisper, Notes,
the engine's `/v1/stt/*` API) all share the same "default backend
per language" plumbing, so the rollout is configuration, not code
forking. The shape is:

1. **Default stays Parakeet TDT for English and other Parakeet-supported
   languages.** A 4.6x slower default would regress every existing
   English user for a quality bump (0.069 -> 0.063 WER) that no one
   asked for. Parakeet keeps the English seat.
2. **Qwen3-ASR-1.7B ships as an opt-in `qwen3_asr_preview` backend.**
   It joins `parakeet`, `fast_conformer`, and `apple_stt` in
   `STTBackend`, surfaces in the engine config as
   `stt.backend = "qwen3_asr_preview"`, and is labeled as
   *Multilingual (Preview / alpha)* in client UI so users opt in with
   eyes open.
3. **Arabic flips to Qwen3-ASR-1.7B as its default, conditional on a
   FastConformer-ar head-to-head.** Qwen3-ASR-1.7B Arabic is 6.7% WER
   on the yooz-arabic 25-utterance set; FastConformer-ar was not
   measured in Phase 2 (Swift-only, no Python API). The flip ships
   only after the engine HTTP service runs FastConformer-ar against
   the same 25 utterances and Qwen3-ASR wins by >= 3 pp WER at
   comparable or better latency. Until that head-to-head lands,
   Arabic stays on FastConformer-ar; users can opt into the preview
   to try Qwen3-ASR Arabic early.
4. **Persian stays on FastConformer-fa** until the head-to-head
   follow-up resolves it. Qwen3-ASR Persian (28.3% WER) is usable
   but unproven against the current production model on Persian-only
   workloads.
5. **Hebrew stays on FastConformer-he indefinitely.** Qwen3-ASR has
   no Hebrew training data; 82.8% WER makes it actively worse than
   no transcription. Do not surface the preview backend for Hebrew.

The auto-LID story still holds: if a user picks the preview backend,
language identification is free (no language picker needed); if a
user stays on the default, the existing per-language routing decides.

## Telemetry hooks (privacy-preserving)

Without telemetry, the preview cannot graduate to default; with the
wrong telemetry, it violates Yooz's local-first promise. The contract
is: emit *performance signals only*, no transcript content, all
local, opt-in via the same setting that toggles diagnostics today.

```swift
public struct STTBackendMetrics: Codable, Sendable {
    /// Time to first emitted token, measured from end-of-audio.
    public let timeToFirstTokenMs: Int

    /// Wall clock from first audio frame to final transcription.
    public let endToEndLatencyMs: Int

    /// Audio duration in milliseconds. Combined with end-to-end
    /// gives RTFX without keeping the audio.
    public let audioDurationMs: Int

    /// Backend used for this transcription.
    public let backend: STTBackend

    /// Language requested or auto-detected.
    public let language: STTLanguage

    /// Hardware class bucket: "m1", "m2", "m3", "m4", "intel".
    /// Pre-bucketed at emit time; never includes serial / model identifier.
    public let hardwareClass: String

    /// Streaming vs offline path.
    public let mode: STTInvocationMode

    /// Whether the user kept the result (no edit / no re-record). Optional;
    /// only populated when the host app surfaces "Use this transcription".
    public let userAccepted: Bool?
}
```

What the struct deliberately does *not* carry: transcript text,
audio fingerprints, user identifiers, exact device serials,
timestamps that could be cross-correlated with other event streams.
Sink is local-only (a rolling JSONL under
`~/Library/Application Support/Yooz/engine/telemetry/stt/`); only
aggregates are eligible for review when the user explicitly exports
diagnostics.

## Bundle / install impact and mitigation

| Component | Approx size |
| --- | ---: |
| Qwen3-ASR-1.7B-8bit weights (MLX-compatible safetensors) | ~2.3 GB |
| Swift port code (`MLXASR` + adapter + streaming session) | <1 MB |
| **Total cost added on first preview use** | **~2.3 GB** |

The Swift-native path replaces the prior Python-sidecar bundle math:
no venv, no `transformers`, no `mlx-audio`, no `tokenizers`, no
`scipy` — just the MLX safetensors and Swift code. `YoozEngine.app`
does not bundle the weights. On the first transcription request that
resolves to `qwen3_asr_preview`, the engine surfaces a download prompt
with size, expected duration, and a progress bar before any
transcription runs. Weights land in
`~/Library/Application Support/Yooz/engine/models/qwen3-asr-1.7b-8bit/`.
Subsequent launches are instant. We optionally ship a 0.6B-4bit
variant (~400 MB) for slow-network users who want to try the preview
without committing to the full download.

## Graduation criteria

The preview becomes the multilingual default when **all** of the
following hold over a 4-week telemetry window:

- **P50 RTFX < 0.5x on M2 Air baseline.** With current numbers
  (~16x RTFX) we are well inside that envelope; this guards against
  a regression slipping in via mlx-audio updates.
- **WER parity holds.** Specifically, English-Qwen3 stays within 1 pp
  of English-Parakeet, and Arabic-Qwen3 stays within 1 pp of its own
  Phase 2 number on real telemetry distributions (not just curated
  yooz-test-data).
- **No regression complaints over 4 weeks.** Operationalized as
  zero P0/P1 issues filed against the preview backend, and a
  `userAccepted == false` rate that does not exceed Parakeet's by
  more than 5 percentage points.
- **Streaming WER measured and acceptable.** `stream_transcribe`
  benchmark must land before any default flip; offline-only data
  does not generalize to push-to-talk.

Even after graduation: **Qwen3-ASR never replaces Parakeet for
English-only users until streaming-mode WER closes the gap.** The
multilingual preview becomes the default for *multilingual* users
(those whose language settings or auto-LID routinely pull a
non-Parakeet language) first. English-only stays on Parakeet
indefinitely under the current numbers.

## Integration path: native MLX-Swift only

Integration is tracked under **epic #46 — native MLX-Swift port of
the Qwen3-Omni audio encoder + log-mel frontend + encoder-decoder
bridge**. The engine ships `qwen3_asr_preview` only when that Swift
port lands. Yooz Engine is graduating from Python; every inference
path in the engine must be MLX-native Swift, and Qwen3-ASR is not
exempted.

What the Swift port has to deliver, distilled from `MLX_SWIFT_COMPAT.md`:

1. **`MLXASR` library (new) inside `mlx-swift-lm` or vendored in
   YoozEngine.** No equivalent exists upstream today.
2. **`Qwen3AudioEncoder` in Swift/MLX.** Conv2d frontend (3 stride-2
   layers, 128 mel bins, 480-dim hidden), sinusoidal positional
   embeddings, 24 chunked-attention encoder layers (1024-dim, 16-head,
   4096 ffn), the Qwen3-Omni-specific block-attention mask, and the
   two output projections producing the 2048-d audio token stream.
3. **`Qwen3ASRModel` Swift wiring.** Interleaves audio tokens
   (`audio_token_id=151676`, `audio_start=151669`, `audio_end=151670`)
   into the Qwen3 text decoder via the existing `Qwen3.swift` path
   in `mlx-swift-lm`, threading through `MLXLMCommon`'s KV cache and
   sampler.
4. **Swift-side log-mel frontend.** 128 bins, 25 ms / 10 ms frames,
   matching the HF `WhisperFeatureExtractor` variant Qwen3 uses.
   The current `STT/Audio/AudioPreprocessor.swift` (80 mel bins,
   NeMo config, hand-written for Parakeet/FastConformer) does not
   match and cannot be reused as-is.
5. **`Qwen3ASRModelAdapter: STTModel`** in `YoozEngine/STT/Models/Qwen3ASR/`,
   following the existing Parakeet adapter pattern.
6. **Streaming session** (`Qwen3ASRStreamingSession`) wrapping
   `Qwen3ASRModel`'s incremental encode + decode loop, parallel to
   `FastConformerStreamingSessionAdapter`.
7. **Forced-aligner sibling** (separate ~0.6B Swift port) only after
   the offline path is shipping.

Estimated effort, per `MLX_SWIFT_COMPAT.md`: 3-6 weeks for offline
+ another 1-2 weeks for streaming, comparable to the original
Parakeet TDT port, contingent on a Python reference harness producing
identical intermediate tensors so we can validate the Swift kernels
against ground truth at every stage. The benchmark harness committed
in `scripts/` is that reference.

There is **no Python sidecar plan**, no hybrid path, no "ship A now
then B later". The preview backend does not exist in the engine
until the Swift port does.

## Backend selection API

Adds one new `STTBackend` value and threads it through the existing
`STTLanguage` model selection. The backend is explicitly named
`qwen3asr_preview` so callers cannot mistake it for a stable default.

### Public API surface (proposed)

```swift
public enum STTBackend: String, Codable, Sendable, CaseIterable {
    case parakeet            // existing; English / Latin-European default
    case fastConformer       // existing; Persian, Arabic, Hebrew
    case appleSTT            // existing; Apple's on-device STT
    case qwen3asr_preview    // NEW: Qwen3-ASR-1.7B 8-bit, opt-in preview
    // Future: case qwen3asr_0_6b for low-power devices
}

public extension STTBackend {
    var supportsLanguage: (STTLanguage) -> Bool {
        switch self {
        case .parakeet: { lang in lang.isLatinEuropean }
        case .fastConformer: { lang in lang.isRTL }
        case .appleSTT: { _ in true }   // delegated to Apple
        case .qwen3asr_preview: { lang in
            // 30 languages from mlx-community/Qwen3-ASR-1.7B-8bit config:
            // English, Chinese, Cantonese, Arabic, German, French,
            // Spanish, Portuguese, Indonesian, Italian, Korean,
            // Russian, Thai, Vietnamese, Japanese, Turkish, Hindi,
            // Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech,
            // Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
            !lang.isHebrew
        }
        }
    }

    /// Auto-pick the best backend for a language. Used when the user
    /// has set engine=auto (default).
    ///
    /// Note: this picks *defaults*, not previews. A user who has
    /// explicitly opted into `qwen3asr_preview` keeps it; this
    /// function is for the unopinionated path.
    public static func recommended(for language: STTLanguage) -> STTBackend {
        if language == .hebrew { return .fastConformer }
        if language == .arabic { return .qwen3asr_preview }  // Arabic graduates
        if language == .persian { return .fastConformer }    // unchanged pending head-to-head
        return .parakeet                                     // English / Latin-European default
    }
}
```

### REST surface (additive, no breaking changes)

Existing endpoints stay; we add a `backend` discriminator to the
existing `/v1/stt/engine` POST body and report Qwen3-ASR availability
via `/v1/stt/engine` GET. The `preview` flag in the capabilities
block lets clients render the right "alpha" affordance.

```
GET /v1/stt/engine
{
  "current": "parakeet",
  "available": ["parakeet", "fast_conformer", "apple_stt", "qwen3asr_preview"],
  "capabilities": {
    "qwen3asr_preview": {
      "preview": true,
      "supports_streaming": true,
      "supports_auto_lid": true,
      "supports_forced_alignment": false,    // separate model, not bundled in MVP
      "languages": [...30 codes...],
      "memory_required_mb": 2400,
      "rtfx_estimate": 16.0,
      "weights_size_mb": 2300,
      "first_use_download_required": true
    }
  }
}

POST /v1/stt/engine
{ "backend": "qwen3asr_preview" }    -> 200 { "current": "qwen3asr_preview" }
```

### Implementation outline (epic #46, Swift-native)

1. **`MLXASR` Swift library**, vendored in `YoozEngine` (or upstreamed
   to `mlx-swift-lm` if the Apple/MLX team accepts a contribution).
   Holds the Qwen3-Omni audio encoder, the log-mel frontend, and the
   audio-token bridge into the existing `Qwen3.swift` text decoder
   from `MLXLLM/Models`.

2. **Swift `Qwen3ASRModelAdapter`** in `YoozEngine/STT/Models/Qwen3ASR/`:

   ```swift
   final class Qwen3ASRModelAdapter: STTModel {
       let language: STTLanguage
       let modelFamily: ModelFamily = .qwen3ASR
       let preprocessConfig: PreprocessConfig

       func transcribe(_ audio: [Float]) -> TranscriptionResult { ... }
       func createStreamingSession() -> STTStreamingSession { ... }
   }
   ```

   - Consumes 16 kHz `[Float]` directly. The Swift log-mel is computed
     in-process via the new `MLXASR` frontend (128 bins, Whisper-style
     framing); the existing `STT/Audio/AudioPreprocessor.swift` is for
     Parakeet/FastConformer and is not on this path.

3. **Streaming session**: `Qwen3ASRStreamingSession` wraps the
   incremental encode + decode loop of `Qwen3ASRModel`, plays the same
   role as `FastConformerStreamingSessionAdapter`. No subprocess, no
   IPC; runs in-process on the engine's dispatch queue.

4. **Build variants.** The Qwen3-ASR backend joins the full
   `YoozEngine` variant only. Whisper-helper and Lite variants do not
   gain it (consistent with the AGENTS.md build-variant matrix). Bundle
   size impact on the `YoozEngine` target is the Swift code itself
   (small) plus the on-demand weights download — no Python venv, no
   subprocess footprint.

5. **Validation strategy.** The Python harness in `scripts/` is the
   ground-truth reference. The Swift port is validated by running the
   same audio through both paths and asserting bitwise / near-bitwise
   equivalence at each stage (mel features, encoder output, decoder
   logits) before any release-train build. The harness is *not* a
   shipping component; it lives in the research tree only.

## Risks and open questions

- **Streaming latency:** offline RTFX 16x means 1 s of audio takes
  ~60 ms to transcribe; for VAD-driven streaming with 0.5 s chunks
  the per-call overhead (~50 ms cold, ~30 ms warm prefill) starts to
  dominate. Phase 1 follow-up should rerun with `model.stream_transcribe`.
  This is also a hard graduation criterion; the preview cannot
  promote to default until streaming WER and latency are measured.
- **Persian quality vs FastConformer-fa:** still need head-to-head.
  If FastConformer-fa wins meaningfully (>5 pp WER) we keep it as
  the per-language default for `fa`. Persian users who want to try
  the preview can opt in like everyone else.
- **Memory pressure on 8 GB Macs:** 1.7B-8bit takes ~2.3 GB resident.
  Combined with Parakeet (~2.4 GB) and Apple STT, full engine RSS
  could exceed 6 GB. Plan: lazy-load Qwen3 on first non-English
  request, evict Parakeet if it hasn't been used for N seconds. This
  matters more under the preview rollout (more users will have both
  models warm than under a full replacement).
- **No `mlx-swift-lm` Qwen3-Omni audio encoder upstream.** This is
  the largest piece of work and the gating item for the entire
  preview backend; epic #46 owns it. We are not blocking on
  upstream — the port is in scope for Yooz Engine, with the option
  to upstream to `mlx-swift-lm` afterwards. Track upstream signals
  on `ml-explore/mlx-swift-lm` `MLXLLM/Models` and `MLXVLM/Models`
  in case Apple/MLX adds an `MLXASR` library, but do not depend on
  it.
