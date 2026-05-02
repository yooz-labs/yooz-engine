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
3. **Arabic flips to Qwen3-ASR-1.7B as its default.** This is the
   one language where Qwen3-ASR clearly wins (6.7% WER vs FastConformer-ar
   on equivalent material) at latency comparable to what Arabic users
   were already paying with FastConformer. No regression, real upgrade.
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
| `mlx-community/Qwen3-ASR-1.7B-8bit` weights | 2.3 GB |
| Python venv (mlx-audio + mlx + transformers + tokenizers + numpy + scipy) | ~1.2 GB |
| Worker code | <1 MB |
| **Total cost added on first preview use** | **~3.5 GB** |

`YoozEngine.app` does not bundle these. On the first transcription
request that resolves to `qwen3_asr_preview` (or to Arabic, once
that flip is shipped), the engine surfaces a download prompt with
size, expected duration, and a progress bar before any transcription
runs. Weights land in
`~/Library/Application Support/Yooz/engine/models/qwen3-asr-1.7b-8bit/`.
Subsequent launches are instant. We optionally ship a 0.6B-4bit
variant (~700 MB) for slow-network users who want to try the preview
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

## Decision tree: how to integrate

```
                Need Qwen3-ASR in YoozEngine?
                            |
         +------------------+------------------+
         |                                     |
   Want native Swift?              OK with Python sidecar?
         |                                     |
   mlx-swift Qwen3-ASR              Reuse stt-engine .venv path
       port required                  + add a worker subprocess
         |                                     |
   3-6 weeks of work                  1-2 days of work
   (see MLX_SWIFT_COMPAT.md)         (just like the Parakeet
                                      mlx-audio path today)
```

### Recommendation

**Phase 6 plan: ship Qwen3-ASR-1.7B as an opt-in Python-backed STT
backend in YoozEngine, alongside Parakeet (Swift, native).** Default
remains Parakeet; Arabic flips its language default to the preview
backend; Persian and Hebrew untouched.

Rationale:

1. The Phase 5 thin-client architecture already isolates the
   YoozEngine service; adding a Python subprocess for one backend
   does not affect bundle size for downstream apps.
2. mlx-audio already ships qwen3_asr; we don't carry custom Python.
3. The community-maintained `mlx-community/Qwen3-ASR-1.7B-8bit` is
   the canonical drop-in (don't pin `aufklarer/...` — see Phase 1
   notes about the missing preprocessor_config.json).
4. A native Swift port stays on the roadmap as Phase 7 once
   `mlx-swift-lm` ships an audio-encoder library.

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

### Implementation outline

1. **Subprocess worker** patterned after `yooz-benchmark/benchmarks/stt_worker.py`:

   ```
   YoozEngine.app/Contents/Resources/qwen3_asr_worker/
       .venv/                          (uv-managed at first launch or shipped)
       worker.py                       (loads mlx-audio Qwen3-ASR once;
                                        speaks the same DONE/ERROR\t protocol
                                        as the existing Parakeet worker)
   ```

   - First launch: trigger `setup.sh` that creates `.venv` and `pip install`s.
   - Persistent worker bound to a Unix domain socket (`/tmp/yooz-engine/qwen3.sock`).
   - Lifetime: tied to `YoozEngine.app`. Worker exits with engine.

2. **Swift adapter** in `YoozEngine/STT/Models/Qwen3ASR/`:

   ```swift
   final class Qwen3ASRModelAdapter: STTModel {
       let language: STTLanguage
       let modelFamily: ModelFamily = .qwen3ASR
       let preprocessConfig: PreprocessConfig

       func transcribe(_ audio: [Float]) -> TranscriptionResult { ... }
       func createStreamingSession() -> STTStreamingSession { ... }
   }
   ```

   - Sends 16 kHz Float arrays as raw bytes over the socket; receives
     transcription text + (optional) timestamps.
   - Reuses the Audio/AudioPreprocessor only for resampling; the Python
     side does the log-mel itself.

3. **Streaming session**: the worker exposes the existing
   `model.stream_transcribe(...)` generator, framed over the socket
   with newline-delimited JSON. A `Qwen3ASRStreamingSession` Swift
   class plays the same role as `FastConformerStreamingSessionAdapter`.

4. **No mlx-swift-lm dependency added.** Build variant matrix from
   the AGENTS.md table is unchanged for the Lite/Whisper variants;
   only the full `YoozEngine` variant gains the Qwen3 worker.

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
- **No `mlx-swift-lm` Qwen3-Omni audio encoder.** A native Swift port
  is multi-week work; do not block Phase 6 on this. Track upstream
  via `ml-explore/mlx-swift-lm` `MLXLLM/Models` directory and
  consider revisiting once an `MLXASR` library lands.
