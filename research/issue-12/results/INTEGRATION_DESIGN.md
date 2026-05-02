# Phase 3 — Integration design

## Single-model verdict

> **Can Qwen3-ASR-1.7B replace BOTH Parakeet TDT and FastConformer with
> a single model at acceptable latency?**

**Conditional yes.** Replace Parakeet TDT and FastConformer-ar with
Qwen3-ASR-1.7B; keep FastConformer-fa for the moment as a hot-spare
until a head-to-head Persian comparison says otherwise; keep
FastConformer-he indefinitely because Qwen3-ASR does not support
Hebrew.

Backing numbers (Phase 1 + Phase 2, all on this Mac, 16 kHz mono):

| Language | Model | WER | warm latency on 5 s clip | RTFX |
| --- | --- | ---: | ---: | ---: |
| English | Parakeet TDT 0.6B | 0.069 | 0.07 s | 75x |
| English | Qwen3-ASR-1.7B-8bit | **0.063** | 0.31 s | 16x |
| Arabic | Qwen3-ASR-1.7B-8bit | **0.067** | ~0.5 s on real utterances | ~16x |
| Persian | Qwen3-ASR-1.7B-8bit | 0.283 | ~0.9 s on real utterances | ~16x |
| Hebrew | Qwen3-ASR-1.7B-8bit | 0.828 | ~1.0 s | not viable |

The latency tradeoff (Parakeet 0.07 s -> Qwen3 0.31 s on a 5 s clip)
is real but not a UX-killer for dictation: the user does not see
the inference latency, only the time-to-first-text after they
release push-to-talk. With a 5 s utterance, that's a 240 ms
difference. For real-time streaming chunks (sub-1 s) the gap matters
more, and is the reason **streaming on Qwen3-ASR-1.7B is the open
risk** (Phase 1 used offline `generate()`; the upstream `stream_transcribe`
path needs its own benchmark).

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

**Phase 6 plan: ship Qwen3-ASR-1.7B as a Python-backed STT backend
in YoozEngine, alongside Parakeet (Swift, native).**

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
`STTLanguage` model selection.

### Public API surface (proposed)

```swift
public enum STTBackend: String, Codable, Sendable, CaseIterable {
    case parakeet       // existing
    case fastConformer  // existing (Persian, Arabic, Hebrew)
    case appleSTT       // existing (Apple's on-device STT)
    case qwen3asr_1_7b  // NEW: Qwen3-ASR-1.7B 8-bit
    // Future: case qwen3asr_0_6b for low-power devices
}

public extension STTBackend {
    var supportsLanguage: (STTLanguage) -> Bool {
        switch self {
        case .parakeet: { lang in lang.isLatinEuropean }
        case .fastConformer: { lang in lang.isRTL }
        case .appleSTT: { _ in true }   // delegated to Apple
        case .qwen3asr_1_7b: { lang in
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
    public static func recommended(for language: STTLanguage) -> STTBackend {
        if language == .hebrew { return .fastConformer }
        if language == .english { return .parakeet }     // latency win
        return .qwen3asr_1_7b                            // everywhere else
    }
}
```

### REST surface (additive, no breaking changes)

Existing endpoints stay; we add a `backend` discriminator to the
existing `/v1/stt/engine` POST body and report Qwen3-ASR availability
via `/v1/stt/engine` GET:

```
GET /v1/stt/engine
{
  "current": "parakeet",
  "available": ["parakeet", "fast_conformer", "apple_stt", "qwen3asr_1_7b"],
  "capabilities": {
    "qwen3asr_1_7b": {
      "supports_streaming": true,
      "supports_auto_lid": true,
      "supports_forced_alignment": false,    // separate model, not bundled in MVP
      "languages": [...30 codes...],
      "memory_required_mb": 2400,
      "rtfx_estimate": 16.0
    }
  }
}

POST /v1/stt/engine
{ "backend": "qwen3asr_1_7b" }    -> 200 { "current": "qwen3asr_1_7b" }
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

### Bundle / install-size impact

| Component | Approx size |
| --- | ---: |
| `mlx-community/Qwen3-ASR-1.7B-8bit` weights | 2.3 GB |
| Python venv (mlx-audio + mlx + transformers + tokenizers + numpy + scipy) | ~1.2 GB |
| Worker code | <1 MB |
| **Total cost added to YoozEngine.app** | **~3.5 GB** |

This is significant. To keep `YoozEngine.app` as a downloadable, we
should fetch the Qwen3-ASR model on first run from HuggingFace into
`~/Library/Application Support/Yooz/engine/models/qwen3-asr-1.7b-8bit/`,
not bundle it. This also lets us ship the `0.6B-4bit` variant for
slow networks (~700 MB) and upgrade to 1.7B opportunistically.

## Risks and open questions

- **Streaming latency:** offline RTFX 16x means 1 s of audio takes
  ~60 ms to transcribe; for VAD-driven streaming with 0.5 s chunks
  the per-call overhead (~50 ms cold, ~30 ms warm prefill) starts to
  dominate. Phase 1 follow-up should rerun with `model.stream_transcribe`.
- **Persian quality vs FastConformer-fa:** still need head-to-head.
  If FastConformer-fa wins meaningfully (>5 pp WER) we keep it as
  the per-language default for `fa`.
- **Memory pressure on 8 GB Macs:** 1.7B-8bit takes ~2.3 GB resident.
  Combined with Parakeet (~2.4 GB) and Apple STT, full engine RSS
  could exceed 6 GB. Plan: lazy-load Qwen3 on first non-English
  request, evict Parakeet if it hasn't been used for N seconds.
- **No `mlx-swift-lm` Qwen3-Omni audio encoder.** A native Swift port
  is multi-week work; do not block Phase 6 on this. Track upstream
  via `ml-explore/mlx-swift-lm` `MLXLLM/Models` directory and
  consider revisiting once an `MLXASR` library lands.
