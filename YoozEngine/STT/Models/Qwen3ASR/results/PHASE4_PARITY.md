# Phase 4 — Qwen3-ASR Encoder ↔ Decoder Bridge Parity (issue #55)

**Status:** PASS. End-to-end transcription path now runs in pure
Swift / MLX-Swift, no Python sidecar. Output matches the upstream
`mlx_audio` Python reference exactly on the canonical clip and on
the EN/AR/FA WER subsets.

## What landed

- `Qwen3ASRTextConfig.swift` / `Qwen3ASRFullConfig` — Phase 4 config
  surface that decodes the published `config.json` and validates the
  text decoder geometry against `Qwen3ASRError.invalidConfig`.
- `Qwen3ASRTextDecoder.swift` — Qwen3 text decoder (28 layers, 16/8
  attention/KV heads, head_dim 128, hidden_size 2048,
  intermediate_size 6144) with both a token-input forward and a
  precomputed-embeddings forward. Conforms to
  `MLXLMCommon.LanguageModel` so it can drop into the stock
  `TokenIterator` for diagnostics later.
- `Qwen3ASRPipeline.swift` — orchestrator that owns the mel frontend
  (Phase 2), the audio encoder (Phase 3), the text decoder, and the
  HuggingFace tokenizer (loaded via `swift-transformers`'s
  `AutoTokenizer.from(modelFolder:)`). Public API:
  `transcribe(pcm:language:maxNewTokens:)` returning
  `Qwen3ASRTranscription { text, language, generatedTokens,
  numAudioTokens, totalSeconds }`.

## End-to-end parity (canonical 12.6 s English clip)

| Check | Reference | Swift | Bar | Verdict |
| --- | --- | --- | --- | --- |
| Prompt token IDs (auto-detect) | 179 IDs | 179 IDs | exact | exact |
| `<|audio_pad|>` positions | 164 slots | 164 slots | exact | exact |
| `audio_tower` output token count | 164 | 164 | exact | exact |
| Generated token sequence | greedy stream | identical | exact | exact |
| Decoded text | "Unfortunately, studying traffic flow is difficult because driver behavior cannot be predicted with 100% certainty." | identical | exact | exact |
| Detected language | English | English | exact | exact |

Tests: `Qwen3ASRPipelineParityTests`
(`testPromptTemplateMatchesReference`,
`testDecoderPrefillArgmaxMatchesReference`,
`testEndToEndTranscriptionMatchesReference`).

## WER subset parity (5 utterances each)

| Lang | Python WER | Swift WER | Δ | Bar | Verdict |
| --- | --- | --- | --- | --- | --- |
| EN   | 0.0842    | 0.0842    | 0.0000 | ≤ +0.005 | exact |
| AR   | 0.1200    | 0.1200    | 0.0000 | ≤ +0.005 | exact |
| FA   | 0.4000    | 0.4000    | 0.0000 | ≤ +0.005 | exact |

Reference dump: `dump_wer_references.py` runs the upstream Python
pipeline once with `temperature=0` greedy decode and language hint
set; `wer_subset/{en,fa,ar}.json` records the per-clip ground truth
+ Python hypothesis. Swift WER is recomputed in-test against the
same ground truth using a tokenized Levenshtein. Both sides use the
same normalization — punctuation stripped, lower-cased, whitespace
tokenization — so the WERs are directly comparable.

Tests: `Qwen3ASRPipelineWERParityTests.testWERParity{EN,AR,FA}`.

## Auto language identification

Persian clip (`/Volumes/S1/yooz/stt-test-data/persian/test_001.wav`)
fed with `language: nil`. Pipeline emits the canonical
`language Persian<asr_text>` preamble; `extractLanguageAndContent`
parses it back to the `Persian` label.

Test: `Qwen3ASRPipelineWERParityTests.testAutoLanguageIDPersianClip`.

## Latency micro-benchmark

Run on M-series Apple Silicon, MLX 0.31.3, mlx-swift-lm 2.30.3,
canonical 1.7B-8bit checkpoint. Numbers refresh on every CI run
into `/Volumes/S1/yooz/research/issue-46/phase4-bridge/results/PHASE4_LATENCY.md`.

| Metric | Value |
| --- | --- |
| Cold-start pipeline load | ~1.1 s |
| Resident memory after load | ~2.5 GB |
| Resident memory peak (warm) | ~2.6 GB |
| Warm transcribe — 2 s clip | ~0.15 s |
| Warm transcribe — 5 s clip | ~0.32 s |
| Warm transcribe — 15 s clip | ~0.70 s |

Test: `Qwen3ASRPipelineLatencyTests.testLatencyMicroBenchmark`
(records numbers; never gates).

## Decoder error paths (typed)

| Failure | Swift error |
| --- | --- |
| Empty PCM input | `Qwen3ASRError.invalidInput` |
| Missing safetensors | `Qwen3ASRError.fileNotFound` |
| Text-only checkpoint (no `audio_tower.*`) | `Qwen3ASRError.noAudioTowerWeights` |
| Wrong shape / dtype on `audio_tower.*` | `Qwen3ASRError.shapeMismatch` / `.dtypeMismatch` (Phase 3 path, retained) |
| Tokenizer emits wrong audio_pad count | `Qwen3ASRError.invalidInput` |

Tests: `Qwen3ASRPipelineErrorTests`.

## Build status

- `swift build --target Qwen3ASR` ✅
- `swift test --filter Qwen3ASR*` ✅ — 60 tests pass on M3 Max with
  the `/Volumes/S1` artifacts mounted.
- `xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine
  -configuration Debug build` ✅
- mlx-swift-lm and swift-transformers are added as engine
  dependencies (project.yml + Package.swift); the SwiftPM tree pins
  to `mlx-swift-lm` 2.30.3, the Xcode tree resolves to 2.31.x — the
  decoder and pipeline code compiles cleanly under both.

## What Phase 5 inherits

`Qwen3ASRPipeline.transcribe(pcm:language:maxNewTokens:)` is the
exact API the engine HTTP layer should call from the
`/v1/stt/batch` handler. The pipeline owns model lifetime; instantiate
it once on engine startup and reuse across requests. Greedy decode
is deterministic and reproducible — no sampler state to manage.

`Qwen3ASRTranscription` is `Sendable` and `Equatable`; it can cross
actor boundaries inside the engine without copying caveats.
