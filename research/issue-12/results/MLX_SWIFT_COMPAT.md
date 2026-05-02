# Qwen3-ASR — MLX-Swift / mlx-swift-lm compatibility

**Verdict: native MLX-Swift port required, and it is the integration
path.** mlx-swift-lm has no audio module today (no `MLXASR`, no
Qwen3-Omni / Qwen3-ASR model class, no Swift mel-frontend), so a
direct off-the-shelf `mlx-swift` load is not possible. That work — a
new audio-encoder subsystem plus model-class wiring — is the entire
scope of **epic #46**, accepted at an estimated 3–6 weeks for offline
plus 1–2 weeks for streaming. Yooz Engine is graduating from Python;
no Python sidecar will ship from `YoozEngine.app`. The 3–6-week porting
cost is accepted because the engine has no other viable shape, and
because the research below shows the architecture maps cleanly onto
existing MLX-Swift primitives (Conv2d, attention, KV cache).

## Evidence

### 1. mlx-swift-lm has no audio module at all

Listing the libraries directory of `ml-explore/mlx-swift-lm@main`
(via `gh api repos/ml-explore/mlx-swift-lm/contents/Libraries`) yields:

| Library | Purpose |
| --- | --- |
| `MLXLLM` | Text-only LLMs |
| `MLXVLM` | Vision-language models |
| `MLXEmbedders` | Sentence/token embedders |
| `MLXLMCommon` | Shared utilities (KV cache, sampler, tokenizer) |
| `MLXHuggingFace` / `MLXHuggingFaceMacros` | HF model loaders |
| `BenchmarkHelpers`, `IntegrationTestHelpers` | Tooling |

There is **no `MLXASR`, `MLXAudio`, or `MLXSpeech` library**. The Swift
side of the MLX ecosystem currently has zero ASR / audio-encoder code
paths, mel-spectrogram extractors, or audio tokenizers.

### 2. `MLXLLM/Models` does not include Qwen3-Omni / Qwen3-ASR

`gh api repos/ml-explore/mlx-swift-lm/contents/Libraries/MLXLLM/Models`
returns 50 model files. Qwen-family entries are:

```
Qwen2.swift
Qwen3.swift
Qwen35.swift
Qwen35MoE.swift
Qwen3MoE.swift
Qwen3Next.swift
```

`MLXVLM/Models` adds `Qwen2VL.swift`, `Qwen25VL.swift`, `Qwen3VL.swift`,
`QwenVL.swift`. No `Qwen3Omni`, no `Qwen3ASR`, no `Qwen3AudioEncoder`.

### 3. The Python reference implementation uses a custom audio encoder
the Swift port would have to re-create

`Blaizzy/mlx-audio` ships `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`
(50 KB). The audio encoder is **not** a reused Whisper/Conformer block —
it is a Qwen3-Omni-specific stack:

- Three stride-2 `Conv2d` downsampling layers on log-mel input
  (`num_mel_bins=128`, `downsample_hidden_size=480`).
- A `Linear` projection that flattens frequency into channels.
- A custom `SinusoidalPositionEmbedding(max_source_positions=1500)`.
- 24 `AudioEncoderLayer`s (1024-dim, 16-head, 4096 ffn).
- **Chunked / block attention** with custom `_create_block_attention_mask`
  (cu_seqlens-style ragged masks).
- A custom feature-length formula `_get_feat_extract_output_lengths`
  that mirrors a peculiar HF Whisper convention but is not Whisper-compatible.
- Two output projections (`proj1`, `proj2`) that produce a 2048-d
  audio token stream consumed by the Qwen3 text decoder.

Special-token plumbing is also custom:
`audio_token_id=151676`, `audio_start_token_id=151669`,
`audio_end_token_id=151670`. The text decoder runs autoregressively
on a hybrid prompt that interleaves audio tokens with regular Qwen3
tokens — there is no existing Swift abstraction for that interleaving
(current MLXVLM models all gate on image tokens, not streamed audio
embeddings).

### 4. The mel-frontend would need a Swift implementation

The Python path uses `mlx_audio.stt.models.qwen3_asr` plus
`mlx_audio.audio_io.read` (libsndfile-backed) and a
log-mel feature extractor. None of those are available on the Swift
side; the existing `STT/Audio/AudioPreprocessor.swift` in YoozEngine
is hand-written for Parakeet/FastConformer (80 mel bins, NeMo
preprocess config) and does not match Qwen3-ASR's 128-bin chunked layout.

## Implication for the engine roadmap

A `mlx-swift` Qwen3-ASR backend would require, at minimum:

1. **New Swift package or YoozEngine target** mirroring `MLXASR/Models`.
2. **Port `Qwen3AudioEncoder`** (Conv2d frontend, sinusoidal pos emb,
   chunked block-mask attention, projections) into Swift/MLX.
3. **Port `Qwen3ASRModel`** that interleaves the audio encoder output
   with Qwen3 text tokens through the existing `Qwen3.swift` decoder
   (this part is mostly wiring once MLXLMCommon's KV cache and
   tokenizer can accept the audio-token boundary tokens).
4. **Swift-side log-mel** (128 bins, 25 ms / 10 ms frames) consistent
   with the HF `WhisperFeatureExtractor` variant Qwen3 uses.
5. **Forced-aligner sibling model** if we also want word timestamps
   (separate 0.6B model, similar but with extra projection heads).

Estimated effort: comparable to the original Parakeet TDT port that
landed in YoozEngine — roughly 2–4 weeks for a working offline path
plus another 1–2 weeks for streaming, assuming we have a reference
Python harness producing identical intermediate tensors to validate
against.

## Path forward

Yooz Engine is graduating from Python. Every engine inference path
must be MLX-native Swift, and Qwen3-ASR is not exempted. Of the three
options the team considered:

| Option | Effort | Status |
| --- | --- | --- |
| **A. Python sidecar via existing yooz-stt-engine `.venv`** | 1–2 days | **Rejected.** Conflicts with the engine's Python-graduation direction. |
| **B. Wait for upstream mlx-swift-lm Qwen3-Omni support** | unknown | **Rejected.** Apple/MLX team has not announced an audio-encoder roadmap for mlx-swift-lm; we will not block on it. |
| **C. In-house Swift port** | 3–6 weeks offline + 1–2 weeks streaming | **Selected.** Tracked under epic #46. |

The benchmark harness in `scripts/bench_phase1_latency.py` and
`scripts/bench_phase2_quality.py` is the Python reference *for
validation only*, used to verify that the Swift kernels produce
identical or near-identical intermediate tensors at every stage
(mel features, encoder output, decoder logits). The harness is not a
shipping component and never enters `YoozEngine.app`.
