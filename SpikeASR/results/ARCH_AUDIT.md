# Phase 1 Spike — Qwen3-ASR Audio Encoder Architecture Audit

**Scope:** the Qwen3-ASR-1.7B-8bit `audio_tower` (the new audio
encoder). The text decoder is a stock Qwen3 implementation already
covered by `mlx-swift-lm@MLXLLM/Models/Qwen3.swift` and is **out of
scope for the Phase 1 spike**; it is in scope for Phase 4 wiring but
not for the port-feasibility decision.

**Verdict (preview, full case in DECISION.md):** **Continue with
mlx-swift unchanged.** Every audio-encoder op is already present in
mlx-swift `0.21.2`. No fork required.

## Source of truth

- Python reference:
  `/Volumes/S1/yooz/research/issue-46/reference/qwen3_asr_mlx_audio.py`
  (snapshot of `mlx_audio.stt.models.qwen3_asr.qwen3_asr` at
  `mlx-audio==0.4.3`).
- Checkpoint:
  `/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/models--mlx-community--Qwen3-ASR-1.7B-8bit/`,
  snapshot `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`.
- Tensor key dump:
  `/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/safetensors_keys.json`
  (1101 keys total: 397 audio_tower bf16 + 704 model.* 8-bit).
- mlx-swift version pin: `0.21.2` (per
  `/Users/yahya/Documents/git/yooz/yooz-engine/project.yml`).

## Audio encoder configuration (from `config.json` thinker.audio_config)

```
num_mel_bins              128
d_model                   1024
encoder_attention_heads   16
encoder_ffn_dim           4096
encoder_layers            24
downsample_hidden_size    480
output_dim                2048
max_source_positions      1500
n_window                  50
n_window_infer            800
conv_chunksize            500
scale_embedding           false
activation_function       gelu
```

Mel-spectrogram extractor (from `preprocessor_config.json`):

```
feature_extractor_type    WhisperFeatureExtractor
feature_size (mel bins)   128
n_fft                     400
hop_length                160
chunk_length              30 s   -> 480 000 samples at 16 kHz
nb_max_frames             3000
padding_side              right
```

## Tensor shapes — audio_tower (all bf16, unquantized)

The 8-bit checkpoint **does not quantize the audio encoder**. Only
`model.*` (the Qwen3 text decoder) is 8-bit affine-quantized with
`group_size=64`. The audio encoder loads as plain bf16 weights,
which simplifies Phase 1 considerably: no `MLXNN.QuantizedLinear`
plumbing for the spike.

### Convolutional frontend

| Tensor | Shape | Note |
| --- | --- | --- |
| `conv2d1.weight` | [480, 3, 3, 1]   | `Conv2d(in=1, out=480, k=3, s=2, p=1)` MLX layout `(C_out, kH, kW, C_in)` |
| `conv2d1.bias`   | [480]            | |
| `conv2d2.weight` | [480, 3, 3, 480] | `Conv2d(480, 480, k=3, s=2, p=1)` |
| `conv2d2.bias`   | [480]            | |
| `conv2d3.weight` | [480, 3, 3, 480] | `Conv2d(480, 480, k=3, s=2, p=1)` |
| `conv2d3.bias`   | [480]            | |
| `conv_out.weight` | [1024, 7680]    | `Linear(in=480 * 16 = 7680, out=1024, bias=False)` |

`7680 = downsample_hidden_size (480) * freq_after_conv (16)`, where
`freq_after_conv = ((((128 + 1) // 2) + 1) // 2 + 1) // 2 = 16`.

### Per-layer weights (24 layers, each identical layout)

| Tensor (per layer N) | Shape | Note |
| --- | --- | --- |
| `layers.N.self_attn_layer_norm.{weight,bias}` | [1024] | Pre-attn LayerNorm |
| `layers.N.self_attn.q_proj.{weight,bias}` | [1024, 1024] / [1024] | Linear |
| `layers.N.self_attn.k_proj.{weight,bias}` | [1024, 1024] / [1024] | Linear |
| `layers.N.self_attn.v_proj.{weight,bias}` | [1024, 1024] / [1024] | Linear |
| `layers.N.self_attn.out_proj.{weight,bias}` | [1024, 1024] / [1024] | Linear |
| `layers.N.final_layer_norm.{weight,bias}` | [1024] | Pre-FFN LayerNorm |
| `layers.N.fc1.{weight,bias}` | [4096, 1024] / [4096] | Linear, GELU after |
| `layers.N.fc2.{weight,bias}` | [1024, 4096] / [1024] | Linear |

### Post-encoder projections

| Tensor | Shape | Note |
| --- | --- | --- |
| `ln_post.{weight,bias}` | [1024] | LayerNorm |
| `proj1.{weight,bias}` | [1024, 1024] / [1024] | Linear, GELU after |
| `proj2.{weight,bias}` | [2048, 1024] / [2048] | Linear -> 2048-d audio token stream consumed by the Qwen3 text decoder |

### Sinusoidal positional embedding

Computed at construction time from `(max_source_positions=1500, channels=1024,
max_timescale=10000.0)`. Not stored as a tensor in the checkpoint.

## Op-by-op mapping to mlx-swift `0.21.2`

| Python (mlx-audio) op | mlx-swift availability | Swift symbol |
| --- | --- | --- |
| `nn.Conv2d` (3 stride-2 frontend convs) | Present | `MLXNN.Conv2d(inputChannels:outputChannels:kernelSize:stride:padding:bias:)` |
| `nn.Linear` | Present | `MLXNN.Linear` |
| `nn.LayerNorm` | Present | `MLXNN.LayerNorm` -> `MLXFast.layerNorm` |
| `nn.gelu` | Present | `MLXNN.gelu(_:)` (full erf-based GELU; matches `mlx_audio` default) |
| `mx.fast.scaled_dot_product_attention` | Present | `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)` |
| Sinusoidal positional embedding (manual) | Composable | `MLX.exp / arange / sin / cos / concatenated` |
| Block-attention mask (cu_seqlens style) | Composable | `MLXArray.full + .indexed assignment` (or precomputed numpy then `MLXArray(_)`) |
| `mx.pad(chunk, [(0,0),(0,pad)])` | Present | `MLX.padded(_:widths:value:)` (Ops.swift line 1973) |
| `mx.stack` | Present | `MLX.stacked(_:axis:)` |
| `mx.concatenate` | Present | `MLX.concatenated(_:axis:)` |
| `mx.load` (safetensors) | Present | `MLX.loadArrays(url:)` (Ops.swift line 1588) |
| Tensor `.transpose(0, 2, 3, 1).reshape(...)` | Present | `.transposed(...)` / `.reshaped(...)` |

**No missing ops.** No fork blockers. No experimental features
required. mlx-swift `0.21.2` is sufficient for the entire encoder
forward pass.

## Numerical conventions to verify in parity check

These are the points where Python and Swift implementations are
known to drift in subtle ways. The parity script exercises each:

1. **GELU variant.** `mlx_audio` calls `nn.gelu(...)`. mlx-core's
   `nn.gelu` is the exact erf-based form
   `0.5 * x * (1 + erf(x / sqrt(2)))`. mlx-swift's `MLXNN.gelu`
   matches; do **not** substitute `geluApproximate` /
   `geluFastApproximate`.
2. **`scaled_dot_product_attention` scale.** Python passes
   `scale=1.0` because the query is pre-scaled by
   `self.scaling = head_dim ** -0.5` before the call. Swift port must
   replicate the pre-scale, not pass `scale = head_dim ** -0.5`,
   to keep numerical drift below `1e-6` on bf16.
3. **LayerNorm eps.** `nn.LayerNorm(embed_dim)` Python default is
   `eps=1e-5`. mlx-swift `MLXNN.LayerNorm` default is also `1e-5`.
   No override needed; assert in tests.
4. **bf16 accumulation.** All audio_tower weights ship in bf16.
   Forward pass runs in bf16 unless we explicitly cast to f32 inside
   the Swift port. Parity tolerance must reflect bf16 noise:
   `max-abs-delta <= 1e-3` is the right pass bar (see `DECISION.md`),
   not `1e-6`.
5. **Conv2d weight layout.** MLX core uses `(C_out, kH, kW, C_in)`
   for both Python and Swift. Verified directly in Python:
   `nn.Conv2d(1, 480, 3, 2, 1).weight.shape == (480, 3, 3, 1)`,
   exactly matching `conv2d1.weight` in the checkpoint. No transpose
   needed at load time.
6. **Sinusoidal positional embedding init dtype.** Python builds
   `inv_timescales` and `scaled_time` as `mx.float32` then concatenates.
   The `_positional_embedding` cache is **not** explicitly cast back
   to bf16, so Python keeps it in f32 across the forward call (but
   adds it to a bf16 `x`, which is implicitly upcast by mlx). Swift
   must match (compute in f32, let MLX promote at the add).
7. **Block attention mask dtype.** Python builds the mask in the same
   dtype as `hidden_states` (bf16) and uses `-1e9` as the additive
   penalty. Swift must do the same; using `-Float.infinity` will
   cause NaNs after softmax in bf16.

## Out of scope for the spike

- Mel-spectrogram (Whisper-style 128-bin) frontend — Phase 2 owns it.
  The parity dump uses Python's mel features as input and feeds them
  directly to the Swift encoder, so Phase 1 measures encoder parity
  in isolation.
- Text decoder (Qwen3 28-layer, 8-bit quantized) — Phase 4 owns it.
- Streaming session and chunked block mask logic — Phase 5 owns it.
  The spike runs the offline path with `feature_attention_mask=None`,
  which means a single chunk with full attention; we still verify the
  block-mask plumbing is wired correctly by feeding a 30 s clip and
  letting `_create_block_attention_mask` run.

## File locations

- Audit: `/Volumes/S1/yooz/research/issue-46/phase1-spike/results/ARCH_AUDIT.md`
- Tensor key dump: `/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/safetensors_keys.json`
- Python reference (read-only): `/Volumes/S1/yooz/research/issue-46/reference/qwen3_asr_mlx_audio.py`
