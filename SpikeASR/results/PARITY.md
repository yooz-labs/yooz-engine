# Phase 1 Spike — Parity Results

**Status:** [x] PASS. Swift port of the Qwen3-ASR audio encoder
matches the Python `mlx-audio` reference within fp32 noise.

## Setup

- **Reference clip:** 5.0 s of synthetic mono audio at 16 kHz
  (deterministic 440 Hz + 220 Hz tones plus seeded noise) at
  `/Volumes/S1/yooz/research/issue-46/reference/clip.wav`. Generated
  by the harness in `dump_parity.py` so anyone can regenerate.
- **Mel features:** computed by HuggingFace
  `WhisperFeatureExtractor.from_pretrained("Qwen3-ASR-1.7B-8bit")`.
  Padded to the canonical 30 s / 3000-frame Whisper window.
- **Reference outputs:** Python encoder forward dumped to
  `parity_outputs.safetensors`. Swift port reads identical input
  features and audio_tower weights from the same directory and
  produces `swift_encoder_outputs.safetensors` plus
  `parity_swift_metrics.txt`.
- **Tolerance bar:** `max-abs-delta <= 1e-3` (set in `ARCH_AUDIT.md`
  to reflect bf16 quantization noise across a 24-layer transformer).

## Results

| Metric | Value | Bar | Verdict |
| --- | --- | --- | --- |
| Output shape | `[65, 2048]` | match Python `[65, 2048]` | match |
| Output dtype | `float32` | float32 | match |
| `max-abs-delta` | **9.611e-07** | <= 1e-3 | three orders under |
| `mean-abs-delta` | **3.532e-08** | < 5e-5 | four orders under |
| NaN count in Swift output | 0 | 0 | match |

The `max-abs-delta` of ~1e-6 is **fp32 round-off noise**, not bf16
noise. The Swift forward path runs the whole encoder in bf16 (matching
weights) and only casts to f32 for the parity comparison; the agreement
to ~1e-6 across 24 transformer layers, 16 attention heads, three
Conv2d frontends, and the chunked block-attention mask demonstrates
the port is bit-equivalent to mlx-audio modulo numerical reordering
in MLX-fast kernels.

## How to reproduce

```bash
# 1. Dump Python reference (one-shot; ~6 s on M1 Pro)
/Volumes/S1/yooz/research/issue-12/.venv/bin/python \
  /Volumes/S1/yooz/research/issue-46/reference/dump_parity.py

# 2. Run the Swift parity test
cd /Users/yahya/Documents/git/yooz/engine-asr-phase1-spike/SpikeASR
xcodebuild -scheme SpikeASR-Package -destination 'platform=macOS' test \
  | grep "testEncoderParityWithPython"

# 3. Read the metrics file the test wrote
cat /Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/parity_swift_metrics.txt
```

## Subtle issues caught during the spike

These are documented in the test suite and the loader so they don't
recur in Phase 2 / 3:

1. **Python `//` is floor-division; Swift `/` truncates toward zero.**
   The output-length helper `_get_feat_extract_output_lengths` does
   `(L % 100 - 1) // 2` which can go negative when `L % 100 == 0`.
   Direct `Int / Int` translation produces an off-by-one for every
   multiple-of-100 input. Fixed via an explicit `floorDiv` helper;
   covered by `testFeatExtractOutputLengthMatchesReference`.
2. **`SinusoidalPositionEmbedding.table` must not be a parameter.**
   When subclassed from `MLXNN.Module`, the cached table gets
   discovered by `parameters()` and `update(parameters:)` then
   demands a `positional_embedding.table` key during weight loading,
   which doesn't exist in the checkpoint. Fixed by making
   `SinusoidalPositionEmbedding` a plain `final class` (the encoder
   keeps a regular Swift reference; the table is a derived constant).
3. **`-Float.infinity` in the block-attention mask NaNs softmax in bf16.**
   Reference uses `-1e9`; the Swift port matches.
4. **Pre-scaled query.** `MLXFast.scaledDotProductAttention` is called
   with `scale: 1.0`; the query is pre-multiplied by
   `head_dim ** -0.5` before the call. Substituting `scale = head_dim ** -0.5`
   into the kernel produces a tiny bf16 drift. Verified the current
   layout matches Python by reading the parity numbers above.
5. **Conv2d weight layout.** MLX core uses `(C_out, kH, kW, C_in)`
   for both Python and Swift, so `audio_tower.conv2d1.weight` of shape
   `[480, 3, 3, 1]` loads directly with no transpose.

## File pointers

- Audit: `/Volumes/S1/yooz/research/issue-46/phase1-spike/results/ARCH_AUDIT.md`
- Decision: `/Volumes/S1/yooz/research/issue-46/phase1-spike/results/DECISION.md`
- Parity reference dump (Python): `/Volumes/S1/yooz/research/issue-46/reference/dump_parity.py`
- Swift package: `/Users/yahya/Documents/git/yooz/engine-asr-phase1-spike/SpikeASR/`
- Tensor key dump: `/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/safetensors_keys.json`
- Swift outputs: `/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/swift_encoder_outputs.safetensors`
- Per-run metrics: `/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/parity_swift_metrics.txt`
