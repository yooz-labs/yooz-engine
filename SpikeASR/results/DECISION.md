# Phase 1 Spike — Decision

**Verdict:** **Continue with mlx-swift unchanged. Do not fork.**
The Qwen3-ASR audio encoder ports cleanly to mlx-swift `0.21.2`+ and
matches the Python reference within fp32 noise on the canonical 5 s
parity clip. No mlx-swift fork, no new MLX kernels, no upstream PR
required to ship Phase 2-5.

## Why

1. **Every audio-encoder op exists in mlx-swift today.**
   `Conv2d`, `Linear`, `LayerNorm` (via `MLXFast.layerNorm`), `gelu`
   (exact erf form), `MLXFast.scaledDotProductAttention`, `padded`,
   `stacked`, `concatenated`, `loadArrays` (safetensors). See the
   table in `ARCH_AUDIT.md` -> "Op-by-op mapping". The audio_tower
   slice of the 8-bit checkpoint loads directly as bf16 with **zero**
   quantization plumbing required (only `model.*` is 8-bit).

2. **Numerical parity is essentially fp32 noise.**
   `max-abs-delta = 9.6e-7`, `mean-abs-delta = 3.5e-8` over 65 audio
   tokens by 2048 dimensions, against the Python `mlx-audio` reference
   on the same Whisper-mel input. Tolerance was 1e-3; we cleared it
   by three orders of magnitude. See `PARITY.md`.

3. **The end-to-end Swift loader runs.** `SpikeLoader.loadAudioTower`
   ingests the 606 MB `audio_tower_bf16.safetensors` slice, applies
   it to a fresh `AudioEncoder`, and runs both a synthetic-zero smoke
   pass and the real parity input without exception.

4. **No upstream blockers were discovered.** Subtle drift sources
   (Python floor-div vs Swift truncation, additive mask infinity
   policy, parameter-vs-constant separation for the positional cache,
   pre-scaled query) were all resolvable inside the spike with no
   external dependency.

## What this means for the rest of epic #46

| Phase | Scope | Estimate (M2 Pro dev box) |
| --- | --- | --- |
| 2 | Swift log-mel frontend (Whisper 128-bin, 25 ms / 10 ms) | 3-5 days |
| 3 | Encoder integration into YoozEngine; expose as a module | 2-3 days |
| 4 | Audio-token bridge into the Qwen3 text decoder (`MLXLLM/Models/Qwen3.swift`) using the existing 8-bit quantized weights and `MLXLMCommon`'s KV cache + sampler | 5-8 days |
| 5 | Streaming session (chunked block-mask, per-chunk encode + decoder prefill) | 5-7 days |
| 6 | `Qwen3ASRModelAdapter: STTModel` + REST/HTTP plumbing in `YoozEngine/STT/Models/Qwen3ASR/` | 2-3 days |
| 7 | Forced-aligner sibling (separate 0.6B model) | 4-6 days, only after offline path ships |

Total Phase 2-6 envelope: **~3-4 weeks** of focused work, one
engineer, no external blockers. Matches the 3-6 week offline budget
in the original `MLX_SWIFT_COMPAT.md`.

## What this does NOT validate (out of scope for the spike)

- **Mel-spectrogram parity.** The Phase 1 parity test uses Python's
  `WhisperFeatureExtractor` to produce mel features and feeds the
  identical tensor to both Python and Swift encoders. Phase 2 owns
  the Swift log-mel implementation and its own parity check.
- **Text decoder integration.** The 8-bit Qwen3 decoder is not loaded
  or executed in the spike. The audio_tower output shape `(65, 2048)`
  matches what the Qwen3 decoder expects, but actual interleaving
  with `audio_token_id=151676 / start=151669 / end=151670` is Phase 4.
- **Streaming.** The block-attention mask runs as a single chunk for
  the 30 s reference window; the per-utterance `cu_seqlens` plumbing
  is wired but not exercised end-to-end with a real stream session.
  Phase 5 owns it.
- **Real audio quality.** The reference clip is synthetic; full WER /
  RTFX numbers from Phase 1 / 2 of `engine-issue-12-asr` already
  exist and are not re-litigated here.
- **Performance.** The spike uses Debug Swift builds and DerivedData
  on `/Volumes/S1`. End-to-end RTFX numbers belong in Phase 5.

## New dependencies

None beyond the existing `mlx-swift from: 0.21.2` pin already in
`yooz-engine/project.yml`. xcodebuild (not `swift test`) is required
to compile the bundled Metal shaders into a `default.metallib`; this
is true of every mlx-swift consumer and does not change with the
audio encoder port. `xcodebuild -downloadComponent MetalToolchain`
must be run once on every dev box / CI runner that uses Xcode 16+;
the Metal Toolchain dependency is documented at the top of the
SpikeASR test file and reiterated in the build instructions in this
PR.

## What ships out of Phase 1

- `SpikeASR/` SwiftPM package: the audio encoder library plus an
  end-to-end CLI, kept under the worktree as a side-package so Phase 2
  can promote it into `YoozEngine/STT/Models/Qwen3ASR/MLXASR/` once
  the API stabilizes. Not added to `project.yml` as a YoozEngine
  dependency; it is research code, not ship code.
- `phase1-spike/results/{ARCH_AUDIT, PARITY, DECISION}.md` documenting
  the audit, parity numbers, and rollout plan.
- `phase1-spike/artifacts/` weights slice (606 MB) + parity inputs /
  outputs / metrics, regeneratable from the Python harness.
- 8 Swift unit + integration tests, including a parity test that
  fails the build if any Phase 2-5 change drifts the encoder beyond
  the 1e-3 tolerance.

## Recommendation

Promote the Phase 1 spike to Phase 2 by:

1. Renaming `SpikeASR` -> `MLXASR` and moving it under
   `YoozEngine/STT/Models/Qwen3ASR/MLXASR/` once the directory
   convention is settled.
2. Adding the Whisper 128-bin log-mel frontend (Phase 2 work).
3. Carrying the parity test forward as a regression gate (it should
   keep passing through every subsequent phase, including post-mel-port
   end-to-end runs).
4. Closing issue #47 once this PR lands.

There is no blocker, no fork, and no abandonment scenario on the
table. The spike's job — answering "can mlx-swift do this?" — is
done; the answer is yes, with margin.
