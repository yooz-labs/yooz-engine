# SpikeASR — Phase 1 spike (issue #47)

> **Status: reference only.** Production code lives in
> `YoozEngine/STT/Models/Qwen3ASR/`. This directory is the
> historical Phase 1 spike preserved for the parity numbers and
> architecture audit; archival is tracked in the forced-aligner
> follow-up issue (#63).

Native Swift / mlx-swift port of the Qwen3-ASR audio encoder, used to
answer the Phase 1 question:

> Can mlx-swift host the Qwen3-ASR-1.7B audio encoder end-to-end with
> bit-equivalent numerics, without forking mlx-swift or shipping a
> Python sidecar?

**Answer:** yes. See
[`results/DECISION.md`](results/DECISION.md). Per-layer parity numbers
are in [`results/PARITY.md`](results/PARITY.md). The full architecture
audit, including the op-by-op compatibility table and known numerical
foot-guns, is in [`results/ARCH_AUDIT.md`](results/ARCH_AUDIT.md).

## Layout

```
SpikeASR/
├── Package.swift                # SwiftPM entry point (mlx-swift only)
├── Sources/SpikeASR/            # The encoder library
│   ├── AudioEncoder.swift       # Conv2d frontend + 24 encoder layers
│   │                            # + chunked block-attention + projections
│   ├── AudioEncoderConfig.swift # Codable config matching HF config.json
│   └── SafetensorsLoader.swift  # audio_tower.* slice loader
├── Sources/SpikeASRTool/        # CLI: spike-asr {smoke|forward}
├── Tests/SpikeASRTests/         # 8 tests; parity test is the gate
└── results/                     # Phase 1 deliverable docs
```

## Build / test

`mlx-swift` ships its Metal kernels as `.metal` source files; SwiftPM
CLI (`swift build` / `swift test`) does not compile them, so tests
must run through `xcodebuild`.

One-time setup (per dev box / CI runner, Xcode 16+):

```bash
xcodebuild -downloadComponent MetalToolchain
```

Then:

```bash
cd SpikeASR
xcodebuild -scheme SpikeASR-Package -destination 'platform=macOS' test
```

The CLI:

```bash
xcodebuild -scheme spike-asr -destination 'platform=macOS' build
$(xcodebuild -showBuildSettings -scheme spike-asr | awk '/ TARGET_BUILD_DIR /{print $3}')/spike-asr smoke
$(xcodebuild -showBuildSettings -scheme spike-asr | awk '/ TARGET_BUILD_DIR /{print $3}')/spike-asr forward
```

The CLI's `forward` mode reads
`/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/parity_inputs.safetensors`
(produced by `dump_parity.py`) and writes
`swift_encoder_outputs.safetensors` next to it.

## Test coverage

| Test | Gate | Skip rule |
| --- | --- | --- |
| `testFreqAfterConvMatchesReference` | structure | always runs |
| `testFeatExtractOutputLengthMatchesReference` | structure | always runs |
| `testEncoderShapesMatchCheckpoint` | structure | always runs |
| `testLoaderRoundTripsRandomWeights` | loader integrity | always runs |
| `testLoaderRejectsMissingFile` | error path | always runs |
| `testLoaderRejectsUnrelatedSafetensors` | error path | always runs |
| `testEncoderLoadsRealWeights` | smoke | skips if `audio_tower_bf16.safetensors` absent |
| `testEncoderParityWithPython` | **parity gate** | skips if any of the three artifacts absent |

The parity test is the regression gate: any future Phase 2-5 change
that drifts the encoder past `max-abs-delta = 1e-3` against the
Python reference fails the test.

## Phase 1 artifacts

These live on `/Volumes/S1` (not in git) because they're large or
regenerable:

```
/Volumes/S1/yooz/research/issue-46/
├── phase1-spike/
│   ├── artifacts/
│   │   ├── audio_tower_bf16.safetensors     # 606 MB  (Phase 1 input weights)
│   │   ├── parity_inputs.safetensors        # 1.5 MB  (mel features, mask)
│   │   ├── parity_outputs.safetensors       # 520 KB  (Python reference)
│   │   ├── safetensors_keys.json            # full key/shape dump
│   │   ├── swift_encoder_outputs.safetensors  # written by spike-asr forward
│   │   └── parity_swift_metrics.txt         # written by parity test
│   └── results/                             # symlinked into ./results/
└── reference/
    ├── clip.wav                             # 5 s synthetic 16 kHz mono
    ├── dump_parity.py                       # regen the artifacts above
    └── qwen3_asr_mlx_audio.py               # Python reference snapshot
```

To regenerate the artifacts on a new machine:

```bash
/Volumes/S1/yooz/research/issue-12/.venv/bin/python \
  /Volumes/S1/yooz/research/issue-46/reference/dump_parity.py
```

## Status

[x] All 8 tests pass under `xcodebuild test`.
[x] `max-abs-delta = 9.6e-7`, three orders under the 1e-3 bar.
[x] DECISION verdict: continue with mlx-swift unchanged. No fork.
