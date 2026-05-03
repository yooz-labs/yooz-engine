# Yooz Engine

**Sovereign Intelligence. Built for the skeptical.**

Yooz Engine is a unified local AI service for macOS. It runs as a menu bar app on `localhost:19920` and provides on-device speech-to-text, LLM inference, grammar correction, voice activity detection, and (soon) text-to-speech to every Yooz app on your system. Models stay on your device; nothing is sent to the cloud.

This is the source repository. Built with Swift 5.9 (with Swift 6 concurrency idioms), SwiftUI, [Hummingbird](https://github.com/hummingbird-project/hummingbird) for HTTP/WebSocket, and [MLX-Swift](https://github.com/ml-explore/mlx-swift) for on-device inference on Apple Silicon.

## What's inside

```
YoozEngine.app — menu bar service exposing localhost:19920
├── /v1/health, /v1/models                  — service introspection
├── /v1/stt/{languages,status,load,batch}   — speech to text (REST)
├── /v1/stt/stream                          — speech to text (WebSocket)
├── /v1/llm/generate                        — LLM text generation
├── /v1/touchup                             — STT cleanup pipeline
├── /v1/grammar/check                       — rule-based grammar
└── /v1/vad/detect                          — voice activity detection
```

Build variants pick which modules ship (`YoozEngine` full, `YoozEngineWhisper` no VAD, `YoozEngineLite` no MLX STT and no VAD). All Yooz apps consume the engine via the `YoozEngineClient` Swift Package — auto-discovery, auto-launch, REST + WebSocket clients in one.

## Quick start

```bash
# Build
xcodegen generate
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation \
  -derivedDataPath build build

# Run
open build/Build/Products/Debug/Yooz\ Engine.app
curl http://localhost:19920/v1/health
```

`-skipMacroValidation` bypasses the per-machine trust prompt for `MLXHuggingFaceMacros` (required for headless / first-run CLI builds). `-derivedDataPath build` keeps the artifact next to the source tree instead of `~/Library/Developer/Xcode/DerivedData`.

The full architecture, build variants, conventions, and module specs live in [`AGENTS.md`](AGENTS.md). The strategic licensing position lives in [`LICENSING.md`](LICENSING.md).

## Models

Yooz Engine expects model weights from HuggingFace. The Yooz-Labs published checkpoints live at [huggingface.co/YoozLabs](https://huggingface.co/YoozLabs):

- [`YoozLabs/Qwen3-ASR-1.7B-Swift`](https://huggingface.co/YoozLabs/Qwen3-ASR-1.7B-Swift) — multilingual ASR with Swift-friendly tokenizer
- [`YoozLabs/Yooz-Quality-v2-Qwen3.5-0.8B-LoRA`](https://huggingface.co/YoozLabs/Yooz-Quality-v2-Qwen3.5-0.8B-LoRA) — STT cleanup LLM (proofread + rewrite)
- More coming as we ship them.

All checkpoints are Apache 2.0. The engine code itself is source-available under PolyForm Shield 1.0.0 (see below).

## License

The engine source code is licensed under [**PolyForm Shield 1.0.0**](LICENSE.md). You can:

- Read, fork, modify, and use it for any purpose **except** building a competing product.
- Embed it in apps that aren't direct Yooz Engine substitutes.
- Contribute back via PRs.

You cannot offer a "managed Yooz Engine" or a re-skinned commercial fork. For the strategic rationale and the full product/weights matrix, see [`LICENSING.md`](LICENSING.md).

**Compiled `.app` binaries** distributed via GitHub Releases are also under PolyForm Shield. Model weights on HuggingFace are Apache 2.0 (a separate, more permissive license — open weights are a deliberate choice).

For commercial-use or dual-license inquiries: **dev@yooz.info**.

## Contributing

PRs welcome. Sign your commits with `Signed-off-by: Your Name <you@example.com>` (DCO style); see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

## Status

Active development. Version 0.6.0 is the modular thin-client architecture (Phase 5). The engine is the heart of the [Yooz ecosystem](https://github.com/yooz-labs).

---

*Part of the Yooz ecosystem. Sovereign Intelligence.*
