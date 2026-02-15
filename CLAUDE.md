# Yooz Engine - Unified Local AI Service

## Project Overview

**Product:** Standalone macOS service providing local AI capabilities to all Yooz apps
**Version:** 0.4.0 | **Status:** Phase 4 - Grammar + VAD
**Tech Stack:** Swift 5.9+ | SwiftUI | Hummingbird (HTTP/WebSocket) | MLX-Swift

## Architecture

Yooz Engine is a macOS menu bar app that runs a local API server. All Yooz apps (Whisper, Notes, Voice, Crisp, Remi) are thin clients that send API calls to the engine.

```
YoozEngine.app (menu bar service)
├── Local API Server (localhost:19920)
│   ├── REST: /v1/health, /v1/models, /v1/touchup, /v1/llm/generate, /v1/grammar/check, /v1/vad/detect
│   ├── REST: /v1/stt/languages, /v1/stt/status, /v1/stt/load, /v1/stt/batch
│   ├── WebSocket: /v1/stt/stream
│   └── Future: /v1/tts/synthesize
├── STT Module (Parakeet TDT, FastConformer)
├── LLM Module (MLX: Qwen 0.5B, 1.7B)
├── TouchUp Module (full pipeline)
├── Grammar Module (Rust text-cleanup)
├── VAD Module (Silero CoreML)
└── TTS Module [future]

YoozEngineClient (Swift Package)
├── Auto-discovery + auto-launch of engine
├── REST + WebSocket clients
└── Shared types
```

**No embedded fallback.** Apps auto-launch the engine if not running.

## Repository Structure

```
yooz-engine/
├── YoozEngine/          # macOS app (menu bar service)
│   ├── App/             # App entry, lifecycle
│   ├── Server/          # Hummingbird HTTP/WS server
│   ├── STT/             # Speech-to-text (from yooz-stt-engine)
│   ├── LLM/             # LLM inference (MLX)
│   ├── TouchUp/         # Text cleanup pipeline
│   ├── VAD/             # Voice activity detection
│   ├── Grammar/         # Rule-based correction (Rust bridge)
│   ├── TTS/             # Text-to-speech [future]
│   └── Core/            # Config, model management
├── Sources/YoozEngineClient/  # Swift Package (thin client SDK)
├── Tests/
├── Vendor/YoozTextCleanup/    # Rust xcframework
├── text-cleanup/              # Rust source
└── project.yml                # XcodeGen
```

## Build Instructions

```bash
# Generate Xcode project
xcodegen generate

# Build
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine -configuration Debug build

# Run
open build/Debug/Yooz\ Engine.app

# Test health
curl http://localhost:19920/v1/health
```

## API Port

Fixed port: **19920** (localhost only)

## Dependencies

| Package | Purpose |
|---------|---------|
| Hummingbird | HTTP server |
| HummingbirdWebSocket | WebSocket support |
| mlx-swift | MLX runtime for Apple Silicon |
| mlx-swift-lm | LLM inference |
| YoozTextCleanup.xcframework | Rust grammar rules |

## Migration Status

| Module | Source | Status |
|--------|--------|--------|
| Scaffold | New | [x] Done |
| STT | yooz-stt-engine | [x] Done |
| LLM | yooz-stt-engine/TouchUp/LLM | [x] Done |
| TouchUp | yooz-whisper/TouchUp | [x] Done |
| Grammar | yooz-stt-engine/text-cleanup | [x] Done |
| VAD | yooz-whisper/Audio | [x] Done |
| TTS | Future | Not started |

---

*Part of the Yooz ecosystem; Sovereign Intelligence*
