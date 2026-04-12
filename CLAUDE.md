# Yooz Engine - Unified Local AI Service

## Project Overview

**Product:** Standalone macOS service providing local AI capabilities to all Yooz apps
**Version:** 0.5.0 | **Status:** Phase 5 - Thin Client Migration (ready)
**Tech Stack:** Swift 5.9+ | SwiftUI | Hummingbird (HTTP/WebSocket) | MLX-Swift

All AI modules are complete and synced. The engine is now the source of truth for STT, LLM, TouchUp, Grammar, and VAD. The Rust `text-cleanup` source lives in this repo.

## Architecture

Yooz Engine is a macOS menu bar app that runs a local API server. All Yooz apps (Whisper, Notes, Voice, Crisp, Remi) are thin clients that send API calls to the engine.

```
YoozEngine.app (menu bar service)
├── Local API Server (localhost:19920)
│   ├── REST: /v1/health, /v1/models
│   ├── REST: /v1/stt/languages, /v1/stt/status, /v1/stt/load, /v1/stt/batch
│   ├── REST: /v1/llm/generate
│   ├── REST: /v1/touchup
│   ├── REST: /v1/grammar/check
│   ├── REST: /v1/vad/detect
│   ├── WebSocket: /v1/stt/stream
│   └── Future: /v1/tts/synthesize
├── STT Module (Parakeet TDT, FastConformer)
├── LLM Module (MLX: Qwen 0.5B, 1.7B; Apple Intelligence on macOS 26+)
├── TouchUp Module (full pipeline: regex + grammar + LLM)
├── Grammar Module (Rust text-cleanup xcframework + source)
├── VAD Module (Silero v6.0.0 CoreML, energy-based fallback)
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
│   ├── VAD/             # Voice activity detection (Silero v6.0.0)
│   ├── Grammar/         # Rule-based correction (Rust bridge)
│   ├── TTS/             # Text-to-speech [future]
│   └── Core/            # Config, model management
├── Sources/YoozEngineClient/  # Swift Package (thin client SDK)
├── Tests/
├── Vendor/YoozTextCleanup/    # Rust xcframework (prebuilt)
├── text-cleanup/              # Rust source (engine owns this)
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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /v1/health | Service health; reports all module statuses |
| GET | /v1/models | List loaded and available models |
| GET | /v1/stt/languages | Available STT languages |
| GET | /v1/stt/status | STT model load status |
| POST | /v1/stt/load | Load STT model for a language |
| POST | /v1/stt/batch | Batch transcribe audio samples |
| WS | /v1/stt/stream | Real-time streaming STT |
| POST | /v1/llm/generate | LLM text generation |
| POST | /v1/touchup | Text cleanup pipeline (regex + grammar + LLM) |
| POST | /v1/grammar/check | Rule-based grammar correction |
| POST | /v1/vad/detect | Voice activity detection on audio samples |

## Dependencies

| Package | Purpose |
|---------|---------|
| Hummingbird | HTTP server |
| HummingbirdWebSocket | WebSocket support |
| mlx-swift | MLX runtime for Apple Silicon |
| mlx-swift-lm | LLM inference |
| YoozTextCleanup.xcframework | Rust grammar rules |
| FoundationModels | Apple Intelligence on-device LLM (macOS 26+, conditional) |

## Migration Status

| Phase | Module | Source | Status |
|-------|--------|--------|--------|
| 1 | Scaffold | New | [x] Done |
| 2 | STT | yooz-stt-engine | [x] Done |
| 3 | LLM | yooz-stt-engine/TouchUp/LLM | [x] Done |
| 3 | TouchUp | yooz-whisper/TouchUp | [x] Done |
| 4 | Grammar | yooz-stt-engine/text-cleanup | [x] Done |
| 4 | VAD | yooz-whisper/Audio | [x] Done |
| 4.5 | Engine sync | -- | [x] Done (v0.5.0) |
| 5 | Thin client migration | yooz-whisper | Not started |
| 6 | Archive yooz-stt-engine | -- | Not started |
| 7 | TTS (Kokoro) | Future | Not started |

---

*Part of the Yooz ecosystem; Sovereign Intelligence*
