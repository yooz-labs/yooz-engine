# Yooz Engine — Unified Local AI Service

Project-specific agent instructions. The ecosystem-wide rules live in `../AGENTS.md` and apply on top of these.

## Project Overview

- **Product:** Standalone macOS service providing local AI capabilities to all Yooz apps
- **Version:** 0.5.0
- **Status:** Phase 5 — Thin Client Migration (ready)
- **Tech Stack:** Swift 5.9+, SwiftUI, Hummingbird (HTTP/WebSocket), MLX-Swift

All AI modules are complete and synced. The engine is the source of truth for STT, LLM, TouchUp, Grammar, and VAD. The Rust `text-cleanup` source lives in this repo.

## Architecture

`YoozEngine.app` is a macOS menu bar service running a local API on `localhost:19920`. All Yooz apps (Whisper, Notes, Voice, Crisp, Remi) are thin clients that hit this API via the `YoozEngineClient` Swift Package.

```
YoozEngine.app (menu bar service)
├── Local API Server (localhost:19920)
│   ├── REST: /v1/health, /v1/models
│   ├── REST: /v1/stt/{languages,status,load,batch}
│   ├── REST: /v1/llm/generate
│   ├── REST: /v1/touchup
│   ├── REST: /v1/grammar/check
│   ├── REST: /v1/vad/detect
│   ├── WebSocket: /v1/stt/stream
│   └── Future: /v1/tts/synthesize
├── STT Module (Parakeet TDT, FastConformer, Apple STT)
├── LLM Module (MLX: Qwen 0.5B, 1.7B; Apple Intelligence on macOS 26+)
├── TouchUp Module (regex + grammar + LLM pipeline)
├── Grammar Module (Rust text-cleanup xcframework + source)
├── VAD Module (Silero v6.0.0 CoreML, energy-based fallback)
└── TTS Module [future]

YoozEngineClient (Swift Package)
├── Auto-discovery + auto-launch
├── REST + WebSocket clients
└── Shared types
```

**No embedded fallback.** Apps auto-launch the engine if it's not running.

## Repository Layout

```
yooz-engine/
├── YoozEngine/                # macOS app (menu bar service)
│   ├── App/                   # Entry, lifecycle
│   ├── Server/                # Hummingbird HTTP/WS
│   ├── STT/, LLM/, TouchUp/, VAD/, Grammar/, TTS/, Core/
├── Sources/YoozEngineClient/  # Swift Package (thin client SDK)
├── Tests/
├── Vendor/YoozTextCleanup/    # Rust xcframework (prebuilt)
├── text-cleanup/              # Rust source (engine owns this)
└── project.yml                # XcodeGen
```

## Build

```bash
xcodegen generate
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation \
  -derivedDataPath build build
open build/Build/Products/Debug/Yooz\ Engine.app
curl http://localhost:19920/v1/health
```

`-derivedDataPath build` keeps the artifact next to the source tree; without it, xcodebuild writes to `~/Library/Developer/Xcode/DerivedData/<hash>/Build/Products/Debug/`.

`-skipMacroValidation` is required from the CLI because `MLXHuggingFaceMacros`
(used by `#huggingFaceTokenizerLoader`) is an external Swift macro that Xcode
otherwise prompts to trust on first run. The Xcode UI handles this prompt; CLI
and CI builds need the flag. CI sets it automatically (see
`.github/workflows/ci.yml`).

### Running tests

```bash
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation \
  -destination 'platform=macOS' test
```

The TurboQuant live integration test is gated behind `KVCOMPRESSION_LIVE=1`
so CI does not need the Qwen3-1.7B weights cached. To run it locally with a
cached model:

```bash
KVCOMPRESSION_LIVE=1 xcodebuild -project YoozEngine.xcodeproj \
  -scheme YoozEngine -skipMacroValidation \
  -destination 'platform=macOS' test
```

## API

Fixed port: **19920** (localhost only).

| Method | Path | Description |
|---|---|---|
| GET | `/v1/health` | Service health; reports per-module statuses |
| GET | `/v1/models` | Loaded and available models |
| GET | `/v1/modules` | Build variant + per-module health manifest |
| GET | `/v1/stt/languages` | Available STT languages |
| GET | `/v1/stt/status` | STT model load status |
| GET | `/v1/stt/engine` | Current + available STT backends + capability flags |
| POST | `/v1/stt/engine` | Switch backend: `parakeet` / `fast_conformer` / `apple_stt` |
| POST | `/v1/stt/load` | Load STT model for a language |
| POST | `/v1/stt/batch` | Batch transcribe (`aligned=true` for token timestamps) |
| WS | `/v1/stt/stream` | Real-time streaming STT |
| POST | `/v1/llm/generate` | LLM text generation |
| POST | `/v1/touchup` | Text cleanup pipeline |
| GET | `/v1/touchup/models` | Picker: list TouchUp models with state |
| POST | `/v1/touchup/model` | Picker: set active TouchUp model + preload |
| POST | `/v1/grammar/check` | Rule-based grammar correction |
| POST | `/v1/vad/detect` | Voice activity detection (501 on Whisper / Lite variants) |

## Module model picker pattern

This is the **canonical wiring shape** every module that exposes a model picker (TouchUp today, STT engine + TTS voice next) MUST follow. Get it right once and the SDK + UI layers in every consumer app (whisper, notes, voice, crisp, remi) become a template instead of a redesign per module.

### Engine

Two routes per module:

```
GET  /v1/<module>/models   → ModelsResponse { models: [ModelInfo], activeId: String }
POST /v1/<module>/model    body { id: String, preload: Bool? } → ModelInfo (the new active row)
```

`ModelInfo` fields below MUST appear on every picker; modules MAY add **optional** extension fields per the "Module-specific picker extensions" subsection below.

| Field | Type | Notes |
|---|---|---|
| `id` | String | Stable wire id; never rename without major SDK bump |
| `displayName` | String | Picker-visible name |
| `description` | String | One-line subtitle for picker UX |
| `tier` | enum (typed) | `light` / `quality` / `premium` / `unknown` (SDK-side `unknown` is the forward-compat fallback so a newer engine can ship a fifth tier without breaking older clients) |
| `sizeBytes` | Int64? | Approximate on-disk size; `nil` for OS-provided backends (`.premium` tier) |
| `loadState` | enum (typed) | `unavailable < available < cached < loaded`. Replaces the old three-flag pattern (`isAvailable / isCached / isLoaded`) with a total-ordering enum so illegal combinations like "loaded but not cached" are unrepresentable |
| `isActive` | Bool | Server guarantees exactly one row has `isActive == true` (precondition'd in `availableModels()`) |

Engine state lives in the module's actor (e.g. `TouchUpEngine.activeModel`); the primary processing entry point (`/v1/touchup`) routes through the active model.

Field semantics across modules:

- `loadState` for OS-provided backends (Apple Intelligence, Apple STT) reports `.cached` whenever the OS reports the backend as available — there is no on-disk artifact the engine controls, so `.available` and `.cached` collapse for that tier. Document the convention in the module's selection enum.
- `loadState == .available` for the user's perspective means "selectable now without a config change". For Apple Intelligence specifically, the engine cannot read the user's opt-in state without attempting a load; `availableModels()` may report `.available` for a model that fails to actually load. The route handler maps that load failure to `501 model_unavailable` so the picker UI can render a contextual error.

### Wire codes

Every picker route maps these error codes consistently. Picker UIs branch on `code` to render contextual messages — renaming a code is a user-visible regression.

| HTTP | code | When |
|---|---|---|
| 400 | `invalid_request` | Body fails to decode |
| 400 | `invalid_model` | `id` not in the module's selection enum |
| 501 | `model_unavailable` | Backend declared selectable per `loadState` but cannot actually load on this system (e.g. Apple Intelligence on pre-26 macOS) |
| 500 | `model_set_failed` | Underlying load failure (network, OOM, weights corrupt) |

### Module-specific picker extensions

The canonical fields above MUST appear on every picker so the SDK's `ModelPickerStore<T>` template stays generic. When a module genuinely needs additional state that doesn't fit (e.g. `STTBackendInfo` carries `supportsBatch`, `supportsStreaming`, `supportedLanguages`), add them as **optional** fields on the same struct rather than inventing a parallel response shape. Picker UIs that don't care about the extension ignore it; the few that do read it directly. Document the extension in the module's `<Module>BackendID.swift` source-of-truth file.

Adopters today:

| Module | Selection enum | Picker types | Routes |
|---|---|---|---|
| TouchUp | `TouchUpModelSelection` | `TouchUpModelInfo` / `TouchUpModelsResponse` | `GET/POST /v1/touchup/model[s]` |
| STT engine | `STTBackendID` | `STTBackendInfo` / `STTBackendsResponse` (+ `supportsBatch`, `supportsStreaming`, `supportedLanguages`) | `GET/POST /v1/stt/engine` |

### SDK

Two methods per module client:

```swift
public func availableModels() async throws -> ModelsResponse
@discardableResult
public func setModel(id: String, preload: Bool = true) async throws -> ModelInfo
```

Wire types (`ModelInfo`, `ModelsResponse`, `SetModelRequest`) live in `Sources/YoozEngineClient/Types/<Module>Types.swift` so consumer apps only depend on the SDK module.

### App

```swift
@StateObject private var picker = ModelPickerStore<TouchUpModelInfo>(
    fetch: { try await client.touchUp.availableModels().models },
    setActive: { id in try await client.touchUp.setModel(id: id, preload: true) }
)
```

UI binds to `picker.options` and writes through `picker.select(id:)`; the store handles refresh + persistence to `SettingsManager.<module>Model`. The same store works for every module.

### Adding a new picker

1. Define `<Module>ModelSelection: String, Codable, Sendable, CaseIterable` in the module's directory (e.g. `STT/STTBackendSelection.swift`).
2. Add `private(set) var activeModel: <Module>ModelSelection` to the module's actor.
3. Implement `availableModels() -> [ModelInfo]` and `setActiveModel(_:preload:)` on the actor.
4. Wire `GET /v1/<module>/models` and `POST /v1/<module>/model` in `APIServer`.
5. Mirror `ModelInfo` / `ModelsResponse` / `SetModelRequest` in `Sources/YoozEngineClient/Types/<Module>Types.swift`.
6. Add `availableModels()` + `setModel(id:preload:)` to `<Module>Client.swift`.
7. Pin the wire id contract with a `<Module>SelectionTests` file (mirrors `STTLanguageHuggingFaceIDTests` shape).

The TouchUp module is the reference implementation — copy from there.

## Dependencies

| Package | Purpose |
|---|---|
| Hummingbird | HTTP server |
| HummingbirdWebSocket | WebSocket support |
| mlx-swift | MLX runtime for Apple Silicon |
| mlx-swift-lm | LLM inference |
| YoozTextCleanup.xcframework | Rust grammar rules |
| FoundationModels | Apple Intelligence on-device LLM (macOS 26+, conditional) |

## Build Variants (Phase 5)

Each consuming app bundles only the modules it needs. See `.context/phase5_epic.md`.

| Variant | Modules | Consumer | Bundle |
|---|---|---|---|
| `YoozEngine` | STT + AppleSTT + Grammar + LLM + VAD | Standalone menu bar service | full |
| `YoozEngineWhisper` | STT + AppleSTT + Grammar + LLM (no VAD) | Yooz Whisper helper | full-minus-VAD |
| `YoozEngineLite` | AppleSTT + Grammar + LLM (no MLX STT, no VAD) | Remi-class apps, iOS | sub-GB |

VAD stays whisper-embedded (~64ms call rate makes HTTP round-trip non-viable).

## Migration Status

| Phase | Module | Source | Status |
|---|---|---|---|
| 1 | Scaffold | New | [x] Done |
| 2 | STT | yooz-stt-engine | [x] Done |
| 3 | LLM | yooz-stt-engine/TouchUp/LLM | [x] Done |
| 3 | TouchUp | yooz-whisper/TouchUp | [x] Done |
| 4 | Grammar | yooz-stt-engine/text-cleanup | [x] Done |
| 4 | VAD | yooz-whisper/Audio | [x] Done |
| 4.5 | Engine sync | — | [x] Done (v0.5.0) |
| 5 | Modular engine + whisper thin-client | — | [x] Done (v0.6.0); integration hardening in flight |
| 6 | Archive yooz-stt-engine | — | PR #80 open; archive after whisper → stable |
| 7 | TTS (Kokoro) | Future | Not started |

## Conventions

- **Swift 6 concurrency:** use `@MainActor` on classes that hold `@Published` state.
- **Singletons:** Grammar and VAD are actor singletons. VAD requires `load()` for the CoreML model.
- **HummingbirdWebSocket:** inbound yields `WebSocketDataFrame`; use `.messages(maxSize:)` for `WebSocketMessage`.
- **Bundle id:** `live.yooz.engine`.

---

*Part of the Yooz ecosystem. Sovereign Intelligence.*
