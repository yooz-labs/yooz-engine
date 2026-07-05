# Yooz Engine — Unified Local AI Service

Project-specific agent instructions. The ecosystem-wide rules live in `../AGENTS.md` and apply on top of these.

## Project Overview

- **Product:** Standalone macOS service providing local AI capabilities to all Yooz apps
- **Version:** 0.7.5 (`EngineConfig.version`, `Sources/EngineCore/EngineConfig.swift:42`)
- **Status:** Modular engine + three-transport SDK shipped. Epic #192 ("in-sandbox engine packaging") closed 2026-06-23 via PR #206, phase PRs #196-#205; the 0.7.x hardening arc continued through PR #221 (2026-06-30) under adjacent epics (GPU residency #216, whisper zero-download whisper#253). Follow-on work tracked by epic #230 ("engine substrate contract v2", opened from the 2026-07-02 architecture review) — see `.context/plan.md` for the current sub-issue list.
- **Tech Stack:** Swift 5.9+, SwiftUI, Hummingbird (HTTP/WebSocket), MLX-Swift

All AI modules are complete and synced. The engine is the source of truth for STT, LLM, TouchUp, Grammar, and VAD. The Rust `text-cleanup` source lives in this repo.

## What's NOT in this repo

This is the public engine surface (PolyForm Shield 1.0.0). It does **not** carry training material:

- **Finetune recipes** (configs + scripts that produce the Yooz-Light / Yooz-Quality LLM weights) live in private `yooz-benchmark`. See `yooz-benchmark/finetune-pipeline/`.
- **Per-issue research bundles** (e.g. Qwen3-ASR vs Parakeet TDT, LLM model selection) live in private `yooz-benchmark/research/issue-*/`.
- **Gold-standard datasets** + finetune-ready splits live in private `yooz-benchmark/data/`.
- **Tuned weights** ship openly on HuggingFace under `YoozLabs/...` (Apache 2.0).

Rule of thumb: anything that reveals how an LLM weight was produced (training configs, fine-tune scripts, gold-standard data, per-issue research) is private. The engine that consumes the weight is public. The weight itself is open. (Engine-internal artifacts produced by build scripts, e.g. the Rust `text-cleanup` xcframework, are unrelated to this rule.)

## Architecture

The engine modules are plain `public actor`s — xcodegen framework targets, most of which are also SPM library products (epic #192; `InfiniteModule` is the exception, see "Packaging and transports") — reachable over three interchangeable transports behind one `YoozEngineClient` SDK surface. See "Packaging and transports" below for which transport each consumer uses today. The loopback packaging (`YoozEngine.app`, a macOS menu bar service on `localhost:19920`) is the super-yooz / local-dev shape; App Store standalones (e.g. Yooz Whisper) link the modules in-process instead.

```
Local API Server (localhost:19920, loopback packaging only)
├── REST: /v1/health, /v1/models, /v1/modules
├── REST: /v1/session/{begin,end}
├── REST: /v1/stt/{languages,status,load,batch,engine}
├── REST: /v1/llm/generate
├── REST: /v1/touchup, /v1/touchup/{models,model}
├── REST: /v1/grammar/check
├── REST: /v1/vad/detect
├── REST: /v1/infinite/{models,model,status,sessions,...}
├── WebSocket: /v1/stt/stream
└── Future: /v1/tts/synthesize

Engine modules (framework targets; all but Infinite also SPM products — epic #192)
├── STT Module (Parakeet TDT, FastConformer, Apple STT)
├── LLM Module (MLX: Qwen 0.5B, 1.7B; Apple Intelligence on macOS 26+)
├── TouchUp Module (regex + grammar + LLM pipeline)
├── Grammar Module (Rust text-cleanup xcframework + source)
├── VAD Module (Silero v6.0.0 CoreML, energy-based fallback)
├── Infinite Module (long-context sessions, native MLX backend — loopback-only consumer today)
└── TTS Module [future]

YoozEngineClient (Swift Package) — identical SDK surface over any transport
├── HTTPTransport (loopback), InProcessTransport (YoozEngineInProcess), XPCTransport
├── Auto-discovery + auto-launch (loopback)
└── Shared types
```

**No embedded fallback.** Apps that use the loopback transport auto-launch the engine helper if it's not running; apps using in-process/XPC link the modules directly.

## Repository Layout

```
yooz-engine/
├── YoozEngine/                    # macOS app targets (loopback + module sources)
│   ├── App/                       # Entry, lifecycle
│   ├── Server/                    # Hummingbird HTTP/WS (APIServer.swift)
│   ├── STT/, LLM/, TouchUp/, VAD/, Grammar/, Infinite/, TTS/, Core/
├── Sources/EngineCore/            # Module-agnostic core (SPM product, epic #192)
├── Sources/YoozEngineClient/      # Swift Package (thin client SDK) + Transport/ seam
├── Sources/YoozEngineInProcess/   # In-process EngineTransport facade (SPM product)
├── docs/                          # CONSUMER_INTEGRATION.md, INFINITE_MODULE.md, RELEASE.md
├── Tests/
├── Vendor/YoozTextCleanup/        # Rust xcframework (prebuilt, gitignored; fetched from HF)
├── text-cleanup/                  # Rust source (engine owns this)
└── project.yml                    # XcodeGen
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
and scripted builds need the flag. The dispatch-only CI jobs set it (see
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

**Infinite live suites** (`Tests/InfiniteModuleTests/*LiveTests.swift`, engine#265-268):
gated behind `YOOZ_INFINITE_LIVE=1`, run headless via direct `xcrun xctest`
(the `swift test`/`xcodebuild test` runners don't reliably forward
`DYLD_FRAMEWORK_PATH` to a bundle-hosted `.xctest`, hence the workaround):

```bash
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation -derivedDataPath build \
  -destination 'platform=macOS' build-for-testing -only-testing:InfiniteModuleTests
DYLD_FRAMEWORK_PATH=build/Build/Products/Debug YOOZ_INFINITE_LIVE=1 \
  xcrun xctest -XCTest InfiniteModuleTests.InfiniteQuantizedKVLiveTests \
  build/Build/Products/Debug/InfiniteModuleTests.xctest
```

Swap the `-XCTest` target for any other live class in that directory
(`InfiniteCheckpointResumeLiveTests`, `InfiniteTurnCommitLiveTests`, ...).
The Gemma4-hosted counterpart (`Tests/YoozEngineTests/InfiniteGemma4ParityTests.swift`)
needs `INFINITE_LIVE=1` plus a desktop GUI session (app-hosted XCTest, see
below) — it cannot run from this headless path. Full field/error reference:
`docs/INFINITE_MODULE.md` → "Verification".

### Build verification policy (merge gate) — RULE

**Builds and tests are verified LOCALLY in a sandboxed environment; GitHub CI
runs only cheap Ubuntu checks.** Decided 2026-07-03 (closes #237): hosted
macos runners are billed at a premium and proved slow to useless here — the
post-merge engine-app job never once completed inside its 120-minute timeout.

- **The merge gate for every PR** is local sandboxed verification on the dev
  machine: a clean git worktree with an isolated `-derivedDataPath`, running
  `swift test` (SPM suites) plus `xcodebuild build` for the three app variants
  (`YoozEngine`, `YoozEngineWhisper`, `YoozEngineLite`; add the XPC harness
  scheme when XPC packaging is touched). Agent sessions MUST run this gate
  before opening the PR and report exactly what ran.
- **GitHub CI on PRs/pushes** = SwiftLint (official container on Ubuntu),
  typos spellcheck, path detection. Anything that can run on Ubuntu belongs
  in CI; actual macOS builds do not run automatically.
- **The heavy macOS jobs still exist** behind `workflow_dispatch` in
  `.github/workflows/ci.yml` for on-demand runs (e.g. release tags).
- **App-hosted XCTest suites** (`YoozEngineTests`, hosted by Yooz Engine.app)
  additionally require a desktop GUI session — the runner cannot attach from
  headless/sandboxed contexts. Before running them, kill stale
  `Yooz Engine` processes and check nothing holds port 19920, or the run
  wedges. Live XPC proofs (harness executable) are headless-safe.
- `Vendor/YoozTextCleanup` is gitignored: copy it from the canonical checkout
  (or build from `text-cleanup/`) into any fresh worktree before xcodebuild.
- A local pre-push hook remains tracked in #23.

## API

This table describes the loopback `APIServer` route surface (fixed port **19920**, localhost only). The in-process and XPC transports serve most of the same paths without a socket — see "Packaging and transports" for the routing gaps between transports.

| Method | Path | Description |
|---|---|---|
| GET | `/v1/health` | Service health; reports per-module statuses |
| GET | `/v1/models` | Disk-hygiene model inventory (real on-disk sizes + delete targets) |
| GET | `/v1/modules` | Build variant + per-module health manifest |
| POST | `/v1/session/begin` | Fan `resetForNewSession()` out to every `SessionResettable` module; returns a session id. Drops per-recording state (KV caches, streaming buffers) without unloading models |
| POST | `/v1/session/end` | Same reset fan-out, called at end of a recording |
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
| POST | `/v1/touchup/model` | Picker: set active TouchUp model + preload — non-blocking (engine#226): returns immediately, never awaits a download |
| GET | `/v1/state` | Cross-module snapshot: every module's picker catalog + active id, in one call (engine#226) |
| WS | `/v1/events` | Live push feed of `modelChanged` / `loadStateChanged` / `downloadProgress` / `residencyChanged` events (engine#226) |
| POST | `/v1/grammar/check` | Rule-based grammar correction |
| POST | `/v1/vad/detect` | Voice activity detection (501 on Whisper / Lite variants, no VAD) |
| GET | `/v1/infinite/models` | Picker: long-context models with RAM-tier gating |
| POST | `/v1/infinite/model` | Picker: set active Infinite model + preload |
| GET | `/v1/infinite/status` | Infinite module status |
| GET/POST | `/v1/infinite/sessions` | List / create long-context sessions (max 16 active) |
| GET/DELETE | `/v1/infinite/sessions/:sessionID` | Fetch / delete a session |
| POST | `/v1/infinite/sessions/:sessionID/append` | Append context to a session |
| POST | `/v1/infinite/sessions/:sessionID/generate` | Generate on Swift-runtime-supported models; `501 generation_unavailable` for retrieval-only models |
| POST | `/v1/infinite/sessions/:sessionID/checkpoint` | Checkpoint a session |

`/v1/infinite/*` is 501 `module_not_bundled` on `YoozEngineWhisper` and `YoozEngineLite` (neither links `InfiniteModule`; see "Build Variants" below).

## Packaging and transports

The engine modules are portable — epic #192 (closed 2026-06-23) made this concrete by exposing `EngineCore`, `STTModule`, `LLMModule`, `GrammarModule`, `AppleSTTModule`, and `VADModule` as SwiftPM library products (`Package.swift`, PR #196) alongside the same source dirs the xcodegen framework targets ingest. `InfiniteModule` is the deliberate exception: it exists only as an xcodegen framework target (full `YoozEngine` variant), not as an SPM product — its consumer is the loopback host, and `YoozEngineInProcess` intentionally does not depend on it. The `YoozEngineClient` SDK surface is served by three interchangeable `EngineTransport` implementations (`Sources/YoozEngineClient/Transport/EngineTransport.swift` is the seam):

| Transport | Type | Status as of 2026-07 | Who ships it |
|---|---|---|---|
| **Loopback** (`HTTPTransport`) | HTTP/WebSocket over `127.0.0.1:<port>` | Default, fully shipping | super-yooz host (planned — super-yooz has no code against the substrate yet as of 2026-07) + local dev; the standalone `Yooz Engine.app` helper `.app` shape (`Contents/Helpers/`) fails App Store validation (ITMS-90296) so it is **not** the App Store packaging |
| **In-process** (`InProcessTransport`, product `YoozEngineInProcess`) | Direct calls into the linked module actors, no socket | Shipping — Yooz Whisper links this today (see `yooz-whisper` `project.yml`, engine pinned by revision, currently 0.7.5) | App Store standalone apps |
| **XPC** (`XPCTransport` + `XPCServiceHandler`) | `NSXPCConnection` to a sandboxed XPC service, code-signing pinned | Packaged (#227) and self-contained in Release (#248): `project.yml`'s `YoozEngineXPC` target builds `Contents/XPCServices/YoozEngineXPC.xpc` (Whisper module set via `YoozEngineInProcess`, entitlements per `docs/engine-app-packaging.md`, app-group weights wiring via `AppGroupWeightsLocation`), with `postbuildScripts` (`scripts/embed-xpc-package-frameworks.sh`) embedding + signing the SPM dynamic package frameworks (MLX, MLXNN, Hub, Tokenizers, ...) that xcodegen has no embed support for and that `.xpc` bundles don't auto-embed the way `.app` targets do; `YoozEngineXPCHarness` embeds it (with its own `postbuildScripts` re-sign pass, `scripts/resign-embedded-xpc.sh`, since Xcode's embed-into-app step strips nested `Modules/*.swiftmodule` content without re-signing) and round-trips health + streaming STT — `scripts/verify-xpc-portability.sh` proves this from a Release build copied outside DerivedData. `/v1/events` bridges over the same callback-proxy shape as streaming STT (engine#244). See `docs/CONSUMER_INTEGRATION.md` "XPC service embed recipe" for the consumer-side embed pattern, including the same two `postbuildScripts` phases a consumer's own `.xpc` target needs | Reference packaging only — no shipping app has adopted it yet (Whisper's migration is a follow-up, whisper#267) |

One known gap in `InProcessTransport` (a transport-routing gap, not a module gap — the underlying actors work fine over loopback):

- `/v1/infinite/*` is intentionally unrouted in-process — Infinite's consumer is the loopback host (super-yooz), per the header comment in `InProcessTransport.swift`. `XPCServiceHandler` forwards to `InProcessTransport`, so this gap carries forward into XPC once #227 packages it. (The former `/v1/session/*` gap closed in #232; sessions now dispatch through the shared endpoint table on both transports.)

`/v1/events` (engine#226, the push channel behind `EngineStateStore` — see `docs/CONSUMER_INTEGRATION.md` → "Engine-owned selection state") is now reachable on ALL THREE transports (engine#244 closed the former XPC gap): `XPCTransport.openEvents()` bridges it over the same callback-proxy shape `XPCServiceHandler` already used for streaming STT (`openEvents`/`closeEvents` on `YoozEngineXPCProtocol`, `eventDidOccur`/`eventsDidFinish` pushed to the client's exported `YoozEngineXPCStreamClientProtocol` object), one `EngineEventBus` subscription per `openEvents()` call (in practice one per client connection, since `EngineStateStore` opens exactly one; the wire protocol itself supports N concurrent subscriptions keyed by client-generated id), torn down on `closeEvents` or connection interruption/invalidation. The returned `AsyncStream` FINISHES (no further frames) rather than silently going quiet when the connection dies — a caller that wants events across a service crash/respawn must detect the finish and re-subscribe on a reconnected transport (see `XPCTransport.openEvents()`'s doc and `ReconnectingXPCTransport` in yooz-whisper). `/v1/events` needs no `RouteParityAllowlist` entry on any transport — see `RouteManifest`'s `/v1/events` entry doc.

Route drift between `APIServer` and `InProcessTransport` is caught by two automated layers: `RouteParityTests` (#223/#233 — every manifest route must be in-process-reachable or explicitly allowlisted with a reason) and, for converted families, the typed endpoint table (see "Typed endpoint table" below), which makes drift structurally impossible by deriving both transports from one declaration.

Full design rationale + macOS feasibility research (sandboxing, entitlements, Metal/MLX-under-XPC): `docs/engine-app-packaging.md` in the private `yooz` ecosystem repo (sibling checkout, e.g. `../yooz/docs/engine-app-packaging.md`).

## Typed endpoint table

Converted route families are declared exactly once (engine#225 Phase B) and
both transports derive from that declaration:

- **`EndpointSpecs`** (`Sources/EngineCore/EndpointSpecs.swift`) is the
  single declaration of every REST route's `(method, path)`. `RouteManifest`
  is a pure projection of it (plus the two WebSocket routes, `/v1/stt/stream`
  and `/v1/events` — neither is REST-dispatchable so neither lives in
  `EndpointSpecs`), pinned by
  `EndpointTableTests.testManifestIsAProjectionOfTheSpecCatalog`. Add or
  rename a route in the catalog, nowhere else.
- **Handlers** for converted families are single-homed `WireHandler`
  closures living in the module that owns the actor, so every product that
  links the module compiles the same body: `SessionEndpoints` (EngineCore),
  `TouchUpEndpoints` + `ModelManagementEndpoints` (LLMModule; the one
  STT-owned input is injected as a closure since LLMModule cannot depend on
  STTModule). Handlers throw `WireError` — rendered as the standard
  `{"error", "code"}` body on loopback and rethrown as
  `YoozEngineError.serverError` in-process, so the two transports share one
  error vocabulary by construction.
- **Dispatch**: `APIServer.register(_:on:)` registers each table entry on
  the Hummingbird router through one generic adapter;
  `InProcessTransport.dispatchViaTable` consults the table before its legacy
  switch. `RouteParityTests.testEndpointTableCoversExactlyTheConvertedSpecs`
  pins that the table binding and `EndpointSpecs.converted` never drift.

Converted so far: session (`/v1/session/*`), TouchUp picker
(`/v1/touchup/model[s]`), model management (`/v1/models*`), engine state
(`GET /v1/state`, engine#226). The remaining families are hand-implemented
per transport (`EndpointSpecs.legacy`); the conversion order and
per-family blockers (e.g. `POST /v1/stt/engine`'s MainActor UI-state
coupling) are tracked on engine#225.

Converting a family: move its specs from `legacy` to `converted` in
`EndpointSpecs`, write the shared handlers in the owning module as
`[Endpoint]`, add them to BOTH transport bindings (`APIServer.buildRouter`'s
`register(...)` call and `InProcessTransport.endpointTable`), and delete the
hand-written route registrations + in-process switch cases. The parity and
table tests enforce the rest.

## Single wire-type home

Every request/response DTO the loopback server, the SDK, and the in-process
transport share is declared exactly once, in `Sources/YoozEngineWire/`
(engine#225). `YoozEngineWire` is a dependency-free SwiftPM target (Foundation
only) so both `EngineCore` (engine side) and `YoozEngineClient` (SDK side)
can depend on it without pulling anything else in — it is deliberately its
own target rather than folded into `EngineCore`, which also carries engine
orchestration (`AIModule`, `ModuleRegistry`, `SessionCoordinator`,
`ModelStore`, `MLXResidency`, ...) that the thin-client SDK has no business
re-exporting to consumer apps.

Both `EngineCore` and `YoozEngineClient` `@_exported import YoozEngineWire`
(`Sources/EngineCore/WireReexport.swift`,
`Sources/YoozEngineClient/WireReexport.swift`), so every existing
`import EngineCore` / `import YoozEngineClient` call site keeps resolving
wire types (`ModulesResponse`, `TouchUpModelInfo`, `LLMStatus`, ...)
unqualified — no source changes needed downstream. `YoozEngine.xcodeproj`
mirrors this via `project.yml`'s `YoozEngineWire` framework target, linked
everywhere `EngineCore` is.

A handful of module-owned domain types happen to share a name with a wire
type but are NOT the same shape (e.g. `STTModule.AlignedToken` uses
`start` + `duration` while the wire `AlignedToken` uses `start` + `end`,
and `STTModule.TranscriptionResult` is a token container with computed
text, unlike the wire `TranscriptionResult`'s flat text fields;
`LLMModule`'s internal `LLMModelInfo` has fields `type`/`isLoaded`/`isCached`,
unrelated to the wire `LLMModelInfo`). Call sites that import both the
domain module and a wire re-export qualify explicitly
(`YoozEngineWire.TranscriptionResult`, `STTModule.TranscriptionResult`) —
`Sources/YoozEngineInProcess/SDKTypeAliases.swift` documents the in-process
transport's cases; `YoozEngine/Server/APITypes.swift` documents the
loopback server's.

Not every DTO moved: `/v1/health`'s `EngineModules.detail` carries
`ModuleDetailMap`, whose value type lives in the app target's
`ModuleEagerLoader.swift` (eager-load diagnostics, not a plain DTO) — moving
it was out of scope for #225. The `/v1/infinite/*` family and the
`/v1/stt/stream` WebSocket frame types are also unmoved: Infinite is
loopback-only by design (see "Packaging and transports" above) and wasn't
named in #225's family list; the WS frames are a loopback-only wire
protocol, not a REST body.

Adding a new shared DTO: define it once in `Sources/YoozEngineWire/`, add a
`ResponseEncodable`/`ResponseCodable` Hummingbird conformance extension in
`YoozEngine/Server/APITypes.swift` (the wire-transport concern; the target
itself must stay Hummingbird-free), and add a decode-compat fixture test
under `Tests/YoozEngineWireTests/` if the DTO is replacing an existing
shape. One xcodegen gotcha: a new `project.yml` target that depends on
`EngineCore` (or any module framework) must also declare
`- target: YoozEngineWire` — the `@_exported import` satisfies the
compiler but not the linker, so omitting it fails at link time.

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
| Infinite | `InfiniteModelSelection` | `InfiniteModelInfo` / `InfiniteModelsResponse` (+ RAM-tier gating fields) | `GET/POST /v1/infinite/model[s]` |

### SDK

Two methods per module client:

```swift
public func availableModels() async throws -> ModelsResponse
@discardableResult
public func setModel(id: String, preload: Bool = true) async throws -> ModelInfo
```

Wire types (`ModelInfo`, `ModelsResponse`, `SetModelRequest`) live once in `Sources/YoozEngineWire/` (#225 wire-type consolidation) — a dependency-free SPM target the server (`EngineCore` re-export), the SDK (`YoozEngineClient` re-export), and the in-process transport all resolve through their existing imports. Consumer apps still only depend on the SDK module; they never import `YoozEngineWire` directly.

### App

**Canonical wiring (engine#226): `EngineStateStore`, not a hand-rolled `ModelPickerStore` + `SettingsManager` key.** For a module that has adopted the engine-owned-selection contract (persisted selection + non-blocking `setModel` + `/v1/events`; TouchUp today), the app-side store has zero local persistence and zero polling:

```swift
@StateObject private var state = EngineStateStore(client: client)
// state.start() at mount (GET /v1/state, then subscribe to /v1/events);
// state.stop() at teardown. state.modules["touchup"] drives the picker UI;
// state.latestEvent(module:kind:) surfaces in-flight downloadProgress.
```

The prior recipe (`ModelPickerStore<T>` + `SettingsManager.<module>Model`) remains valid for a module that has NOT adopted the contract yet (STT engine, Infinite as of 2026-07) or that needs extension fields `EngineModelSnapshotRow`'s seven canonical fields don't carry — layer its refresh signal off `EngineStateStore.latestEvent(module:kind:)` rather than a poll loop when both apply. Full recipe + adoption status: `docs/CONSUMER_INTEGRATION.md` → "Engine-owned selection state".

### Adding a new picker

1. Define `<Module>ModelSelection: String, Codable, Sendable, CaseIterable` in the module's directory (e.g. `STT/STTBackendSelection.swift`).
2. Add `private(set) var activeModel: <Module>ModelSelection` to the module's actor.
3. Implement `availableModels() -> [ModelInfo]` and `setActiveModel(_:preload:)` on the actor.
4. Wire `GET /v1/<module>/models` and `POST /v1/<module>/model` in `APIServer`.
5. Define `ModelInfo` / `ModelsResponse` / `SetModelRequest` once, in `Sources/YoozEngineWire/<Module>WireTypes.swift` (not the SDK or the server — see "Single wire-type home" below).
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

Each consuming app bundles only the modules it needs (`project.yml` target `dependencies:` is the source of truth). See `.context/phase5_epic.md`.

| Variant | Modules | Consumer | Bundle |
|---|---|---|---|
| `YoozEngine` | STT + AppleSTT + Grammar + LLM + VAD + **Infinite** | Standalone menu bar service (dev-only, loopback) / super-yooz host | full |
| `YoozEngineWhisper` | STT + AppleSTT + Grammar + LLM (no VAD, **no Infinite**) | Yooz Whisper helper (in-process today) | full-minus-VAD-minus-Infinite |
| `YoozEngineLite` | AppleSTT + Grammar + LLM (no MLX STT, no VAD, no Infinite) | Remi-class apps, iOS | sub-GB |

VAD stays whisper-embedded (~64ms call rate makes HTTP round-trip non-viable). Infinite is loopback-only by design (its consumer is the loopback host, per `InProcessTransport.swift`), so it is dropped from `YoozEngineWhisper` and `YoozEngineLite` regardless of transport.

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
| 5 | Modular engine + whisper thin-client | — | [x] Done (v0.6.0) |
| 6 | Archive yooz-stt-engine | — | [x] Done — `yooz-stt-engine` archived 2026-05-08 (issue #79, PR #80 in that repo) |
| 6.5 | InfiniteModule (long-context, native MLX backend) | New | [x] Done — epic (engine#171), Phases 1-7, PRs #166-#170/#183/#185; follow-on Gemma4 model onboarding in #188/#189 and (partly) #190 |
| 7 | In-sandbox packaging: SPM module products, in-process facade, XPC transport | New | [x] Done — epic #192 closed 2026-06-23 via PR #206 (phase PRs #196-#205); 0.7.x hardening continued through #221 under adjacent epics |
| 8 | Engine substrate contract v2 (session routing parity, XPC packaging, model-selection state) | New | In flight — epic #230 (2026-07-02 architecture review), tracking #222-#229; see `.context/plan.md` |
| 9 | TTS (Kokoro) | Future | Not started |

## Versioning and releases

Patch-bump on every batch of work merged to `main`; the 0.7.x line patch-bumps
even for feature epics. One release = one commit + one tag:

1. Bump BOTH version strings to `X.Y.Z` in the same commit:
   - `Sources/EngineCore/EngineConfig.swift` (`public static let version`)
   - `YoozEngine/Info.plist` (`CFBundleShortVersionString`)
2. Commit `bump version to X.Y.Z` on `main`, tag `vX.Y.Z`, push commit + tag.
3. The tag push triggers `release-notes.yml`, which creates a **draft** GitHub
   release (no macOS runners for this repo, engine#23; artifacts are
   maintainer-built).
4. Build + sign locally: `bash scripts/release-engine.sh` (emits
   `dist/*.app.zip` + `dist/RELEASE.md`), then
   `bash scripts/smoke-test-release.sh` (all three variants must print ALIVE;
   port 19920 must be free).
5. Upload and publish:

   ```bash
   gh release upload vX.Y.Z dist/YoozEngine.app.zip dist/YoozEngineLite.app.zip \
       dist/YoozEngineWhisper.app.zip dist/RELEASE.md
   gh release edit vX.Y.Z --notes-file dist/RELEASE.md
   gh release edit vX.Y.Z --draft=false
   ```

Do not leave a tag without a published release: "Latest" on GitHub is the
artifact source for consumers, and SDK consumers resolve versions from tags.
Full runbook, prerequisites (Developer ID cert, xcodegen), and
troubleshooting: `docs/RELEASE.md`.

## Conventions

- **Swift 6 concurrency:** use `@MainActor` on classes that hold `@Published` state.
- **Singletons:** Grammar and VAD are actor singletons. VAD requires `load()` for the CoreML model.
- **HummingbirdWebSocket:** inbound yields `WebSocketDataFrame`; use `.messages(maxSize:)` for `WebSocketMessage`.
- **Bundle id:** `live.yooz.engine`.

---

*Part of the Yooz ecosystem. Sovereign Intelligence.*
