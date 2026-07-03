# Consumer Integration Guide

How to integrate Yooz Engine (0.7.5 as of 2026-07) into a Swift/SwiftUI app.

The `YoozEngineClient` SDK surface is identical across three transports (loopback HTTP/WS, in-process, XPC) — see "Transport selection" below to pick the right one for your packaging. The rest of this document, unless a section says otherwise, describes the **loopback transport** (`HTTPTransport`), which remains the right choice for local dev and for the super-yooz host. **App Store standalone apps should use the in-process transport today** (see "Transport selection"); it is what Yooz Whisper ships.

## TL;DR (loopback transport)

```swift
// Package.swift / project.yml — pin by REVISION, not version. The engine
// declares mlx-swift-lm by revision, and SwiftPM forbids a version-pinned
// package from depending on a revision-pinned one, so `from:` pins fail to
// resolve. yooz-whisper's project.yml documents the same rule.
.package(url: "https://github.com/yooz-labs/yooz-engine", revision: "<engine-commit-sha>")

// Anywhere in your app
import YoozEngineClient
let client = YoozEngineClient()
try await client.connect()
let result = await client.touchUp.touchUp(text: "um yeah", mode: .standard)
```

That's the happy path for the loopback transport. The SDK auto-discovers a bundled helper under `<host>.app/Contents/Helpers/` (preferred), or falls back to a LaunchServices lookup for a standalone Yooz Engine.app (legacy). It launches the helper headless (`--headless` argv flag plus `YOOZ_ENGINE_HEADLESS=1` env var, see "Helper-mode signal" below) so your host app's menu-bar UI stays clean.

## Transport selection

Pick a transport based on how your app ships, not on which modules you need (module selection is the separate "Variant selection" below):

| Transport | Use when | Status as of 2026-07 |
|---|---|---|
| **In-process** (`YoozEngineInProcess` SPM product, `InProcessTransport`) | You're an App Store standalone app. No socket, no separate process — the engine actors run in your sandbox. **Recommended today.** Yooz Whisper ships this (pinned by revision in its `project.yml`, engine 0.7.5 at time of writing). | Shipping. One gap: `/v1/infinite/*` is intentionally unsupported in-process — Infinite's consumer is the loopback host. (The former `/v1/session/*` gap closed in engine#232; sessions dispatch through the shared endpoint table on every transport.) |
| **XPC** (`XPCTransport` + a packaged `.xpc` service) | You're an App Store standalone app and want process-isolated crash/OOM containment (an engine crash does not take your app down). This is the **preferred long-term shape** per the packaging design (avoids ITMS-90296 the way an unsandboxed helper `.app` cannot). | **Packaged, self-contained, and provable (engine#227, engine#248), full route surface (engine#244).** This repo's own `project.yml` ships `YoozEngineXPC` (the `.xpc` target), entitlements, and a dev-only harness that round-trips health + streaming STT through it in a Release build copied outside DerivedData (`scripts/verify-xpc-portability.sh`) — see "XPC service embed recipe" below. `/v1/events` bridges over the same callback-proxy shape as streaming STT (engine#244), so a picker built on `EngineStateStore` now works on all three transports. No shipping consumer app has adopted XPC yet; Whisper's migration is a follow-up (whisper#267), in-process stays its fallback in the meantime. |
| **Loopback** (`HTTPTransport`, this document's default) | You're the super-yooz host, or doing local dev/testing. Not valid for an App Store standalone — the unsandboxed helper `.app` under `Contents/Helpers/` fails App Store review (ITMS-90296). | Shipping, default. |

In-process integration looks like this instead of the TL;DR above:

```swift
// Package.swift / project.yml — pin by REVISION (same rule as the TL;DR:
// the engine's mlx-swift-lm revision pin makes `from:` pins unresolvable).
// Pick a revision at or past the v0.7.x line — YoozEngineInProcess first
// shipped in 0.7.0 (epic #192). yooz-whisper's project.yml is the working
// reference for this exact setup.
.package(url: "https://github.com/yooz-labs/yooz-engine", revision: "<engine-commit-sha>")
// Link the YoozEngineInProcess product (pulls in EngineCore + the module
// products your variant needs) instead of just YoozEngineClient.

import YoozEngineClient
import YoozEngineInProcess

let client = YoozEngineClient(transport: InProcessTransport())
try await client.connect()
let result = await client.touchUp.touchUp(text: "um yeah", mode: .standard)
```

Full design rationale + macOS feasibility research (sandboxing, entitlements, Metal/MLX-under-XPC): `docs/engine-app-packaging.md` in the private `yooz` ecosystem repo (sibling checkout).

## XPC service embed recipe

This is the "preferred long-term shape" row from the table above, spelled out.
`YoozEngineXPC` + `YoozEngineXPCHarness` in this repo's own `project.yml` are
the reference implementation (engine#227) — copy their shape into your own
app's project rather than re-deriving it. Adopting this transport is a bigger
lift than in-process (a real `.xpc` target, entitlements, an app group), so
budget it as its own piece of work; in-process remains the recommended
starting point.

### 1. Add the XPC service target

```yaml
# your app's project.yml
YourAppXPC:
  type: xpc-service
  platform: macOS
  sources:
    - XPCService   # a handful of lines — copy XPCService/main.swift verbatim
  settings:
    base:
      PRODUCT_BUNDLE_IDENTIFIER: live.yooz.yourapp.xpc
      PRODUCT_NAME: YourAppXPC
      INFOPLIST_FILE: XPCService/Info.plist
      CODE_SIGN_ENTITLEMENTS: XPCService/YourAppXPC.entitlements
      SKIP_INSTALL: "YES"
  dependencies:
    - package: yooz-engine   # your existing package: block, pinned by revision
      product: EngineCore
    - package: yooz-engine
      product: YoozEngineClient
    - package: yooz-engine
      product: YoozEngineInProcess
```

`XPCService/main.swift` forwards everything to `InProcessTransport` — the
SAME transport your in-process build already links, so there's no second
route table to maintain:

```swift
import EngineCore
import Foundation
import YoozEngineClient
import YoozEngineInProcess

// See "Weights/app-group wiring" below for what this line does.
if let groupID = Bundle.main.object(forInfoDictionaryKey: "YoozAppGroupIdentifier") as? String {
    AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: groupID)
}

let delegate = XPCServiceListenerDelegate { XPCServiceHandler(transport: InProcessTransport()) }
let listener = NSXPCListener.service()   // system-launched; resume() does not return
listener.delegate = delegate
listener.resume()
```

`XPCService/Info.plist` needs `CFBundlePackageType = XPC!` and an
`XPCService` dict with `ServiceType = Application` (not `System`) —
`Application` is what lets MLX/Metal work inside the sandboxed service with
no GPU entitlement (proven precedent: Pico AI Server, cited in
`../yooz/docs/engine-app-packaging.md`). See this repo's `XPCService/Info.plist`
for the full template, including the `YoozAppGroupIdentifier` key the
snippet above reads.

**Self-containment (engine#248):** an `xpc-service` bundle does not get
Xcode's automatic "Embed Frameworks" treatment for SPM dynamic package
products the way an `.app` target does — `EngineCore`/`YoozEngineInProcess`
and everything they pull in transitively (MLX, MLXNN, MLXLMCommon, Hub,
Tokenizers, ...) build into the shared `PackageFrameworks` build products
directory but never get copied into `YourAppXPC.xpc`'s own
`Contents/Frameworks`. Debug builds are masked by an absolute,
machine-local DerivedData `LC_RPATH` Xcode also writes; a Release build
crash-loops at spawn with `dyld: Library not loaded: @rpath/MLX.framework/...`.
Add the equivalent of this repo's `scripts/embed-xpc-package-frameworks.sh`
as a `postbuildScripts` phase (`basedOnDependencyAnalysis: false`) on your
own `YourAppXPC` target — it copies `PackageFrameworks/*.framework` into
the `.xpc`'s `Contents/Frameworks` and signs each copy with
`$EXPANDED_CODE_SIGN_IDENTITY` so they satisfy hardened-runtime library
validation (which requires the loaded framework to share the loading
binary's Team ID). Copy the script's header comment along with it — it
documents why signing has to happen there and not be left to Xcode's own
implicit sign step for this bundle.

### 2. Entitlements — sandboxed, never `inherit`

```xml
<key>com.apple.security.app-sandbox</key><true/>
<key>com.apple.security.network.client</key><true/>
<key>com.apple.security.application-groups</key>
<array><string>$(TeamIdentifierPrefix)live.yooz.<yourapp>.shared</string></array>
```

Never `com.apple.security.inherit` — that entitlement is for child processes
an app spawns itself. A system-launched XPC service needs its own sandbox;
`inherit` is exactly the mechanism that made the deprecated nested-helper-`.app`
shape (`Contents/Helpers/`) trip ITMS-90296. See this repo's
`XPCService/YoozEngineXPC.entitlements` for the annotated original.

### 3. Embed under your app's `Contents/XPCServices/`

```yaml
YourApp:
  dependencies:
    - target: YourAppXPC
      embed: true
      codeSign: true
      link: false   # a target dependency + Copy Files phase, never linked
```

xcodegen infers the `Contents/XPCServices/` destination automatically from
the dependency's `xpc-service` product type — no manual Copy Files phase
needed.

**A second, distinct self-containment gap (engine#248) surfaces right
here:** Xcode's embed step strips `Modules/*.swiftmodule` interface
content from each nested framework inside the `.xpc` as part of copying it
into `YourApp` — a size optimization — without re-signing the now-smaller
framework, which invalidates its signature (`codesign --verify` reports "a
sealed resource is missing or invalid"). Add a second `postbuildScripts`
phase to `YourApp` (this repo's `scripts/resign-embedded-xpc.sh` is the
template) that re-signs every framework under
`Contents/XPCServices/YourAppXPC.xpc/Contents/Frameworks` plus the `.xpc`
itself, AFTER the embed Copy Files phase runs. Both this and the step 1
script only need `$EXPANDED_CODE_SIGN_IDENTITY` to resolve to your app's
real signing identity to work correctly in a shipping build — this repo's
own dev-only ad-hoc default (`CODE_SIGN_IDENTITY: "-"`) additionally needs
`com.apple.security.cs.disable-library-validation` to spawn locally
(`scripts/dev-adhoc-xpc.entitlements`, used only by
`scripts/verify-xpc-portability.sh`), because two independently ad-hoc
signed artifacts never share a Team ID; skip that entitlement in your own
entitlements once you're signing with a real identity, since matching Team
IDs already satisfies library validation without it.

### 4. Talk to it from your app

```swift
import YoozEngineClient

let transport = XPCTransport(serviceName: "live.yooz.yourapp.xpc")
let client = YoozEngineClient(transport: transport)
try await client.connect()   // GET /v1/health across XPC
```

Pin the peer's code signature once you have a real signing identity:
`XPCTransport(serviceName:codeSigningRequirement:)` — the first-class trust
check loopback has no equivalent for.

### Weights/app-group wiring

STT/LLM weights download via `network.client` into the HuggingFace (HF) hub
cache on first use (AGENTS.md "HF model auto-download"). Left alone, a
sandboxed process's HF cache resolves inside ITS OWN container — meaning
your app and its XPC service would each hold a separate copy of the same
multi-hundred-MB-to-multi-GB weights, exactly the duplication problem the
whole packaging design exists to avoid at the process level.

Fix: both your app AND its XPC service declare the SAME
`com.apple.security.application-groups` entry, following the convention
`<TeamID>.live.yooz.<app>.shared` (substitute your app's name for `<app>`;
this is scoped to your app + its own XPC service — it does NOT share weights
ACROSS different apps, e.g. Whisper and Crisp each get their own group.
Cross-app weight sharing is super-yooz's job, via the loopback transport).
`AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier:)`
(`Sources/EngineCore/AppGroupWeightsLocation.swift`) resolves the group's
shared container and points `HF_HUB_CACHE` at
`<container>/Library/Application Support/YoozEngine/huggingface/hub` —
Application Support, NOT Caches, because the OS purges container Caches
under disk pressure and would force a multi-GB re-download.

Two details worth knowing:

- The group id template lives in `XPCService/Info.plist`
  (`YoozAppGroupIdentifier` = `$(TeamIdentifierPrefix)live.yooz.<app>.shared`)
  rather than being computed at runtime — Xcode substitutes
  `$(TeamIdentifierPrefix)` at build/sign time, so the service reads back the
  fully-resolved string via `Bundle.main.object(forInfoDictionaryKey:)`
  without ever needing to look up its own team id in code.
- `redirectHuggingFaceCache` is a documented no-op (returns `false`, never
  crashes) when the group can't be resolved — no `application-groups`
  entitlement, no provisioning profile, or an unsandboxed dev run. The
  engine still works in that case, just without cross-process cache sharing;
  it falls through to `EngineConfig.huggingFaceCacheDirectory`'s own
  sandboxed-container fallback.

Known gap: `EngineConfig.modelsDirectory` (a handful of smaller LLM/STT
artifacts, distinct from the HF hub cache) is NOT yet app-group-aware — it
still resolves to each sandboxed process's own container. Tracked as a
follow-up; the HF hub cache above is where the large downloads land, so this
gap is small in practice.

### Verifying the round trip

This repo's `YoozEngineXPCHarness` target is a dev-only, non-shipped proof:
it embeds `YoozEngineXPC.xpc` under its own `Contents/XPCServices/` and
round-trips `GET /v1/health` plus an `openSTTStream`/`sendAudio`/`receive`/
`close` cycle through `XPCTransport`. It's a plain executable, not an XCTest
target — XPC services are launchd-managed with no GUI test runner involved,
so there's nothing for XCTest to attach to. Build and run it directly:

```bash
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngineXPCHarness \
  -configuration Debug -skipMacroValidation -derivedDataPath build build
"build/Build/Products/Debug/YoozEngineXPCHarness.app/Contents/MacOS/YoozEngineXPCHarness"
```

The harness forces the Apple STT backend before streaming (no download —
Parakeet, the default, needs a multi-hundred-MB HF pull the harness has no
business triggering just to prove wiring). On a machine where Speech
Recognition authorization for the harness/service hasn't been granted yet,
the streaming leg comes back as a typed `YoozEngineError` rather than
transcribing anything — that still proves the round trip (the request
crossed the XPC boundary, the service dispatched it, and a structured
failure returned instead of a hang), it just isn't a transcription. Grant
Speech Recognition access in System Settings first if you want the streaming
leg to actually transcribe.

## Variant selection

The engine ships in three variants. Pick based on what services your app needs:

| Variant | Modules | Bundle size | When to use |
|---|---|---|---|
| `YoozEngineLite` | Apple STT + Grammar + LLM | sub-GB | Apps that don't need MLX STT (e.g. Crisp, Remi). Smallest bundle. |
| `YoozEngineWhisper` | Lite + MLX STT (Parakeet, FastConformer) | ~600 MB | Apps that need high-accuracy non-Apple STT (Whisper, Notes). No VAD or Infinite module — caller handles VAD locally. |
| `YoozEngine` | All of the above + VAD + Infinite | ~600 MB + HF cache | The full engine host. Use for dev/standalone, super-yooz, VAD, or engine-hosted long-context sessions. |

**The standalone `Yooz Engine.app` (full variant, menu-bar UI) is dev-only and not shipped publicly.** The future is super-yooz hosting all of this; for now consumer apps embed their own variant.

## Building the helper bundle (loopback transport)

Applies to the loopback transport only — super-yooz host and local dev. App Store standalones use the in-process transport (see "Transport selection") and skip this section entirely; there is no helper `.app` to build or embed.

From the engine repo:

```bash
# Whisper variant (full minus VAD)
bash scripts/build-whisper-helper.sh
# Output: dist/YoozEngineWhisper.app  (signed ad-hoc by default)

# Lite variant (Apple STT only)
bash scripts/build-engine-lite.sh
# Output: dist/YoozEngineLite.app
```

Both scripts:
- Build via xcodegen + xcodebuild
- Sign ad-hoc (override with `YOOZ_SIGNING_IDENTITY=...` for Developer ID)
- Pass `-skipMacroValidation` (required from CLI for `MLXHuggingFaceMacros`)
- Land output in `dist/`

## Embedding into your host app (loopback transport)

In your host app's `project.yml`, add a post-build phase that copies the helper into `Contents/Helpers/`:

```yaml
postBuildScripts:
  - name: Embed Yooz Engine Helper
    inputFiles:
      - $(YOOZ_ENGINE_HELPER_PATH)
    outputFiles:
      - $(BUILT_PRODUCTS_DIR)/$(WRAPPER_NAME)/Contents/Helpers/Yooz Engine (Whisper).app
    script: |
      cp -R "$YOOZ_ENGINE_HELPER_PATH" "$BUILT_PRODUCTS_DIR/$WRAPPER_NAME/Contents/Helpers/"
```

See `scripts/build-whisper-helper.sh` (engine) and `scripts/embed-engine-helper.sh` + `scripts/build-with-engine.sh` (whisper) for a working reference.

## Lifecycle (loopback transport)

`InProcessTransport.connect()` just registers + bootstraps the linked module actors — no polling, no launch, no port. The rest of this section is loopback-specific.

The SDK's `connect()` does this:

1. Probe `/v1/health` on `127.0.0.1:19920`. If 200 → done, attach to existing engine.
2. If TCP refused → look for a bundled helper at `<host>.app/Contents/Helpers/Yooz Engine*.app`. If found, launch it via `NSWorkspace.openApplication(at:configuration:)` with `--headless` in `OpenConfiguration.arguments` and `YOOZ_ENGINE_HEADLESS=1` in `OpenConfiguration.environment` (see "Helper-mode signal"). Poll `/v1/health` for up to 10s.
3. If no bundled helper → look up `live.yooz.engine` via LaunchServices. Same launch path. (Legacy / dev-only.)
4. If TCP accepted but `/v1/health` doesn't respond → stale engine. Throws `YoozEngineError.portHeldByStaleEngine` unless `YOOZ_ENGINE_AUTO_RECOVER=1` is set, in which case the SDK kills the holder and relaunches.

The SDK is `Sendable`; you can safely use one `YoozEngineClient` for the lifetime of your app. Each service client (`stt`, `llm`, `touchUp`, `grammar`, `vad`, `infinite`) is a thin wrapper, cheap to construct.

## Available services

Transport-agnostic except where noted. Two different kinds of restriction apply, don't conflate them: `vad` is a **variant** restriction (only the full `YoozEngine` variant links `VADModule`; the in-process transport does route `/v1/vad/detect` when the module is linked), while `infinite` is a **transport** restriction (Infinite is loopback-only by design). The former `/v1/session/*` transport gap closed in engine#232 — sessions now dispatch through the shared endpoint table on every transport. See "Transport selection" above and "Build Variants" in `AGENTS.md`.

```swift
let client = YoozEngineClient()
try await client.connect()

// Health + introspection
let health = try await client.health()           // module readiness map
let modules = try await client.modules()         // build variant + per-module health

// STT
let langs = try await client.stt.languages()
let result = try await client.stt.batchTranscribe(audioSamples: [...])
let aligned = try await client.stt.batchTranscribeAligned(audioSamples: [...])
let stream = try await client.stt.startStream(language: .english)

// STT engine picker (canonical pattern)
let backends = try await client.stt.availableEngines()  // STTBackendsResponse
try await client.stt.setEngine(id: "apple_stt", preload: true)

// LLM
let text = try await client.llm.generate(prompt: "...", systemPrompt: "...")

// TouchUp (LLM-backed text cleanup pipeline)
let cleaned = await client.touchUp.touchUp(text: "...", mode: .standard)

// TouchUp model picker (canonical pattern)
let models = try await client.touchUp.availableModels()  // TouchUpModelsResponse
try await client.touchUp.setModel(id: "yooz-quality-v2", preload: true)

// Cross-module state snapshot + live events (engine#226) — see "Engine-owned
// selection state" below for the full `EngineStateStore` recipe.
let snapshot = try await client.engineState.snapshot()  // EngineStateSnapshot
let events = try await client.openEvents()              // AsyncStream<EngineEvent>
for await event in events { print(event.kind, event.module, event.modelId ?? "") }

// Grammar
let result = try await client.grammar.check(text: "...")

// VAD (only available in YoozEngine full variant)
let segments = try await client.vad.detect(samples: [...])

// Infinite long-context sessions (only available in YoozEngine full variant)
let infiniteModels = try await client.infinite.availableModels()
_ = try await client.infinite.status()
let session = try await client.infinite.createSession(
    modelId: infiniteModels.activeId,
    label: "analysis-context"
)
try await client.infinite.append(sessionId: session.id, text: longDocumentText)
try await client.infinite.checkpoint(sessionId: session.id, label: "loaded")
try await client.infinite.deleteSession(id: session.id)
```

## The canonical model picker pattern

Every module that exposes a model selection follows the same shape:

```
GET  /v1/<module>/models   → ModelsResponse { models: [ModelInfo], activeId: String }
POST /v1/<module>/model    body { id: String, preload: Bool? } → ModelInfo
```

`ModelInfo` carries `{ id, displayName, description, tier, sizeBytes, loadState, isActive }`:
- `tier`: `.light` / `.quality` / `.premium` / `.unknown` (forward-compat fallback)
- `loadState`: `.unavailable` < `.available` < `.cached` < `.loaded` (total ordering)
- `isActive`: exactly one row has `isActive == true`

Full field/route pattern documented in `AGENTS.md` → "Module model picker pattern".

### Engine-owned selection state (engine#226) — use this, not a hand-rolled store

**Do not build your own persistence or polling for a model picker.** The
engine is the source of truth for "which model is selected" and "is it
loaded yet" — it remembers the selection across restarts and pushes every
state transition. The previous recipe here ("build a `ModelPickerStore<T>`
that polls and persists to `SettingsManager`") is what every consumer app
had to reinvent, bugs included (see the engine#226 issue body: whisper's
`LLMModelPickerStore` reconciliation latches, its three-level STT rollback
ladder, `awaitModelReady` polling that timed out on multi-GB downloads).
None of that is necessary anymore for a module that has adopted the
engine-owned-selection contract (TouchUp today):

```swift
import YoozEngineClient

@MainActor
final class MyPickerViewModel: ObservableObject {
    let state: EngineStateStore

    init(client: YoozEngineClient) {
        state = EngineStateStore(client: client)
    }

    func onAppear() async {
        await state.start()   // GET /v1/state, then subscribe to /v1/events
    }

    func onDisappear() {
        state.stop()          // cancel the /v1/events subscription
    }
}
```

```swift
// SwiftUI
struct TouchUpPickerView: View {
    @ObservedObject var state: EngineStateStore

    var body: some View {
        if let touchUp = state.modules["touchup"] {
            ForEach(touchUp.models, id: \.id) { row in
                Button(row.displayName) {
                    Task { try? await client.touchUp.setModel(id: row.id) }
                }
                .disabled(row.loadState == .unavailable)
                if row.isActive { Image(systemName: "checkmark") }
            }
            if let progress = state.latestEvent(module: "touchup", kind: .downloadProgress)?.progress {
                ProgressView(value: progress)
            }
        }
    }
}
```

That view has **zero local persistence** (no `SettingsManager` key — the
engine remembers `touchup`'s active model across restarts via
`ModelSelectionStore`) and **zero polling** (no `awaitModelReady` loop —
`/v1/events` pushes `modelChanged` / `loadStateChanged` / `downloadProgress`
/ `residencyChanged` as they happen). `POST /v1/<module>/model` itself
returns immediately regardless of `preload` — it never blocks on a
download; the caller learns the outcome from the event stream, not from
the response.

`EngineStateStore` is transport-agnostic — `client.openEvents()` resolves
through whichever `EngineTransport` the client was built with. On loopback,
in-process, and now XPC (engine#244) it just works: `XPCTransport.openEvents()`
bridges the push channel over the same callback-proxy shape streaming STT
already uses (`openEvents`/`closeEvents` on the XPC protocol, frames pushed
to the client's exported callback object), one `EngineEventBus` subscription
per `openEvents()` call — in practice one per client connection, since
`EngineStateStore` opens exactly one, but the wire protocol supports N
concurrent subscriptions keyed by client-generated id.

One XPC-specific contract to know: the returned `AsyncStream<EngineEvent>`
**finishes** (stops producing frames, with no error to inspect — plain
`AsyncStream` has no `Failure` channel) when the underlying `NSXPCConnection`
is interrupted or invalidated, rather than silently going quiet forever.
`EngineStateStore.subscribeToEvents()`'s `for await` loop exits at that
point without setting `lastError` (a clean finish isn't a fetch failure), so
a consumer that wants live events across an XPC service crash/respawn must
notice the subscription ended and call `state.start()` again — `start()` is
documented as safe to call repeatedly for exactly this reason. In practice
that means wiring the re-subscribe off whatever reconnect signal your
transport wrapper exposes (e.g. yooz-whisper's `ReconnectingXPCTransport.onReconnect`)
rather than assuming the store's live feed survives a service respawn on
its own.

`GET /v1/state` and `/v1/events` are cross-module: `EngineStateStore.modules`
is keyed by module name (`"touchup"` today), so the same store instance
serves every module's picker in your app. `EngineModelSnapshotRow` only
carries the seven canonical `ModelInfo` fields — a module with extension
fields (e.g. STT's `supportsBatch` / `supportsStreaming` /
`supportedLanguages`) still calls its own `GET /v1/<module>/models` for the
rich rows; `EngineStateStore` is the right layer for "is a load in flight /
what's the progress", not a replacement for a module-specific richer type.
The existing `ModelPickerStore<T>` pattern (`yooz-whisper/YoozWhisper/UI/LLMModelPickerStore.swift`)
remains valid for exactly that case — layer it so its refresh/progress
signals come from `EngineStateStore.latestEvent(module:kind:)` instead of a
poll loop, rather than reinventing persistence.

Adoption status: TouchUp is the reference implementation of the full
contract (persisted selection + non-blocking `setModel` + events). STT
engine and Infinite have not adopted persisted selection / async
`setModel` yet — their pickers still behave per the pre-#226 contract
(process-lifetime selection only, and `POST /v1/stt/engine` /
`POST /v1/infinite/model` may still block on a load). Whisper's
reconciler deletion (dropping `LLMModelPickerStore`'s latches, the STT
rollback ladder) is tracked as a whisper-side follow-up issue, gated on
this section's contract actually shipping.

## Infinite long-context module

Infinite follows the engine substrate rule: the engine owns the API surface, session lifecycle, RAM gating, picker catalogue, and cleanup policy; the Infinite project lends backend modules and benchmark evidence. Consumer apps should integrate through `YoozEngineClient.infinite` or `/v1/infinite/*`, not by calling Infinite repo code directly.

Use the full `YoozEngine` variant for Infinite. Lite and Whisper variants do not bundle it and return the standard module-not-bundled `501`. Infinite is also loopback-only by transport, independent of variant: `InProcessTransport` and (once packaged) `XPCTransport` both report `unsupportedOperation` for `/v1/infinite/*`, because Infinite's designed consumer is the loopback host (super-yooz).

| Capability | Contract |
|---|---|
| Model picker | `GET /v1/infinite/models`, `POST /v1/infinite/model` |
| Status | `GET /v1/infinite/status` |
| Sessions | `GET/POST /v1/infinite/sessions`, `GET/DELETE /v1/infinite/sessions/:id` |
| Context append | `POST /v1/infinite/sessions/:id/append` |
| Checkpoints | `POST /v1/infinite/sessions/:id/checkpoint` |
| Generation | `POST /v1/infinite/sessions/:id/generate`; generates on Swift-runtime-supported models (Qwen3.6 `qwen3_5_moe`, Gemma4 26B-A4B and E4B — #184/#186; native window ≤262K). Only retrieval returns `501 generation_unavailable` |

RAM gating is part of the contract. Hosts below 32 GiB cannot select Infinite models. A 32-63 GiB host is the reduced tier and can select `gemma4-e4b-1m`; 64 GiB+ is the full tier and can select the full catalogue. Apps should render `loadState == .unavailable` rows as visible but disabled so users understand the hardware boundary.

Infinite sessions are engine-owned resources. They survive `/v1/session/begin` recording boundaries and are cleaned up only by explicit delete or process exit:

```text
explicit_delete_or_process_exit;max_active_sessions=16
```

Always delete sessions when finished. Leaking sessions blocks capacity for other engine clients.

The full API, model catalogue, RAM tiers, evidence references, and verification commands are documented in [`INFINITE_MODULE.md`](INFINITE_MODULE.md).

## Error handling

`YoozEngineError` distinguishes:

| Case | Transport-level (gate-poisoning candidate) | App-level |
|---|---|---|
| `engineNotInstalled` | ✓ | |
| `engineNotReachable` | ✓ | |
| `engineLaunchFailed` | ✓ | |
| `portHeldByStaleEngine` | ✓ | |
| `webSocketError` | ✓ | |
| `serverError(statusCode:code:message:)` | | ✓ |
| `httpError(statusCode:)` | | ✓ |
| `invalidResponse` | | ✓ |
| `decodingError` | | ✓ |

If your client wraps the SDK with a `ConnectionGate` (whisper does), only re-handshake on transport failures. App-level errors (a 400 `invalid_model`, a decode mismatch) are not reasons to reset the gate.

## HF model auto-download

STT, LLM, TouchUp, and Infinite model artifacts are pulled from HuggingFace on first use. Cache lands at `~/.cache/huggingface/hub/`. There is no embedded / GHCR fallback — the engine assumes network on first use. Subsequent launches use the cached snapshot.

For STT, `POST /v1/stt/load` accepts `allow_fetch: false` to fail fast on a cold cache. The engine surfaces download progress via `/v1/stt/status.progress` (a Double 0.0–1.0).

## Streaming STT partial cadence

The `/v1/stt/stream` WebSocket emits partial transcriptions at a tunable cadence. The engine accumulates audio across frames and only re-runs the encoder + decoder once every `YOOZ_STT_PARTIAL_INTERVAL_SEC` seconds of new audio (default `2.0`). This keeps the per-frame fast path cheap when the caller feeds many small (~64 ms) audio buffers, while still giving the user visible progress every couple of seconds.

| Setting | Behaviour |
|---|---|
| `2.0` (default) | Balanced — partial roughly every two seconds. |
| `0.5`–`1.0` | More visual feedback, more encode cycles. Suitable for quiet desktops where the user wants tight live preview. |
| `5.0`+ | Less CPU, less feedback. Closer to "batch on chunk boundary" behaviour. |
| `0` | Disable the throttle entirely — re-encode on every WebSocket frame. Useful for instrumentation only. |

Set it in your helper-launch environment (alongside `YOOZ_ENGINE_HEADLESS` and `YOOZ_ENGINE_PORT`):

```swift
config.environment = [
    "YOOZ_ENGINE_HEADLESS": "1",
    "YOOZ_ENGINE_PORT": "19921",
    "YOOZ_STT_PARTIAL_INTERVAL_SEC": "1.5",   // tune to taste
]
```

The compiled default lives at `EngineConfig.defaultStreamingPartialIntervalSec`; the resolved (env-aware) value at `EngineConfig.streamingPartialIntervalSec`. Hosts that construct `StreamingTranscriber` directly can pass `partialEmissionInterval:` for per-session overrides.

## Per-app port isolation

Free standalone apps coexist on a user's machine. Each app's embedded helper binds a different loopback port so they don't collide. The SDK constructor takes the port; the helper reads `YOOZ_ENGINE_PORT` from its environment.

| App | Port |
|---|---|
| `yooz-whisper` | 19920 (legacy default) |
| `yooz-crisp` | 19921 |
| `yooz-notes` | 19922 |
| `yooz-voice` | 19923 |
| `remi` | 19924 |
| `super-yooz` | 19920 (when running as host) |

When integrating a new app, pick the next free port and document it in your app's `AGENTS.md`. Configure your SDK + helper-launch path to use that port:

```swift
// Pick your app's port
let client = YoozEngineClient(host: "127.0.0.1", port: 19921)  // crisp
```

```swift
// Helper launch — populate BOTH channels for reliable headless mode.
// The SDK's `helperOpenConfiguration` already does this for you; the
// snippet below is only relevant if you build your own launcher
// (e.g. yooz-whisper's `EngineHelperController`).
config.arguments = ["--headless"]                          // reliable on macOS 26
config.environment = [
    "YOOZ_ENGINE_HEADLESS": "1",                          // backward compat
    "YOOZ_ENGINE_PORT": "19921",
]
```

Helpers are loopback-only by design (`127.0.0.1`). CORS / origin checks ensure browser pages can't poke a local helper from a malicious tab.

## Helper-mode signal (env + argv)

The engine enters headless mode (no menu-bar icon, no Settings scene) when EITHER of two signals is present:

1. **`--headless` command-line argument** (reliable on macOS 26). Passed through `NSWorkspace.OpenConfiguration.arguments`; LaunchServices propagates this to the spawned helper.
2. **`YOOZ_ENGINE_HEADLESS=1` environment variable** (backward compat). Passed through `NSWorkspace.OpenConfiguration.environment` — but LaunchServices does NOT reliably propagate the env dict to nested helper bundles on macOS 26 (engine#117 / whisper#179). Kept for direct shell exec, scripts, and test harnesses.

The SDK's `helperOpenConfiguration` populates both for belt-and-suspenders. The engine's `EngineConfig.isHelperMode(environment:arguments:)` ORs the two channels, so either alone is sufficient. If you build a custom launcher (instead of using the SDK auto-launch path), make sure you set BOTH on `OpenConfiguration` to stay safe on macOS 26.

## Two co-existing distribution models

**Free standalone (today, the primary distribution):**
- Each app is free on App Store, including Pro tier
- Premium tier (optional, one-time per app) only when a feature requires dedicated dev investment
- App embeds its own helper variant — port-isolated, CORS-locked, no cross-app talk
- This is the primary acquisition + trust-building distribution

**super-yooz (paid integration layer, ongoing):**
- Raycast-style host bundles all modules in one shell
- Centralizes models so users don't pay 3-5× disk
- Cross-module workflows (dictate → cleanup → grammar → paste in one keystroke)
- One-time purchase

Both models use the same engine substrate. Design new consumer apps to be complete + valuable as a free standalone first; super-yooz integration is a future migration path that the SDK auto-discovery handles transparently.

## Reference apps

- **yooz-whisper** — the canonical in-process reference. Links `YoozEngineInProcess` (`YoozEngineWhisper` module set: STT + AppleSTT + Grammar + LLM, no VAD/Infinite), pinned by revision in its `project.yml`, drives the LLM + STT pickers, ships an Engine settings tab. See `yooz-whisper/AGENTS.md`.
- **super-yooz** — the canonical loopback reference (not yet implemented as of 2026-07; zero code against the engine substrate). Host-app charter at `super-yooz/.context/charter.md`.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Two engine icons in menu bar (loopback only) | Both your `EngineHelperController.launch()` and `YoozEngineClient.connect()` racing the launch | Just use `client.connect()` — the SDK handles bundled-helper detection + headless launch. Drop the manual `EngineHelperController.launch()` once you're not on legacy v0.5 SDK. |
| Engine starts but shows "port in use" alert (loopback only) | A stale engine instance is holding 19920 | Set `YOOZ_ENGINE_AUTO_RECOVER=1` for dev (SDK will SIGKILL the holder). For ship, surface the error to the user. |
| MLXHuggingFaceMacros build failure from CLI | Macro requires explicit trust on first use | Pass `-skipMacroValidation` to xcodebuild. Xcode UI handles this prompt automatically; CI / scripts must pass the flag. |
| `connect()` succeeds but service calls 501 | Active build variant doesn't bundle that module (e.g. Lite has no MLX STT), OR (in-process only) the route isn't implemented by `InProcessTransport` yet | Check `client.modules()` — `unavailable` modules return 501 by design. For an in-process-only gap (not module-not-bundled), see the `/v1/session/*` and Infinite rows below. |
| Infinite calls throw `unsupportedOperation` (in-process/XPC) | Infinite is loopback-only by design — its consumer is the loopback host (super-yooz), not the in-process/XPC path | Don't wire Infinite into an in-process/XPC-packaged standalone. If you need it, use the loopback transport. |
| Infinite `generate` returns 501 (loopback) | The active model has no runnable Swift MLX backend (only retrieval today); session state is preserved | Switch to a Swift-runtime-supported model (Qwen3.6 `qwen3_5_moe`, Gemma4 26B-A4B or E4B), or branch on the stable `generation_unavailable` code. Create/append/checkpoint/delete keep working regardless. |
| Infinite models are visible but disabled | Host RAM tier cannot run that row | Use `loadState == .unavailable`, `ramTier`, and `maxContextTokens` from `/v1/infinite/models` to explain the requirement. |
| Infinite session creation starts failing after repeated tests | Sessions are engine-owned and capped at 16 | Delete sessions explicitly with `client.infinite.deleteSession(id:)`; `/v1/session/begin` does not clean them up. |
| Building against `XPCTransport` produces no bundle to test | Your own app hasn't packaged a `.xpc` service target yet | Not a gap in the engine — `YoozEngineXPC` in this repo is the reference implementation (engine#227). Copy its shape ("XPC service embed recipe" above) into your app's `project.yml`. |
| `.xpc` service crash-loops in Release only (`dyld: Library not loaded: @rpath/MLX.framework/...`), works fine in Debug | Debug is masked by an absolute, machine-local DerivedData `LC_RPATH`; Release has none. SPM dynamic package frameworks were never embedded into the `.xpc`'s own `Contents/Frameworks` (engine#248) | Add the `postbuildScripts` embed phase from "XPC service embed recipe" step 1 to your own `.xpc` target. |
| Embedded `.xpc` fails `codesign --verify` ("a sealed resource is missing or invalid") only after being embedded into the host app, even though it verified fine standalone | Xcode's embed-into-app step strips `Modules/*.swiftmodule` from nested frameworks without re-signing them (engine#248) | Add the `postbuildScripts` re-sign phase from "XPC service embed recipe" step 3 to your app target, AFTER the embed Copy Files phase. |
| XPC service builds but weights re-download per process | App and XPC service don't share an `application-groups` entitlement, or `AppGroupWeightsLocation.redirectHuggingFaceCache` wasn't called before the first HF fetch | See "Weights/app-group wiring" above. Both processes must declare the SAME group id; call the redirect at the very top of `main.swift`. |
| `EngineStateStore`'s picker UI silently stops updating after an XPC service crash/respawn | `openEvents()`'s `AsyncStream` finished (connection interruption/invalidation) — that's the documented contract, not a bug — but nothing re-subscribed | Detect the finish (the `for await` loop in `subscribeToEvents()` returning) and call `state.start()` again, typically off your transport wrapper's own reconnect signal (e.g. `ReconnectingXPCTransport.onReconnect` in yooz-whisper). See "Engine-owned selection state" above. |

---

*See also: AGENTS.md "Module model picker pattern", docs/INFINITE_MODULE.md, and scripts/build-*.sh for the variant build pipeline.*
