# Consumer Integration Guide

How to integrate Yooz Engine v0.6.0+ into a Swift/SwiftUI app.

## TL;DR

```swift
// Package.swift / project.yml
.package(url: "https://github.com/yooz-labs/yooz-engine", from: "0.6.0")

// Anywhere in your app
import YoozEngineClient
let client = YoozEngineClient()
try await client.connect()
let result = await client.touchUp.touchUp(text: "um yeah", mode: .standard)
```

That's the happy path. The SDK auto-discovers a bundled helper under `<host>.app/Contents/Helpers/` (preferred), or falls back to a LaunchServices lookup for a standalone Yooz Engine.app (legacy). It launches the helper headless (`--headless` argv flag plus `YOOZ_ENGINE_HEADLESS=1` env var, see "Helper-mode signal" below) so your host app's menu-bar UI stays clean.

## Variant selection

The engine ships in three variants. Pick based on what services your app needs:

| Variant | Modules | Bundle size | When to use |
|---|---|---|---|
| `YoozEngineLite` | Apple STT + Grammar + LLM | sub-GB | Apps that don't need MLX STT (e.g. Crisp, Remi). Smallest bundle. |
| `YoozEngineWhisper` | Lite + MLX STT (Parakeet, FastConformer) | ~600 MB | Apps that need high-accuracy non-Apple STT (Whisper, Notes). No VAD or Infinite module — caller handles VAD locally. |
| `YoozEngine` | All of the above + VAD + Infinite | ~600 MB + HF cache | The full engine host. Use for dev/standalone, super-yooz, VAD, or engine-hosted long-context sessions. |

**The standalone `Yooz Engine.app` (full variant, menu-bar UI) is dev-only and not shipped publicly.** The future is super-yooz hosting all of this; for now consumer apps embed their own variant.

## Building the helper bundle

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

## Embedding into your host app

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

## Lifecycle

The SDK's `connect()` does this:

1. Probe `/v1/health` on `127.0.0.1:19920`. If 200 → done, attach to existing engine.
2. If TCP refused → look for a bundled helper at `<host>.app/Contents/Helpers/Yooz Engine*.app`. If found, launch it via `NSWorkspace.openApplication(at:configuration:)` with `--headless` in `OpenConfiguration.arguments` and `YOOZ_ENGINE_HEADLESS=1` in `OpenConfiguration.environment` (see "Helper-mode signal"). Poll `/v1/health` for up to 10s.
3. If no bundled helper → look up `live.yooz.engine` via LaunchServices. Same launch path. (Legacy / dev-only.)
4. If TCP accepted but `/v1/health` doesn't respond → stale engine. Throws `YoozEngineError.portHeldByStaleEngine` unless `YOOZ_ENGINE_AUTO_RECOVER=1` is set, in which case the SDK kills the holder and relaunches.

The SDK is `Sendable`; you can safely use one `YoozEngineClient` for the lifetime of your app. Each service client (`stt`, `llm`, `touchUp`, `grammar`, `vad`, `infinite`) is a thin wrapper, cheap to construct.

## Available services (v0.6.0)

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

Build a generic `ModelPickerStore<T>` once; reuse across TouchUp, STT engine, Infinite, and future TTS voices. See `yooz-whisper/YoozWhisper/UI/LLMModelPickerStore.swift` for the reference implementation.

Full pattern documented in `AGENTS.md` → "Module model picker pattern".

## Infinite long-context module

Infinite follows the engine substrate rule: the engine owns the API surface, session lifecycle, RAM gating, picker catalogue, and cleanup policy; the Infinite project lends backend modules and benchmark evidence. Consumer apps should integrate through `YoozEngineClient.infinite` or `/v1/infinite/*`, not by calling Infinite repo code directly.

Use the full `YoozEngine` variant for Infinite. Lite and Whisper variants do not bundle it and return the standard module-not-bundled `501`.

| Capability | Contract |
|---|---|
| Model picker | `GET /v1/infinite/models`, `POST /v1/infinite/model` |
| Status | `GET /v1/infinite/status` |
| Sessions | `GET/POST /v1/infinite/sessions`, `GET/DELETE /v1/infinite/sessions/:id` |
| Context append | `POST /v1/infinite/sessions/:id/append` |
| Checkpoints | `POST /v1/infinite/sessions/:id/checkpoint` |
| Generation | `POST /v1/infinite/sessions/:id/generate`; currently returns `501 generation_unavailable` until backend inference is wired |

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

- **yooz-whisper** — the canonical reference. Embeds `YoozEngineWhisper`, drives the LLM + STT pickers, ships an Engine settings tab. See `yooz-whisper/AGENTS.md` after PR #170 lands.
- **super-yooz** — host-app charter at `super-yooz/.context/charter.md`.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Two engine icons in menu bar | Both your `EngineHelperController.launch()` and `YoozEngineClient.connect()` racing the launch | Just use `client.connect()` — the SDK handles bundled-helper detection + headless launch. Drop the manual `EngineHelperController.launch()` once you're not on legacy v0.5 SDK. |
| Engine starts but shows "port in use" alert | A stale engine instance is holding 19920 | Set `YOOZ_ENGINE_AUTO_RECOVER=1` for dev (SDK will SIGKILL the holder). For ship, surface the error to the user. |
| MLXHuggingFaceMacros build failure from CLI | Macro requires explicit trust on first use | Pass `-skipMacroValidation` to xcodebuild. Xcode UI handles this prompt automatically; CI / scripts must pass the flag. |
| `connect()` succeeds but service calls 501 | Active build variant doesn't bundle that module (e.g. Lite has no MLX STT) | Check `client.modules()` — `unavailable` modules return 501 by design. Render the limitation in your UI. |
| Infinite `generate` returns 501 | Backend inference is not wired yet; session state is preserved | Treat `generation_unavailable` as the current app-level boundary, not a transport failure. Keep using create/append/checkpoint/delete. |
| Infinite models are visible but disabled | Host RAM tier cannot run that row | Use `loadState == .unavailable`, `ramTier`, and `maxContextTokens` from `/v1/infinite/models` to explain the requirement. |
| Infinite session creation starts failing after repeated tests | Sessions are engine-owned and capped at 16 | Delete sessions explicitly with `client.infinite.deleteSession(id:)`; `/v1/session/begin` does not clean them up. |

---

*See also: AGENTS.md "Module model picker pattern", docs/INFINITE_MODULE.md, and scripts/build-*.sh for the variant build pipeline.*
