# Infinite Module API

`InfiniteModule` is the engine-hosted long-context service. The engine owns the loopback API, picker catalogue, RAM gating, sessions, and cleanup policy. The Infinite research repo lends model evidence, backend adapter modules, and benchmark provenance.

Infinite is bundled only in the full `YoozEngine` variant. `YoozEngineLite` and `YoozEngineWhisper` return the standard module-not-bundled `501` response for `/v1/infinite/*`.

## Availability

| Host memory | `ramTier` | Behaviour |
|---|---|---|
| `< 32 GiB` | `below_minimum` | Infinite models are unavailable. |
| `32-63 GiB` | `reduced` | Reduced-tier Gemma model is selectable. Full-tier rows remain visible with `loadState: "unavailable"`. |
| `64 GiB+` | `full` | Full catalogue is selectable. |

All Infinite rows require Apple Silicon. Model weights use the normal HuggingFace cache path (`~/.cache/huggingface/hub/`) and are never committed to the repository.

Whether the module is bundled and loaded is reported by `/v1/health` as `modules.infinite` (and in full per-module detail via `/v1/modules`). It is `false` on the Lite/Whisper variants (module not bundled) and until a model is loaded.

## Model Catalogue

The canonical catalogue is `InfiniteModelSelection` in the engine. Consumer apps should read `/v1/infinite/models` rather than duplicating this table.

`Native` is the model's trained attention window; `Max` is the memory-feasible ceiling via paged cache + position extension. The `Max` figure is single-needle-validated; multi-hop accuracy degrades well below it and the **interactive tier is ~256K** (1M is latency-bound). Never quote `Max` as a demonstrated end-to-end capability without the harness caveats (`infinite:research/18,26,27`). For `s3-retrieval`, `Max` is retrieval **index capacity**, not an attention window.

| ID | Display name | Tier | Backend | Adapter | Native | Max | RAM | Evidence |
|---|---|---|---|---|---:|---:|---|---|
| `gemma4-e4b-1m` | Gemma4 E4B 1M | `light` | `paged-kv` | `infinite-paged-kv-mlx-v1` | 131,072 | 1,000,000 | `reduced` | `infinite:research/18-gemma-support-matrix.md` |
| `gemma4-26b-a4b-1m` | Gemma4 26B-A4B 1M | `quality` | `paged-kv` | `infinite-paged-kv-mlx-v1` | 262,144 | 1,000,000 | `full` | `infinite:research/18-gemma-support-matrix.md` |
| `qwen3-35b-1m` | Qwen3.6 35B-A3B 1M | `premium` | `paged-kv` | `infinite-paged-kv-mlx-v1` | 262,144 | 1,000,000 | `full` | `infinite:research/26-flagship-1m.md` |
| `s3-retrieval` | S3 Retrieval | `quality` | `retrieval` | `infinite-retrieval-index-v1` | n/a | 10,000,000 (index) | `full` | `infinite:research/24-dense-retrieval.md` |

## Endpoints

| Method | Path | Response | Notes |
|---|---|---|---|
| `GET` | `/v1/infinite/models` | `InfiniteModelsResponse` | Model picker rows plus `activeId`. |
| `POST` | `/v1/infinite/model` | `InfiniteModelInfo` | Body: `{ "id": String, "preload": Bool? }`. |
| `GET` | `/v1/infinite/status` | `InfiniteStatus` | Active model, load state, active session count, cleanup policy, and resources. |
| `GET` | `/v1/infinite/sessions` | `InfiniteSessionsResponse` | Lists engine-owned long-context sessions. |
| `POST` | `/v1/infinite/sessions` | `InfiniteSessionInfo` | Body: `{ "modelId": String?, "label": String? }`. |
| `GET` | `/v1/infinite/sessions/:id` | `InfiniteSessionInfo` | Returns `404` when the session is gone. |
| `POST` | `/v1/infinite/sessions/:id/append` | `InfiniteAppendSessionResponse` | Body: `{ "text": String }`; empty text is invalid. |
| `POST` | `/v1/infinite/sessions/:id/checkpoint` | `InfiniteCheckpointSessionResponse` | Body: `{ "label": String?, "park": Bool? }`. Persists the session's KV cache to disk (`InfiniteSessionStore`); `park: true` also releases it from RAM (`state` becomes `"parked"`). Response adds `sizeBytes`, `tokenCount`, `durationSeconds`, and `parentCheckpointId` alongside `session`/`checkpoint`. |
| `POST` | `/v1/infinite/sessions/:id/resume` | `InfiniteSessionInfo` | Body: `{ "checkpointId": String? }` (defaults to the latest checkpoint). Integrity-verifies the checkpoint and reloads its KV cache. A no-op success on an already-`"open"` session when `checkpointId` is omitted. Works even after a process restart — the checkpoint's own manifest is enough to rehydrate a minimal session record. |
| `POST` | `/v1/infinite/sessions/:id/fork` | `InfiniteSessionInfo` | Body: `{ "checkpointId": String?, "label": String? }` (defaults to the source session's latest checkpoint). Clones the checkpoint into a **new** session id (respects the 16-session cap) and does **not** auto-resume it — the fork starts `"parked"`. A hot, never-checkpointed source takes one implicit checkpoint first. |
| `POST` | `/v1/infinite/sessions/:id/generate` | `InfiniteGenerateSessionResponse` | Generates from the session's accumulated context on Swift-runtime-supported models (Qwen3.6 `qwen3_5_moe`, Gemma4 26B-A4B, and Gemma4 E4B — #184/#186), bounded to the model's native window (≤262K; 1M paging tracked in #180). Only retrieval mode returns `501 generation_unavailable`. Session state is preserved either way. Transparently resumes a `"parked"` session first. |
| `DELETE` | `/v1/infinite/sessions/:id` | `InfiniteDeleteSessionResponse` | Releases the engine-owned session and deletes its on-disk checkpoint tree. |

The cleanup policy is:

```text
explicit_delete_or_process_exit;max_active_sessions=16
```

`/v1/session/begin` is a recording boundary for STT-style work and does not delete Infinite sessions. Consumers must call `DELETE /v1/infinite/sessions/:id` when finished.

## Session Lifecycle (engine#266)

`InfiniteSessionInfo.state` is one of:

| State | Meaning |
|---|---|
| `open` | Backend-resident (or never yet touched) and idle — the normal target for append/generate/checkpoint. |
| `parked` | Checkpointed to disk with its live KV cache released from RAM. `append`/`generate`/`checkpoint` transparently resume it first — callers do not need to call `resume` themselves except to target a specific (non-latest) checkpoint. |
| `generating` | A call is in flight for this session; any other op on it returns `409 session_busy` rather than racing the backend. |

At most **2 sessions stay hot** (hold a live KV cache) at once, independent of the 16-session tracked-record cap: opening or resuming a 3rd hot session automatically checkpoints + parks the least-recently-touched other hot session first. Switching the active model (a different session's append/generate loading a different backend) parks every session hot under the outgoing model the same way. A session that is currently `generating` is never parked mid-flight: hot-budget eviction picks another victim (or returns `409 session_busy` if every candidate is busy), and a model switch that would orphan a generating session's backend is refused outright with `409 session_busy`; retry after the in-flight call completes.

Checkpoints are durable across process exit, not just RAM eviction: `resume`/`fork` read a checkpoint's own on-disk manifest, so they work even against a session id the current engine process has never seen (e.g. after a restart), as long as its checkpoint directory still exists.

### Error codes

Alongside the existing `invalid_model` (400), `model_unavailable` (501), `model_set_failed` (500), `session_not_found` (404), `invalid_session_input` (400), `session_limit_exceeded` (409), `generation_unavailable` (501), and `generation_failed` (500):

| Code | Status | Meaning |
|---|---|---|
| `checkpoint_not_found` | 404 | The requested (or implied) checkpoint id does not exist for this session. |
| `checkpoint_integrity` | 409 | The checkpoint's on-disk contents failed the schema/model/tokenizer/token-hash integrity gate (`InfiniteSessionStore.verify`) — e.g. a tampered `tokens.bin`, or a resume against a different model/tokenizer than the checkpoint was written with. The session is left untouched; no live state is installed from a failed integrity check. |
| `session_busy` | 409 | The session is currently `generating`; retry once the in-flight call completes. |

## SDK Usage

```swift
import YoozEngineClient

let client = YoozEngineClient()
try await client.connect()

let models = try await client.infinite.availableModels()
let active = try await client.infinite.setModel(
    id: models.activeId,
    preload: false
)

_ = try await client.infinite.status()
let session = try await client.infinite.createSession(
    modelId: active.id,
    label: "draft-context"
)

try await client.infinite.append(
    sessionId: session.id,
    text: longDocumentText
)
try await client.infinite.checkpoint(sessionId: session.id, label: "loaded")
```

Park a session to free RAM, then resume it later — `append`/`generate` do this
transparently, but an explicit `resume` is also available (e.g. to target a
non-latest checkpoint):

```swift
let checkpoint = try await client.infinite.checkpoint(
    sessionId: session.id, label: "before-park", park: true
)
// session.state is now "parked"; append/generate would resume it
// automatically, or resume explicitly:
try await client.infinite.resumeSession(sessionId: session.id)

// Fork a checkpoint into a brand-new, independent session — the fork
// starts "parked" (it does not auto-resume):
let forked = try await client.infinite.forkSession(
    sessionId: session.id,
    checkpointId: checkpoint.checkpoint.id,
    label: "exploration-branch"
)
try await client.infinite.resumeSession(sessionId: forked.id)
```

Generation runs on Swift-runtime-supported models (Qwen3.6 `qwen3_5_moe`, Gemma4 26B-A4B and E4B — #184/#186). Only retrieval mode throws `generation_unavailable`:

```swift
do {
    let reply = try await client.infinite.generate(
        sessionId: session.id,
        prompt: "Summarize the loaded context",
        maxTokens: 256
    )
    print(reply.text, reply.finishReason)
    print(reply.resources.decodeTokensPerSecond ?? 0) // measured decode throughput
} catch YoozEngineError.serverError(_, let code, _) where code == "generation_unavailable" {
    // The active model has no runnable Swift MLX backend (only retrieval today).
    // Switch to a Swift-runtime-supported model, or branch on the stable `code`.
}
```

When finished, release the engine-owned session:

```swift
try await client.infinite.deleteSession(id: session.id)
```

## Status and Resource Fields

`InfiniteStatus.resources` and every session resource snapshot include:

| Field | Meaning |
|---|---|
| `physicalMemoryBytes` | Host physical memory. |
| `wiredMemoryLimitBytes` | Process-visible wired memory ceiling used by the module. |
| `requiredRAMTier` | RAM tier required by the active model or session model. |
| `peakMemoryBytes` | Reserved for measured peak memory. |
| `prefillTokensPerSecond` | Reserved for measured prefill throughput. |
| `decodeTokensPerSecond` | Measured decode throughput from the last generation (populated on Swift-runtime-supported models; decode-only, from the engine's own generation stats). |
| `draftAcceptanceRate` | Reserved for speculative decode runs. |

`decodeTokensPerSecond` is populated from real runs; `peakMemoryBytes`, `prefillTokensPerSecond`, and `draftAcceptanceRate` remain reserved. Infinite research claims must still include the machine spec, peak memory, accuracy, and throughput evidence from the shared harness, not just this single live number.

## Verification

Fast SDK wire-shape check:

```bash
swift test --filter InfiniteTypesTests
```

Xcode graph compile check for `InfiniteModule`, the full app, and the route
test bundles:

```bash
xcodegen generate
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation \
  -derivedDataPath .build/DerivedData \
  build-for-testing
```

Consumer-style route proof:

```bash
scripts/run-integration.sh
```

The integration suite starts a served engine and drives Infinite through `YoozEngineClient`, including `/v1/modules`, model picker, status, create, append, fetch, checkpoint, generation (real text from the default Gemma4 E4B model now that #186 landed; retrieval still returns `501 generation_unavailable`), delete, and deleted-session `404`.

Live checkpoint/park/resume/fork gate (`Tests/InfiniteModuleTests/InfiniteCheckpointResumeLiveTests.swift`, engine#266): token-exact greedy equivalence across checkpoint(park)+resume, fork divergence with parent-checkpoint immutability, and the integrity gate rejecting a tampered `tokens.bin`, against the pinned `mlx-community/Qwen3.5-0.8B-MLX-4bit` (hybrid `MambaCache`/`KVCacheSimple` — the hardest KV-cache-shape case). Needs the `.full` (64 GiB) RAM tier — `installBackendForTesting`'s test-only label must match the injected backend's own `.selection` (`.qwen35B1M`), since `loadBackend`'s cache-hit check keys off that identity, not a separately-tracked label:

```bash
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation -derivedDataPath build \
  -destination 'platform=macOS' build-for-testing -only-testing:InfiniteModuleTests
DYLD_FRAMEWORK_PATH=build/Build/Products/Debug YOOZ_INFINITE_LIVE=1 \
  xcrun xctest -XCTest InfiniteModuleTests.InfiniteCheckpointResumeLiveTests \
  build/Build/Products/Debug/InfiniteModuleTests.xctest
```

A fourth live gate, `testGemma4RotatingCacheCheckpointResume` (`Tests/YoozEngineTests/InfiniteGemma4ParityTests.swift`), covers the same round trip for Gemma4 E4B's `RotatingKVCache` sliding-window layers (512-token window) once genuinely rotated — that suite is app-hosted (`YoozEngineTests`) and needs a desktop GUI session (`INFINITE_LIVE=1` + weights cached, via the `InfiniteLive` scheme), not a headless shell.
