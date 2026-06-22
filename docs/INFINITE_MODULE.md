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
| `POST` | `/v1/infinite/sessions/:id/checkpoint` | `InfiniteCheckpointSessionResponse` | Body: `{ "label": String? }`. |
| `POST` | `/v1/infinite/sessions/:id/generate` | `InfiniteGenerateSessionResponse` | Generates from the session's accumulated context on Swift-runtime-supported models (Qwen3.6 `qwen3_5_moe` today), bounded to the model's native window (≤262K; 1M paging tracked in #180). Models without a Swift backend yet (Gemma4, retrieval) return `501 generation_unavailable` citing #184. Session state is preserved either way. |
| `DELETE` | `/v1/infinite/sessions/:id` | `InfiniteDeleteSessionResponse` | Releases the engine-owned session. |

The cleanup policy is:

```text
explicit_delete_or_process_exit;max_active_sessions=16
```

`/v1/session/begin` is a recording boundary for STT-style work and does not delete Infinite sessions. Consumers must call `DELETE /v1/infinite/sessions/:id` when finished.

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

Generation runs on Swift-runtime-supported models (Qwen3.6 `qwen3_5_moe`). Models without a Swift MLX backend yet (Gemma4, retrieval) throw `generation_unavailable` until the gemma4 port lands (#184):

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
    // The active model has no Swift MLX backend yet (e.g. Gemma4 — see #184).
    // Switch to a supported model, or branch on the stable `code`.
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

The integration suite starts a served engine and drives Infinite through `YoozEngineClient`, including `/v1/modules`, model picker, status, create, append, fetch, checkpoint, generation (`501 generation_unavailable` against the default Gemma4 model, which has no Swift backend yet — #184), delete, and deleted-session `404`.
