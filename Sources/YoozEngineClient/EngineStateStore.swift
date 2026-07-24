import Combine
import Foundation

/// Generic, transport-agnostic picker state (engine#226) — the canonical
/// wiring for a consumer app's model picker. Seeds itself from
/// `GET /v1/state` and stays live via `/v1/events`, so a picker UI built on
/// this store has:
///
///   - **zero local persistence** — the engine remembers the active
///     selection across restarts (`ModelSelectionStore`); the app never
///     writes its own "last selected model" setting, and
///   - **zero polling** — `/v1/events` pushes every state transition
///     (`modelChanged`, `loadStateChanged`, `downloadProgress`,
///     `residencyChanged`); the app never re-fetches on a timer to notice a
///     download finished.
///
/// This is the substrate `ModelPickerStore<T>` (the existing per-module,
/// richly-typed picker pattern documented in AGENTS.md "Module model picker
/// pattern") is expected to sit on top of for a module whose picker needs
/// extension fields (`STTBackendInfo.supportsStreaming`, etc.) that don't
/// fit `EngineModelSnapshotRow`'s canonical seven fields — that module's
/// store still calls its own `GET /v1/<module>/models` for the rich rows,
/// but can drive its "is a load in flight / what's the progress" state off
/// this store's `latestEvent(for:module:)` instead of its own poll loop.
///
/// See `docs/CONSUMER_INTEGRATION.md` "The canonical model picker pattern"
/// for the full wiring recipe.
@MainActor
public final class EngineStateStore: ObservableObject {
    /// Per-module snapshot, keyed by `EngineModuleSnapshot.module`
    /// (`"touchup"` today). Absent until the first `refreshSnapshot()` /
    /// `start()` call completes.
    @Published public private(set) var modules: [String: EngineModuleSnapshot] = [:]

    /// Most recent event per `(module, kind)`. Useful for UI that needs the
    /// live `downloadProgress` fraction or a `loadStateChanged` failure
    /// `message`, neither of which is captured by `modules` (a snapshot has
    /// no "in-flight" representation — see `EngineModelSnapshotRow`).
    @Published public private(set) var latestEvents: [EventKey: EngineEvent] = [:]

    /// Set when either the initial snapshot fetch or the live event
    /// subscription fails. Cleared on the next successful `refreshSnapshot()`.
    @Published public private(set) var lastError: String?

    private let client: YoozEngineClient
    private var eventTask: Task<Void, Never>?

    public init(client: YoozEngineClient) {
        self.client = client
    }

    // No `deinit` cancelling `eventTask`: `deinit` runs nonisolated even for
    // a `@MainActor` type, so it cannot touch a MainActor-isolated stored
    // property without extra ceremony (`nonisolated(unsafe)`) that would
    // only mask a real lifecycle bug. Callers own the subscription's
    // lifetime explicitly via `stop()` — e.g. a SwiftUI view calls it from
    // `.onDisappear`, mirroring how `APIServer` (this repo's other
    // `@MainActor` `ObservableObject`) also has no `deinit` and instead
    // relies on an explicit `stop()`.

    /// Composite key for `latestEvents`. A plain tuple can't be a
    /// `Dictionary` key (no `Hashable`), so this wraps the two fields every
    /// caller cares about together.
    public struct EventKey: Hashable, Sendable {
        public let module: String
        public let kind: EngineEventKind

        public init(module: String, kind: EngineEventKind) {
            self.module = module
            self.kind = kind
        }
    }

    /// Fetch the initial snapshot, then start (or restart) the live event
    /// subscription. Call once at picker mount. Safe to call again (e.g.
    /// after a transport reconnect) — cancels any prior subscription first.
    ///
    /// Order matters (PR #239 review): subscribe FIRST, then fetch the
    /// snapshot. The bus has no replay, so the opposite order drops any
    /// transition landing in the fetch-to-subscribe window (classically: a
    /// download completing right as the picker mounts) and the store stays
    /// stale until the next unrelated event. With subscribe-first, an event
    /// arriving while the snapshot fetch is in flight is applied on this
    /// `@MainActor` store in arrival order; the snapshot — captured by the
    /// engine AFTER the subscription existed — then lands as at-least-as-new
    /// baseline state.
    @available(macOS 14.0, iOS 17.0, *)
    public func start() async {
        subscribeToEvents()
        await refreshSnapshot()
    }

    /// Cancel the live event subscription. `modules`/`latestEvents` are left
    /// as-is (last known state), matching how a picker should render while
    /// disconnected — stale data, not empty data.
    public func stop() {
        eventTask?.cancel()
        eventTask = nil
    }

    /// Re-fetch `GET /v1/state` and replace every module's snapshot.
    /// Normally only needed once at `start()` — `/v1/events` keeps the
    /// store current after that — but exposed for a manual pull-to-refresh
    /// affordance or recovery after `lastError` is set.
    public func refreshSnapshot() async {
        do {
            let snapshot = try await client.engineState.snapshot()
            apply(snapshot)
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Look up the most recent event of `kind` for `module`, or nil if none
    /// has been observed this session.
    public func latestEvent(module: String, kind: EngineEventKind) -> EngineEvent? {
        latestEvents[EventKey(module: module, kind: kind)]
    }

    @available(macOS 14.0, iOS 17.0, *)
    private func subscribeToEvents() {
        eventTask?.cancel()
        eventTask = Task { [weak self, client] in
            guard let self else { return }
            do {
                let stream = try await client.openEvents()
                for await event in stream {
                    if Task.isCancelled { break }
                    self.apply(event)
                }
            } catch {
                if !Task.isCancelled {
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    private func apply(_ snapshot: EngineStateSnapshot) {
        for module in snapshot.modules {
            modules[module.module] = module
        }
    }

    private func apply(_ event: EngineEvent) {
        latestEvents[EventKey(module: event.module, kind: event.kind)] = event

        // Match on the typed payload, not the raw kind + hand-unwrapped
        // optionals: `EngineEventPayload` centralizes the kind-dependent
        // field contract, and a malformed/newer-engine frame lands in one
        // `.unrecognized` arm instead of a per-case `guard let` maze.
        switch event.payload {
        case .modelChanged(let modelId):
            guard let module = modules[event.module] else { return }
            modules[event.module] = EngineModuleSnapshot(
                module: module.module,
                models: module.models.map {
                    EngineModelSnapshotRow(
                        id: $0.id, displayName: $0.displayName, description: $0.description,
                        tier: $0.tier, sizeBytes: $0.sizeBytes, loadState: $0.loadState,
                        isActive: $0.id == modelId,
                        // Carry the cached fraction across an unrelated event
                        // (PR #293 review): dropping it here would blank a
                        // mid-download tier's progress every time the active
                        // model changed, until the next full refresh.
                        downloadProgress: $0.downloadProgress
                    )
                },
                activeId: modelId
            )
        case .loadStateChanged(let modelId, let loadState, _):
            guard let module = modules[event.module] else { return }
            modules[event.module] = EngineModuleSnapshot(
                module: module.module,
                models: module.models.map { row in
                    guard row.id == modelId else { return row }
                    return EngineModelSnapshotRow(
                        id: row.id, displayName: row.displayName, description: row.description,
                        tier: row.tier, sizeBytes: row.sizeBytes, loadState: loadState,
                        isActive: row.isActive,
                        // A settled tier is no longer downloading, so clear
                        // the fraction; anything still loading keeps it
                        // (PR #293 review).
                        downloadProgress: loadState == .cached || loadState == .loaded
                            ? nil : row.downloadProgress
                    )
                },
                activeId: module.activeId
            )
        case .downloadProgress, .residencyChanged:
            // `downloadProgress` frames surface via `latestEvents` /
            // `latestEvent(_:)`; the snapshot's own per-row
            // `downloadProgress` (engine#292) is refreshed by
            // `refreshSnapshot()`, which is the reconciliation arm a
            // consumer falls back on when a frame is dropped.
            // `residencyChanged` carries no snapshot field of its own:
            // The stale-row concern `residencyChanged` might suggest is
            // covered engine-side: eviction publishes a per-model
            // `loadStateChanged` for every evicted tier (see
            // `TouchUpEngine.evictModelsExcept`), which the arm above applies.
            break
        case .unrecognized:
            // Newer-engine event kind or malformed frame — already recorded
            // in `latestEvents` for diagnostics; nothing to fold into
            // `modules`.
            break
        }
    }
}
