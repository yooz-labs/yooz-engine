import Foundation

/// Client for `GET /v1/state` (engine#226): the full cross-module snapshot
/// of every module's picker catalog + active id, in one call. Distinct
/// from the per-module pickers (`touchUp.availableModels()`, ...): this is
/// the module-agnostic view `EngineStateStore` fetches once at `start()`,
/// before subscribing to `/v1/events` for live updates, so a picker UI has
/// something to render before the first pushed event arrives.
public struct EngineStateClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Fetch the current snapshot.
    public func snapshot() async throws -> EngineStateSnapshot {
        let data = try await engine.get("/v1/state")
        return try JSONDecoder().decode(EngineStateSnapshot.self, from: data)
    }
}
