import Foundation

/// Client for the model-management API — the disk-hygiene surface behind the
/// app's "Manage Models" tab and its one-time cleanup migration.
///
/// Distinct from the per-module pickers (`touchUp.availableModels()`,
/// `stt`): this is the cross-module inventory with **real on-disk sizes** plus
/// delete and cleanup. All storage-layout knowledge stays in the engine; the app
/// only calls these three methods.
public struct ModelsClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// The full inventory of models with real on-disk sizes, cache/load/active
    /// flags, and per-row deletability.
    public func list() async throws -> ManagedModelsResponse {
        let data = try await engine.get("/v1/models")
        return try JSONDecoder().decode(ManagedModelsResponse.self, from: data)
    }

    /// Delete one model's reclaimable on-disk copies (hub repo + models-directory
    /// copy). The engine unloads it from memory first and refuses to delete the
    /// active model. Returns bytes reclaimed.
    @discardableResult
    public func delete(id: String) async throws -> DeleteModelResult {
        // Percent-encode the id so a hub dir name (`models--ns--repo`) survives
        // as a single path segment.
        let encoded = id.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed
        ) ?? id
        let data = try await engine.delete("/v1/models/\(encoded)")
        return try JSONDecoder().decode(DeleteModelResult.self, from: data)
    }

    /// Run the one-shot disk-hygiene pass: collapse superseded snapshots across
    /// every cached repo and remove copies made redundant by a higher-priority
    /// (bundled / models-directory) copy. Idempotent — a second call reclaims 0.
    @discardableResult
    public func cleanup() async throws -> ModelCleanupResult {
        let data = try await engine.post("/v1/models/cleanup", body: Data())
        return try JSONDecoder().decode(ModelCleanupResult.self, from: data)
    }
}
