import Foundation

/// Client for the Infinite long-context API endpoints.
public struct InfiniteClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// List Infinite long-context models/modes known to the engine.
    public func availableModels() async throws -> InfiniteModelsResponse {
        let data = try await engine.get("/v1/infinite/models")
        return try JSONDecoder().decode(InfiniteModelsResponse.self, from: data)
    }

    /// Select the active Infinite model/mode.
    @discardableResult
    public func setModel(id: String, preload: Bool = true) async throws -> InfiniteModelInfo {
        let request = InfiniteSetModelRequest(id: id, preload: preload)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/infinite/model", body: body)
        return try JSONDecoder().decode(InfiniteModelInfo.self, from: data)
    }

    /// Status for the active Infinite model/mode.
    public func status() async throws -> InfiniteStatus {
        let data = try await engine.get("/v1/infinite/status")
        return try JSONDecoder().decode(InfiniteStatus.self, from: data)
    }
}
