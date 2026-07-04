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

    /// List engine-owned Infinite long-context sessions.
    public func sessions() async throws -> InfiniteSessionsResponse {
        let data = try await engine.get("/v1/infinite/sessions")
        return try JSONDecoder().decode(InfiniteSessionsResponse.self, from: data)
    }

    /// Create an engine-owned Infinite long-context session.
    public func createSession(
        modelId: String? = nil,
        label: String? = nil
    ) async throws -> InfiniteSessionInfo {
        let request = InfiniteCreateSessionRequest(modelId: modelId, label: label)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/infinite/sessions", body: body)
        return try JSONDecoder().decode(InfiniteSessionInfo.self, from: data)
    }

    /// Fetch one engine-owned Infinite long-context session.
    public func session(id: String) async throws -> InfiniteSessionInfo {
        let data = try await engine.get("/v1/infinite/sessions/\(id)")
        return try JSONDecoder().decode(InfiniteSessionInfo.self, from: data)
    }

    /// Append context text to an Infinite session.
    public func append(
        sessionId: String,
        text: String
    ) async throws -> InfiniteAppendSessionResponse {
        let request = InfiniteAppendSessionRequest(text: text)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post(
            "/v1/infinite/sessions/\(sessionId)/append",
            body: body
        )
        return try JSONDecoder().decode(InfiniteAppendSessionResponse.self, from: data)
    }

    /// Request generation from an Infinite session.
    public func generate(
        sessionId: String,
        prompt: String? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil
    ) async throws -> InfiniteGenerateSessionResponse {
        let request = InfiniteGenerateSessionRequest(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature
        )
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post(
            "/v1/infinite/sessions/\(sessionId)/generate",
            body: body
        )
        return try JSONDecoder().decode(InfiniteGenerateSessionResponse.self, from: data)
    }

    /// Checkpoint an Infinite session for later lifecycle inspection.
    public func checkpoint(
        sessionId: String,
        label: String? = nil
    ) async throws -> InfiniteCheckpointSessionResponse {
        let request = InfiniteCheckpointSessionRequest(label: label)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post(
            "/v1/infinite/sessions/\(sessionId)/checkpoint",
            body: body
        )
        return try JSONDecoder().decode(InfiniteCheckpointSessionResponse.self, from: data)
    }

    /// Delete an Infinite session and release its engine-owned context.
    public func deleteSession(id: String) async throws -> InfiniteDeleteSessionResponse {
        let data = try await engine.delete("/v1/infinite/sessions/\(id)")
        return try JSONDecoder().decode(InfiniteDeleteSessionResponse.self, from: data)
    }
}
