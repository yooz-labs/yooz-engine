import Foundation

/// Client for the touch-up API endpoint.
public struct TouchUpClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Process text through the touch-up pipeline.
    public func process(_ request: TouchUpRequest) async throws -> TouchUpResponse {
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/touchup", body: body)
        return try JSONDecoder().decode(TouchUpResponse.self, from: data)
    }

    /// Convenience: process text with a given mode.
    public func process(text: String, mode: TouchUpMode = .standard) async throws -> String {
        let request = TouchUpRequest(text: text, mode: mode)
        let response = try await process(request)
        return response.result
    }

    // MARK: - Picker (canonical module-picker pattern)
    //
    // See AGENTS.md "Module model picker pattern" — the same SDK
    // shape (`availableModels()` / `setModel(_:preload:)`) is the
    // documented canon for every module that exposes a picker so
    // app-side wiring is templated.

    /// List every TouchUp model the engine knows about, with
    /// availability + cache + load + active flags. Drives the
    /// LLM-model picker UI in consuming apps (yooz-whisper,
    /// yooz-notes, ...).
    public func availableModels() async throws -> TouchUpModelsResponse {
        let data = try await engine.get("/v1/touchup/models")
        return try JSONDecoder().decode(TouchUpModelsResponse.self, from: data)
    }

    /// Set the active model and (optionally) preload it. The default
    /// `preload: true` makes a picker change one-shot — the next
    /// `process(...)` call will not pay a cold-start.
    ///
    /// - Returns: The picker row for the new active model. The flags
    ///   reflect post-preload state (e.g. `isLoaded == true` after
    ///   a successful preload).
    /// - Throws: A 400 from the server for unknown ids, a 501 for
    ///   FoundationModels on pre-26 macOS, or the underlying
    ///   load-path error otherwise.
    @discardableResult
    public func setModel(
        id: String,
        preload: Bool = true
    ) async throws -> TouchUpModelInfo {
        let body = try JSONEncoder().encode(
            TouchUpSetModelRequest(id: id, preload: preload)
        )
        let data = try await engine.post("/v1/touchup/model", body: body)
        return try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
    }
}
