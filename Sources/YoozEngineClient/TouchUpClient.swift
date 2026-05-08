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

// MARK: - LLM model management
//
// These helpers target the LLM routes (`/v1/llm/*`) but live on
// `TouchUpClient` intentionally: the whisper AI settings surface the
// selection under "Touch-up Model", so the thin client's UI code binds
// it next to `process(...)`. Routing is still mode-based inside the
// engine — `setModel` records a user preference that clients can read
// back via `currentModel()` and that whisper uses to restore dropdown
// state on relaunch. Actual load/unload of model weights happens via
// `preloadModel` / `unloadModel`, which call through to TouchUpEngine.

extension TouchUpClient {
    /// Fetch the catalogue of LLM models the engine knows about, including
    /// per-model load state and the currently selected model id.
    public func availableModels() async throws -> LLMModelsResponse {
        let data = try await engine.get("/v1/llm/models")
        return try JSONDecoder().decode(LLMModelsResponse.self, from: data)
    }

    /// Convenience: return just the id of the currently selected model
    /// (process-lifetime preference held by the engine). Equivalent to
    /// `availableModels().current` but saves a round of field decoding
    /// on the caller side.
    public func currentModel() async throws -> String {
        try await availableModels().current
    }

    /// Record `id` as the preferred model. The engine holds this for
    /// the lifetime of its process; clients that need cross-session
    /// persistence should cache their own selection and re-apply via
    /// `setModel(_:)` after reconnect. Does not load weights — use
    /// `preloadModel(_:)` when the UI wants the model warm before the
    /// first request.
    public func setModel(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/model", body: body)
    }

    /// Ensure the named model is loaded and resident. Idempotent:
    /// already-loaded models return immediately. Used by whisper to
    /// warm the model when the user opens the AI tab.
    public func preloadModel(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/preload", body: body)
    }

    /// Free the named model's weights from memory. Whisper uses this to
    /// reclaim GPU memory when the user toggles touch-up off.
    public func unloadModel(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/unload", body: body)
    }
}
