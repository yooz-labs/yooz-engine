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

    /// Explicitly download a model's weights WITHOUT changing the active
    /// selection (engine#288 slice 2) — the "Download" button. Returns the
    /// tier's current row immediately; progress and the terminal outcome
    /// arrive via the event stream (`downloadProgress` /
    /// `loadStateChanged`), exactly like a preloading `setModel`.
    /// 400 (`invalid_model`) for non-downloadable ids (Apple Intelligence).
    @discardableResult
    public func downloadModel(id: String) async throws -> TouchUpModelInfo {
        let body = try JSONEncoder().encode(TouchUpDownloadRequest(id: id))
        let data = try await engine.post("/v1/touchup/download", body: body)
        return try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
    }

    /// Cancel an in-flight download — the "Cancel" button. No-op (returns
    /// the current row) when the tier isn't downloading; never unloads a
    /// resident model.
    @discardableResult
    public func cancelDownload(id: String) async throws -> TouchUpModelInfo {
        let body = try JSONEncoder().encode(TouchUpDownloadRequest(id: id))
        let data = try await engine.post("/v1/touchup/download/cancel", body: body)
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
    ///
    /// Sends `?wait=true` so the call preserves its pre-engine#125
    /// blocking semantics — returns only when the model is loaded
    /// (or the load fails). New code that wants to dispatch and
    /// poll for completion should call
    /// `preloadModelAsync(_:)` instead.
    public func preloadModel(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/preload?wait=true", body: body)
    }

    /// Dispatch a preload on the engine and return immediately
    /// (HTTP 202). Caller polls `/v1/llm/status` for the
    /// `state == .ready` transition. Use for first-run pulls of
    /// large weights (Quality v2 LoRA fused, etc.) where the
    /// blocking call would HTTP-timeout. Idempotent: a second
    /// `preloadModelAsync` for the same tier while a load is in
    /// flight is a no-op on the server (shares the same Task).
    public func preloadModelAsync(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/preload", body: body)
    }

    /// Free the named model's weights from memory. Whisper uses this to
    /// reclaim GPU memory when the user toggles touch-up off.
    public func unloadModel(_ id: String) async throws {
        let body = try JSONEncoder().encode(LLMModelSelection(model: id))
        _ = try await engine.post("/v1/llm/unload", body: body)
    }

    /// Drop the cached prompt-KV state for one LLM tier — or, if `id` is
    /// omitted, every currently-loaded tier — WITHOUT unloading weights
    /// (engine#299). The middle lever between doing nothing and
    /// `unloadModel(_:)`: reclaims the retained-KV memory delta a
    /// steady-state proofreading workload builds up (measured ~1.5 GB after
    /// sustained traffic) while leaving the model warm, so the next call
    /// pays only prompt-recomputation cost, not a full cold reload. Also
    /// scoped to the LLM tiers alone, unlike `/v1/session/begin`, which
    /// resets every module (STT included) as a per-recording boundary.
    ///
    /// - Returns: The wire ids of the tiers actually cleared. Empty when
    ///   nothing was loaded — clearing an already-empty cache is a success
    ///   no-op, never a thrown error.
    @discardableResult
    public func clearCache(_ id: String? = nil) async throws -> [String] {
        let body = try JSONEncoder().encode(LLMClearCacheRequest(model: id))
        let data = try await engine.post("/v1/llm/clear-cache", body: body)
        return try JSONDecoder().decode(LLMClearCacheResponse.self, from: data).cleared
    }
}
