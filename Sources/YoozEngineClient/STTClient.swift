import Foundation

/// Client for the STT API endpoints.
///
/// Supports batch transcription (REST) and streaming (WebSocket).
public struct STTClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    // MARK: - REST Endpoints

    /// Batch transcribe an audio buffer.
    public func transcribe(
        audioSamples: [Float],
        language: STTLanguage = .english,
        mode: AudioMode = .normal
    ) async throws -> TranscriptionResult {
        let request = BatchSTTRequest(
            samples: audioSamples,
            language: language.rawValue,
            mode: mode.rawValue
        )
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/stt/batch", body: body)
        return try JSONDecoder().decode(TranscriptionResult.self, from: data)
    }

    /// Batch transcribe with word/token-level timestamps.
    ///
    /// Identical to `transcribe(...)` but sets the server-side `aligned` flag
    /// so the response includes `TranscriptionResult.tokens` — each token's
    /// `start` / `end` is seconds from the start of `audioSamples`. Used by
    /// callers that need chunk-boundary deduplication, subtitle alignment, or
    /// per-word highlighting.
    ///
    /// Supported backends: Parakeet (TDT token alignment), FastConformer, and
    /// Apple STT (derived from `SFTranscriptionSegment`). `tokens` is non-nil
    /// (possibly empty) on every response from this method — the server ships
    /// `"tokens": []` on silent audio rather than omitting the key. Only the
    /// text-only `transcribe(...)` path ever returns `nil` tokens.
    public func batchTranscribeAligned(
        audioSamples: [Float],
        language: STTLanguage = .english,
        mode: AudioMode = .normal
    ) async throws -> TranscriptionResult {
        let request = BatchSTTRequest(
            samples: audioSamples,
            language: language.rawValue,
            mode: mode.rawValue,
            aligned: true
        )
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/stt/batch", body: body)
        return try JSONDecoder().decode(TranscriptionResult.self, from: data)
    }

    /// Pre-load the STT model for a language. Blocks until the
    /// model is loaded (or the load fails).
    ///
    /// Sends `?wait=true` so the call preserves its pre-engine#125
    /// blocking semantics. New code that wants to dispatch and
    /// poll for completion should call `loadModelAsync(...)` and
    /// poll `/v1/stt/status` for the `state == .ready` transition.
    public func loadModel(language: STTLanguage = .english) async throws -> STTStatus {
        let request = STTLoadRequest(language: language.rawValue)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/stt/load?wait=true", body: body)
        return try JSONDecoder().decode(STTStatus.self, from: data)
    }

    /// Dispatch a load on the engine and return immediately
    /// (HTTP 202). The returned `STTStatus` will have
    /// `loaded == false` and `state == .loading`; poll
    /// `/v1/stt/status` until `state == .ready` (or `.failed`).
    /// Use for first-run pulls of large weights (qwen3 preview
    /// ~2.5 GB, etc.) where the blocking call would HTTP-timeout.
    /// Idempotent: a second `loadModelAsync` for the same language
    /// while a load is in flight shares the same Task on the
    /// server.
    public func loadModelAsync(
        language: STTLanguage = .english
    ) async throws -> STTStatus {
        let request = STTLoadRequest(language: language.rawValue)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/stt/load", body: body)
        return try JSONDecoder().decode(STTStatus.self, from: data)
    }

    /// Get the current STT engine status.
    public func status() async throws -> STTStatus {
        let data = try await engine.get("/v1/stt/status")
        return try JSONDecoder().decode(STTStatus.self, from: data)
    }

    /// Get available STT languages.
    public func languages() async throws -> [STTLanguageInfo] {
        let data = try await engine.get("/v1/stt/languages")
        let response = try JSONDecoder().decode(STTLanguagesResponse.self, from: data)
        return response.languages
    }

    // MARK: - Picker (canonical module-picker pattern, #99)
    //
    // Mirrors `TouchUpClient.availableModels()` /
    // `setModel(id:preload:)` so consumer apps can template a
    // single ModelPickerStore<T> across pickers. See AGENTS.md
    // "Module model picker pattern" for the full recipe.

    /// List every STT backend the engine knows about, with
    /// lifecycle state + active flag + STT-specific capability
    /// extensions (`supportsBatch`, `supportsStreaming`,
    /// `supportedLanguages`).
    public func availableEngines() async throws -> STTBackendsResponse {
        let data = try await engine.get("/v1/stt/engine")
        return try JSONDecoder().decode(STTBackendsResponse.self, from: data)
    }

    /// Set the active STT backend. `preload` is accepted for shape
    /// parity with the TouchUp picker but is currently a no-op on
    /// the server — STT models load lazily on the first
    /// `/v1/stt/batch` or `/v1/stt/load` call after the switch.
    /// Documented as a future enhancement so the SDK shape stays
    /// stable as the engine adds eager preload.
    @discardableResult
    public func setEngine(
        id: String,
        preload: Bool = true
    ) async throws -> STTBackendInfo {
        let body = try JSONEncoder().encode(
            STTSetBackendRequest(id: id, preload: preload)
        )
        let data = try await engine.post("/v1/stt/engine", body: body)
        return try JSONDecoder().decode(STTBackendInfo.self, from: data)
    }

    // MARK: - WebSocket Streaming

    /// Open a streaming STT session over WebSocket.
    ///
    /// Usage:
    /// ```swift
    /// let stream = try await client.stt.startStream(language: .english)
    /// try await stream.sendAudio(samples)
    /// if let result = try await stream.receive() {
    ///     print(result.text)
    /// }
    /// ```
    @available(macOS 14.0, iOS 17.0, *)
    public func startStream(
        language: STTLanguage = .english,
        mode: AudioMode = .normal
    ) async throws -> STTStream {
        // The transport opens the stream: the loopback transport performs the
        // WebSocket config/ready handshake; the in-process transport sets up an
        // engine StreamingTranscriber / Qwen3 session / Apple buffer directly
        // (epic #192 Phase 2b). Either way the SDK gets a uniform session.
        let session = try await engine.openSTTStream(
            language: language.rawValue,
            mode: mode.rawValue
        )
        return STTStream(session: session)
    }
}

// MARK: - Request/Response Types (Client SDK)

struct BatchSTTRequest: Codable {
    let samples: [Float]
    let language: String
    let mode: String
    /// Opt-in flag for per-token alignment in the response. Omitted on the
    /// wire (`encodeIfPresent`) when `nil` so old clients and servers that
    /// predate issue #34 remain byte-identical with today's traffic.
    let aligned: Bool?

    init(samples: [Float], language: String, mode: String, aligned: Bool? = nil) {
        self.samples = samples
        self.language = language
        self.mode = mode
        self.aligned = aligned
    }

    enum CodingKeys: String, CodingKey {
        case samples, language, mode, aligned
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(samples, forKey: .samples)
        try container.encode(language, forKey: .language)
        try container.encode(mode, forKey: .mode)
        try container.encodeIfPresent(aligned, forKey: .aligned)
    }
}

struct STTLoadRequest: Codable {
    let language: String
}

struct STTStreamConfig: Codable {
    let type: String
    let language: String
    let mode: String
}

public struct STTLanguagesResponse: Codable, Sendable {
    public let languages: [STTLanguageInfo]

    public init(languages: [STTLanguageInfo]) {
        self.languages = languages
    }
}

struct WSReadyResponse: Decodable {
    let type: String
    let message: String?
}

public struct STTLanguageInfo: Codable, Sendable {
    public let code: String
    public let name: String
    public let implemented: Bool
    public let family: String

    public init(code: String, name: String, implemented: Bool, family: String) {
        self.code = code
        self.name = name
        self.implemented = implemented
        self.family = family
    }
}

public struct STTStatus: Codable, Sendable {
    public let loaded: Bool
    public let language: String?
    public let streaming: Bool
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. `nil` when the server omits the field (older builds).
    public let progress: Double?
    /// Lifecycle state for the active STT backend (engine#125).
    /// `nil` on pre-#125 server builds; consumers MAY infer state
    /// from `loaded` + `progress` when nil.
    public let state: LoadState?
    /// Human-readable error from the last failed load. `nil` unless
    /// `state == .failed`.
    public let lastError: String?

    public init(
        loaded: Bool,
        language: String?,
        streaming: Bool,
        progress: Double? = nil,
        state: LoadState? = nil,
        lastError: String? = nil
    ) {
        self.loaded = loaded
        self.language = language
        self.streaming = streaming
        self.progress = progress
        self.state = state
        self.lastError = lastError
    }
}

public enum AudioMode: String, Codable, Sendable {
    case normal
    case whispered
}

// MARK: - STT Stream

/// A streaming STT session.
///
/// Send audio samples via `sendAudio(_:)` and receive partial/final results via
/// `receive()`. The session is transport-backed (epic #192 Phase 2b): loopback
/// over a WebSocket, or in-process driving the engine transcriber directly. The
/// public API is identical either way.
@available(macOS 14.0, iOS 17.0, *)
public final class STTStream: @unchecked Sendable {
    private let session: any STTStreamSession

    init(session: any STTStreamSession) {
        self.session = session
    }

    deinit {
        // Release transport resources if the caller dropped the stream without
        // calling close() — cancels the WebSocket (loopback) or finalizes the
        // engine transcriber / Apple buffer (in-process). close() is idempotent.
        session.close()
    }

    /// Send audio samples (Float32 at 16kHz) to the engine.
    public func sendAudio(_ samples: [Float]) async throws {
        try await session.sendAudio(samples)
    }

    /// Receive the next transcription result from the engine.
    ///
    /// Returns nil when the session is closed/exhausted. Throws on decoding
    /// errors or unexpected failures.
    public func receive() async throws -> StreamingSTTResult? {
        try await session.receive()
    }

    /// Close the streaming session.
    public func close() {
        session.close()
    }
}

/// A streaming STT result (partial or final).
public struct StreamingSTTResult: Codable, Sendable {
    /// "partial" or "final"
    public let type: String
    public let text: String
    public let finalized: String
    public let draft: String

    public var isFinal: Bool { type == "final" }

    public init(type: String, text: String, finalized: String, draft: String) {
        self.type = type
        self.text = text
        self.finalized = finalized
        self.draft = draft
    }
}
