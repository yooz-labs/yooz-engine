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

    /// Pre-load the STT model for a language.
    public func loadModel(language: STTLanguage = .english) async throws -> STTStatus {
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
        let wsURL = engine.baseURL
            .appendingPathComponent("v1/stt/stream")
        var urlComponents = URLComponents(url: wsURL, resolvingAgainstBaseURL: false)!
        urlComponents.scheme = "ws"

        guard let url = urlComponents.url else {
            throw YoozEngineError.invalidResponse
        }

        let session = URLSession(configuration: .default)
        let wsTask = session.webSocketTask(with: url)
        wsTask.resume()

        do {
            // Send config
            let config = STTStreamConfig(
                type: "config",
                language: language.rawValue,
                mode: mode.rawValue
            )
            let configData = try JSONEncoder().encode(config)
            let configStr = String(data: configData, encoding: .utf8)!
            try await wsTask.send(.string(configStr))

            // Wait for ready response
            let readyMsg = try await wsTask.receive()
            switch readyMsg {
            case .string(let text):
                if let data = text.data(using: .utf8) {
                    let response = try JSONDecoder().decode(WSReadyResponse.self, from: data)
                    if response.type == "error" {
                        throw YoozEngineError.webSocketError(response.message ?? "Unknown error")
                    }
                }
            case .data:
                break
            @unknown default:
                break
            }
        } catch {
            wsTask.cancel(with: .normalClosure, reason: nil)
            session.invalidateAndCancel()
            throw error
        }

        return STTStream(task: wsTask, session: session)
    }
}

// MARK: - Request/Response Types (Client SDK)

struct BatchSTTRequest: Codable {
    let samples: [Float]
    let language: String
    let mode: String
}

struct STTLoadRequest: Codable {
    let language: String
}

struct STTStreamConfig: Codable {
    let type: String
    let language: String
    let mode: String
}

struct STTLanguagesResponse: Codable {
    let languages: [STTLanguageInfo]
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
}

public struct STTStatus: Codable, Sendable {
    public let loaded: Bool
    public let language: String?
    public let streaming: Bool
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. `nil` when the server omits the field (older builds).
    public let progress: Double?

    public init(
        loaded: Bool,
        language: String?,
        streaming: Bool,
        progress: Double? = nil
    ) {
        self.loaded = loaded
        self.language = language
        self.streaming = streaming
        self.progress = progress
    }
}

public enum AudioMode: String, Codable, Sendable {
    case normal
    case whispered
}

// MARK: - STT Stream

/// A WebSocket-based streaming STT session.
///
/// Send audio samples via `sendAudio(_:)` and receive
/// partial/final results via `receive()`.
@available(macOS 14.0, iOS 17.0, *)
public final class STTStream: @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    private let session: URLSession

    init(task: URLSessionWebSocketTask, session: URLSession) {
        self.task = task
        self.session = session
    }

    deinit {
        task.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
    }

    /// Send audio samples (Float32 at 16kHz) to the engine.
    public func sendAudio(_ samples: [Float]) async throws {
        let data = samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
        try await task.send(.data(data))
    }

    /// Receive the next transcription result from the engine.
    ///
    /// Returns nil when the connection is closed gracefully.
    /// Throws on decoding errors or unexpected network failures.
    public func receive() async throws -> StreamingSTTResult? {
        let message: URLSessionWebSocketTask.Message
        do {
            message = try await task.receive()
        } catch is CancellationError {
            return nil
        } catch let urlError as URLError
            where urlError.code == .cancelled || urlError.code == .networkConnectionLost {
            return nil
        } catch let nsError as NSError
            where nsError.domain == NSPOSIXErrorDomain && nsError.code == 57 {
            // POSIX error 57: socket is not connected (graceful close)
            return nil
        }

        switch message {
        case .string(let text):
            guard let data = text.data(using: .utf8) else {
                throw YoozEngineError.decodingError("Received non-UTF8 text from WebSocket")
            }
            return try JSONDecoder().decode(StreamingSTTResult.self, from: data)
        case .data(let data):
            return try JSONDecoder().decode(StreamingSTTResult.self, from: data)
        @unknown default:
            throw YoozEngineError.invalidResponse
        }
    }

    /// Close the streaming session.
    public func close() {
        task.cancel(with: .normalClosure, reason: nil)
    }
}

/// A streaming STT result received over WebSocket.
public struct StreamingSTTResult: Codable, Sendable {
    /// "partial" or "final"
    public let type: String
    public let text: String
    public let finalized: String
    public let draft: String

    public var isFinal: Bool { type == "final" }
}
