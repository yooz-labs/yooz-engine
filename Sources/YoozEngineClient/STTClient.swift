import Foundation

/// Client for the STT API endpoints.
///
/// Supports both batch transcription (REST) and streaming (WebSocket).
public final class STTClient: @unchecked Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Batch transcribe an audio buffer.
    public func transcribe(
        audioSamples: [Float],
        language: STTLanguage = .english
    ) async throws -> TranscriptionResult {
        let request = BatchSTTRequest(
            samples: audioSamples,
            language: language.rawValue
        )
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/stt/batch", body: body)
        return try JSONDecoder().decode(TranscriptionResult.self, from: data)
    }

    // TODO: Phase 2 - WebSocket streaming
    // public func startStream(language: STTLanguage) async throws -> STTStream
}

struct BatchSTTRequest: Codable {
    let samples: [Float]
    let language: String
}
