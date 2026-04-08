import Foundation

/// Client for the engine's Voice Activity Detection (VAD) endpoint.
///
/// The VAD module uses Silero v6.0.0 with 512-sample windows (32ms at 16kHz)
/// and falls back to energy-based detection when the CoreML model is unavailable.
///
/// ## Chunk-based transcription
///
/// Client apps typically use VAD to drive chunk-based transcription pipelines.
/// The recommended pattern is:
///
/// 1. Stream audio to the engine and call ``detect(audioSamples:reset:)``
///    to find speech boundaries.
/// 2. Determine chunk boundaries using one of:
///    - **Silence**: VAD detects sustained silence after post-speech smoothing.
///    - **Word count**: 25-50 words accumulated from streaming STT results.
///    - **Punctuation**: sentence-ending punctuation (`.`, `!`, `?`) after 25+ words.
///    - **Recording ended**: user stops recording.
/// 3. **Overlap context**: prepend 0.8 seconds (12,800 samples at 16kHz) from the
///    previous chunk to provide acoustic context for edge-word transcription accuracy.
/// 4. **Process pipeline**: VAD -> STT (batch) -> Grammar -> TouchUp.
/// 5. **Track pending chunks** with a 30-second timeout to avoid blocking on
///    slow batch transcriptions.
///
/// When sending consecutive chunks from the same recording, pass `reset: false`
/// to preserve RNN state continuity between calls.
public struct VADClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Detect speech segments in audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: Audio samples at 16kHz, minimum 512 samples (32ms).
    ///   - reset: When true (default), resets the RNN hidden/cell state before
    ///     detection. Set to false when sending consecutive chunks from the same
    ///     recording to preserve inter-frame state continuity.
    /// - Returns: A ``VADResponse`` containing detected speech segments.
    public func detect(audioSamples: [Float], reset: Bool = true) async throws -> VADResponse {
        let request = VADRequest(samples: audioSamples, reset: reset)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/vad/detect", body: body)
        return try JSONDecoder().decode(VADResponse.self, from: data)
    }
}

struct VADRequest: Codable {
    let samples: [Float]
    let reset: Bool?
}

public struct VADResponse: Codable, Sendable {
    public let segments: [SpeechSegment]
}

public struct SpeechSegment: Codable, Sendable {
    public let startMs: Int
    public let endMs: Int
    public let probability: Float
}
