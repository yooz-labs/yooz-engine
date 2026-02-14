import Foundation

/// Client for the VAD API endpoint.
public struct VADClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Detect speech segments in audio samples.
    public func detect(audioSamples: [Float]) async throws -> VADResponse {
        let request = VADRequest(samples: audioSamples)
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/vad/detect", body: body)
        return try JSONDecoder().decode(VADResponse.self, from: data)
    }
}

struct VADRequest: Codable {
    let samples: [Float]
}

public struct VADResponse: Codable, Sendable {
    public let segments: [SpeechSegment]
}

public struct SpeechSegment: Codable, Sendable {
    public let startMs: Int
    public let endMs: Int
    public let probability: Float
}
