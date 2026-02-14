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
}
