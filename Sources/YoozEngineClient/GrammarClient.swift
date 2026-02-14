import Foundation

/// Client for the grammar check API endpoint.
public struct GrammarClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Check and correct grammar in text.
    public func check(_ request: GrammarCheckRequest) async throws -> GrammarCheckResponse {
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/grammar/check", body: body)
        return try JSONDecoder().decode(GrammarCheckResponse.self, from: data)
    }

    /// Convenience: correct text with all grammar categories.
    public func correct(text: String) async throws -> String {
        let request = GrammarCheckRequest(text: text)
        let response = try await check(request)
        return response.result
    }
}
