import Foundation

/// Client for the LLM generation API endpoint.
public struct LLMClient: Sendable {
    private let engine: YoozEngineClient

    init(engine: YoozEngineClient) {
        self.engine = engine
    }

    /// Generate text using the LLM.
    public func generate(_ request: LLMGenerateRequest) async throws -> LLMGenerateResponse {
        let body = try JSONEncoder().encode(request)
        let data = try await engine.post("/v1/llm/generate", body: body)
        return try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
    }

    /// Convenience: generate with just a prompt.
    public func generate(
        prompt: String,
        model: String? = nil,
        systemPrompt: String? = nil
    ) async throws -> String {
        let request = LLMGenerateRequest(prompt: prompt, model: model, systemPrompt: systemPrompt)
        let response = try await generate(request)
        return response.text
    }

    /// Status + HF download progress for the preferred LLM tier.
    /// Whisper polls this during the first-run cold-cache pull to render
    /// a progress banner. Shape parity with `STTClient.status()`.
    public func status() async throws -> LLMStatus {
        let data = try await engine.get("/v1/llm/status")
        return try JSONDecoder().decode(LLMStatus.self, from: data)
    }
}
