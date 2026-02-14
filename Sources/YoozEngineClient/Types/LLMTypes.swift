import Foundation

public struct LLMGenerateRequest: Codable, Sendable {
    public let prompt: String
    public let model: String?
    public let maxTokens: Int?
    public let temperature: Double?

    public init(
        prompt: String,
        model: String? = nil,
        maxTokens: Int? = nil,
        temperature: Double? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

public struct LLMGenerateResponse: Codable, Sendable {
    public let text: String
    public let model: String
    public let tokensGenerated: Int?
    public let processingTimeMs: Int?
}
