import Foundation

public struct LLMGenerateRequest: Codable, Sendable {
    public let prompt: String
    public let model: String?

    public init(
        prompt: String,
        model: String? = nil
    ) {
        self.prompt = prompt
        self.model = model
    }
}

public struct LLMGenerateResponse: Codable, Sendable {
    public let text: String
    public let model: String
    public let tokensGenerated: Int?
    public let processingTimeMs: Int?
}
