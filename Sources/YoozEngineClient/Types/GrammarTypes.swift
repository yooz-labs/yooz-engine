import Foundation

public struct GrammarCheckRequest: Codable, Sendable {
    public let text: String
    public let categories: [String]?

    public init(text: String, categories: [String]? = nil) {
        self.text = text
        self.categories = categories
    }
}

public struct GrammarCheckResponse: Codable, Sendable {
    public let result: String
    public let correctionsApplied: Int
}
