import Foundation

public enum TouchUpMode: String, Codable, Sendable {
    case off
    case light
    case standard
    case full
}

public struct TouchUpRequest: Codable, Sendable {
    public let text: String
    public let mode: TouchUpMode
    public let language: String?

    public init(text: String, mode: TouchUpMode, language: String? = nil) {
        self.text = text
        self.mode = mode
        self.language = language
    }
}

public struct TouchUpResponse: Codable, Sendable {
    public let result: String
    public let mode: TouchUpMode
    public let processingTimeMs: Int?
    public let modelUsed: String?
    public let warnings: [String]?
}
