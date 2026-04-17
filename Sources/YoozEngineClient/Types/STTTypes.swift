import Foundation

public struct TranscriptionResult: Codable, Sendable {
    public let text: String
    public let finalized: String
    public let draft: String
    public let language: String?

    public init(text: String, finalized: String, draft: String, language: String? = nil) {
        self.text = text
        self.finalized = finalized
        self.draft = draft
        self.language = language
    }
}

public enum STTLanguage: String, Codable, Sendable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case dutch = "nl"
    case polish = "pl"
    case russian = "ru"
    case ukrainian = "uk"
    case arabic = "ar"
    case persian = "fa"
    case hebrew = "he"
    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case cantonese = "yue"
}
