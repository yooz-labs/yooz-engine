// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - Model Family

/// Model family determines which architecture and weights to use
public enum ModelFamily: String, Codable, Sendable {
    /// Apple's built-in SFSpeechRecognizer (on-device)
    case apple

    /// Parakeet TDT - Conformer encoder with Token-and-Duration Transducer
    /// Supports English and European languages
    case parakeetTDT = "parakeet-tdt"

    /// FastConformer - NVIDIA's FastConformer architecture
    /// Supports Arabic, Persian, and Hebrew
    case fastConformer = "fast-conformer"

    /// SenseVoice or similar CJK-optimized model
    /// Supports Chinese, Japanese, Korean
    case cjk
}

// MARK: - STT Language

/// Languages supported by the STT engine
/// Each language maps to a specific model family and identifier
public enum STTLanguage: String, Codable, Sendable, CaseIterable {
    // MARK: - Latin/European Languages (Parakeet TDT)

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

    // MARK: - CJK Languages

    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case cantonese = "yue"

    // MARK: - RTL Languages (FastConformer)

    case arabic = "ar"
    case persian = "fa"
    case hebrew = "he"

    // MARK: - Properties

    /// The model family to use for this language
    public var modelFamily: ModelFamily {
        switch self {
        case .chinese, .japanese, .korean, .cantonese:
            .cjk
        case .arabic, .persian, .hebrew:
            .fastConformer
        default:
            .parakeetTDT
        }
    }

    /// Model identifier for loading the appropriate weights
    public var modelIdentifier: String {
        switch self {
        case .arabic:
            "fastconformer-ar"
        case .persian:
            "fastconformer-fa"
        case .hebrew:
            "fastconformer-he"
        case .chinese, .japanese, .korean, .cantonese:
            "sensevoice-cjk"
        default:
            "parakeet-tdt"
        }
    }

    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .english: "English"
        case .spanish: "Spanish"
        case .french: "French"
        case .german: "German"
        case .italian: "Italian"
        case .portuguese: "Portuguese"
        case .dutch: "Dutch"
        case .polish: "Polish"
        case .russian: "Russian"
        case .ukrainian: "Ukrainian"
        case .chinese: "Chinese (Mandarin)"
        case .japanese: "Japanese"
        case .korean: "Korean"
        case .cantonese: "Cantonese"
        case .arabic: "Arabic"
        case .persian: "Persian (Farsi)"
        case .hebrew: "Hebrew"
        }
    }

    /// ISO 639-1 or 639-3 language code
    public var isoCode: String {
        rawValue
    }

    /// Whether this language uses right-to-left script
    public var isRTL: Bool {
        switch self {
        case .arabic, .persian, .hebrew:
            true
        default:
            false
        }
    }

    /// Whether this language is currently implemented
    /// Parakeet TDT (English/European) and FastConformer (Arabic/Persian) are implemented
    public var isImplemented: Bool {
        switch self {
        case .arabic, .persian:
            true
        default:
            modelFamily == .parakeetTDT
        }
    }

    /// Languages that share the same model weights
    /// Useful for knowing which languages can switch instantly
    public var modelSiblings: [STTLanguage] {
        STTLanguage.allCases.filter { $0.modelIdentifier == self.modelIdentifier }
    }

    // MARK: - Factory Methods

    /// Get language from ISO code
    public static func fromCode(_ code: String) -> STTLanguage? {
        STTLanguage(rawValue: code.lowercased())
    }

    /// All implemented languages (ready to use)
    public static var implemented: [STTLanguage] {
        allCases.filter(\.isImplemented)
    }

    /// Languages grouped by model family
    public static var byFamily: [ModelFamily: [STTLanguage]] {
        Dictionary(grouping: allCases) { $0.modelFamily }
    }
}

// MARK: - CustomStringConvertible

extension STTLanguage: CustomStringConvertible {
    public var description: String {
        displayName
    }
}
