// NLTagger Bridge for Yooz Text Cleanup
// Provides POS-tagged tokens using Apple's NLTagger for use with POS-based rules

import Foundation
import NaturalLanguage

/// Swift-side POS tagger using NLTagger
/// Converts NLTagger output to Yooz POSToken format
public class NLTaggerBridge {
    private let tagger: NLTagger

    public init() {
        self.tagger = NLTagger(tagSchemes: [.lexicalClass])
    }

    /// Tokenize text and get POS tags using NLTagger
    /// Returns an array of POSTokens that can be passed to Rust
    /// Uses .joinContractions to keep "don't", "I'm", etc. as single tokens
    public func tokenize(_ text: String) -> [PosToken] {
        var tokens: [PosToken] = []

        // Add sentence start marker
        tokens.append(PosToken(
            text: "",
            tag: .sentenceStart,
            start: 0,
            end: 0
        ))

        tagger.string = text
        let range = text.startIndex..<text.endIndex

        // Use joinContractions to keep "don't" as one token instead of "do" + "n't"
        let options: NLTagger.Options = [.omitWhitespace, .joinContractions]

        tagger.enumerateTags(in: range, unit: .word, scheme: .lexicalClass, options: options) { tag, tokenRange in
            let word = String(text[tokenRange])
            let posTag = self.convertNLTaggerTag(tag)

            // Calculate byte offsets
            let start = text.distance(from: text.startIndex, to: tokenRange.lowerBound)
            let end = text.distance(from: text.startIndex, to: tokenRange.upperBound)

            tokens.append(PosToken(
                text: word,
                tag: posTag,
                start: UInt32(start),
                end: UInt32(end)
            ))

            return true
        }

        // Add sentence end marker
        let endPos = text.count
        tokens.append(PosToken(
            text: "",
            tag: .sentenceEnd,
            start: UInt32(endPos),
            end: UInt32(endPos)
        ))

        return tokens
    }

    /// Convert NLTagger.Tag to Yooz POSTag
    private func convertNLTaggerTag(_ tag: NLTag?) -> PosTag {
        guard let tag = tag else { return .unknown }

        switch tag {
        case .noun:
            return .noun
        case .verb:
            return .verb
        case .adjective:
            return .adjective
        case .adverb:
            return .adverb
        case .pronoun:
            return .pronoun
        case .determiner:
            return .determiner
        case .preposition:
            return .preposition
        case .conjunction:
            return .conjunction
        case .particle:
            return .particle
        case .number:
            return .number
        case .interjection:
            return .interjection
        case .punctuation, .sentenceTerminator, .openQuote, .closeQuote,
             .openParenthesis, .closeParenthesis, .wordJoiner, .dash:
            return .punctuation
        default:
            return .unknown
        }
    }
}

// MARK: - Convenience extensions

extension NLTaggerBridge {
    /// Tokenize and return as string representation for debugging
    public func tokenizeDebug(_ text: String) -> String {
        let tokens = tokenize(text)
        return tokens.map { "\($0.text)/\($0.tag)" }.joined(separator: " ")
    }
}

// MARK: - Global convenience function

/// Global function to tokenize text with NLTagger
/// Returns POSTokens for use with POS-aware grammar correction
public func tokenizeWithNLTagger(_ text: String) -> [PosToken] {
    let bridge = NLTaggerBridge()
    return bridge.tokenize(text)
}
