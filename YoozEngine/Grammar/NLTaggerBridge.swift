// NLTaggerBridge.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import NaturalLanguage

/// Swift-side POS tagger using Apple's NLTagger.
///
/// Converts NLTagger output to ``PosToken`` values that the Rust grammar
/// engine understands. Uses ``createPosToken`` FFI call to construct tokens,
/// ensuring they match the Rust-side representation exactly.
///
/// Thread safety: instances are **not** thread-safe. The ``GrammarEngine``
/// actor owns a single instance, so access is serialized by the actor.
final class NLTaggerBridge {
    private let tagger: NLTagger

    init() {
        self.tagger = NLTagger(tagSchemes: [.lexicalClass])
    }

    /// Tokenize text and produce POS-tagged tokens for the Rust engine.
    ///
    /// Uses `.joinContractions` to keep "don't", "I'm", etc. as single tokens.
    /// Wraps the token sequence with sentenceStart/sentenceEnd markers.
    func tokenize(_ text: String) -> [PosToken] {
        var tokens: [PosToken] = []

        // Sentence start marker
        tokens.append(createPosToken(text: "", tag: .sentenceStart, start: 0, end: 0))

        tagger.string = text
        let range = text.startIndex..<text.endIndex
        let options: NLTagger.Options = [.omitWhitespace, .joinContractions]

        tagger.enumerateTags(in: range, unit: .word, scheme: .lexicalClass, options: options) { tag, tokenRange in
            let word = String(text[tokenRange])
            let posTag = self.convertNLTaggerTag(tag)

            let start = text.distance(from: text.startIndex, to: tokenRange.lowerBound)
            let end = text.distance(from: text.startIndex, to: tokenRange.upperBound)

            tokens.append(createPosToken(
                text: word,
                tag: posTag,
                start: UInt32(start),
                end: UInt32(end)
            ))

            return true
        }

        // Sentence end marker
        let endPos = text.count
        tokens.append(createPosToken(text: "", tag: .sentenceEnd, start: UInt32(endPos), end: UInt32(endPos)))

        return tokens
    }

    /// Convert NLTagger.Tag to Rust PosTag enum.
    private func convertNLTaggerTag(_ tag: NLTag?) -> PosTag {
        guard let tag else { return .unknown }

        switch tag {
        case .noun: return .noun
        case .verb: return .verb
        case .adjective: return .adjective
        case .adverb: return .adverb
        case .pronoun: return .pronoun
        case .determiner: return .determiner
        case .preposition: return .preposition
        case .conjunction: return .conjunction
        case .particle: return .particle
        case .number: return .number
        case .interjection: return .interjection
        case .punctuation, .sentenceTerminator, .openQuote, .closeQuote,
             .openParenthesis, .closeParenthesis, .wordJoiner, .dash:
            return .punctuation
        default: return .unknown
        }
    }
}

// MARK: - Debug

extension NLTaggerBridge {
    /// Tokenize and return as "word/tag" string for debugging.
    func tokenizeDebug(_ text: String) -> String {
        tokenize(text).map { "\($0.text)/\($0.tag)" }.joined(separator: " ")
    }
}
