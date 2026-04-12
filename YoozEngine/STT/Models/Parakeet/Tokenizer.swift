// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Simple BPE tokenizer for Parakeet
/// Handles token-to-text conversion using the model vocabulary
public struct Tokenizer {
    public let vocabulary: [String]

    public init(vocabulary: [String]) {
        self.vocabulary = vocabulary
    }

    /// Decode a single token to text
    /// - Parameter token: Token ID
    /// - Returns: Decoded text with ▁ replaced by space
    public func decode(token: Int) -> String {
        guard token >= 0, token < vocabulary.count else {
            return ""
        }
        return vocabulary[token].replacingOccurrences(of: "▁", with: " ")
    }

    /// Decode multiple tokens to text
    /// - Parameter tokens: Array of token IDs
    /// - Returns: Concatenated decoded text
    public func decode(tokens: [Int]) -> String {
        tokens.map { decode(token: $0) }.joined()
    }
}
