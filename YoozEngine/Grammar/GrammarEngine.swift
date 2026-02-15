// GrammarEngine.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "GrammarEngine")

/// Rust-based rule engine for grammar correction.
///
/// Wraps the YoozTextCleanup Rust FFI via UniFFI bindings. Uses NLTagger
/// for POS tagging, then applies 1,560+ rules in <1ms.
actor GrammarEngine {

    // MARK: - Singleton

    static let shared = GrammarEngine()

    // MARK: - Properties

    private let tagger = NLTaggerBridge()

    /// Grammar module is always available (no model loading needed).
    nonisolated var isAvailable: Bool { true }

    /// Number of rules for the given language (defaults to English).
    var ruleCount: Int {
        Int(getRuleCount(language: .english))
    }

    // MARK: - Grammar Checking

    /// Check and correct grammar in text.
    ///
    /// - Parameters:
    ///   - text: Input text to check.
    ///   - categories: Optional category names to restrict which rules apply.
    ///                 If nil or empty, all categories are used.
    /// - Returns: Corrected text and number of corrections applied.
    func check(text: String, categories: [String]?) -> (result: String, correctionsApplied: Int) {
        let resolvedCategories = resolveCategories(categories)
        let tokens = tagger.tokenize(text)
        let corrected = correctGrammarFull(
            tokens: tokens,
            text: text,
            language: .english,
            categories: resolvedCategories
        )
        let corrections = text == corrected ? 0 : countDifferences(original: text, corrected: corrected)
        logger.debug("Grammar check: \(corrections) correction(s) applied")
        return (result: corrected, correctionsApplied: corrections)
    }

    // MARK: - Helpers

    private func resolveCategories(_ names: [String]?) -> [Category] {
        guard let names = names, !names.isEmpty else {
            return getAllCategories()
        }
        return names.compactMap { name in
            switch name.lowercased() {
            case "basic": return .basic
            case "grammar": return .grammar
            case "articles": return .articles
            case "informal": return .informal
            case "verbs": return .verbs
            case "numbers": return .numbers
            case "punctuation": return .punctuation
            case "style": return .style
            case "advanced": return .advanced
            default:
                logger.warning("Unknown grammar category: \(name)")
                return nil
            }
        }
    }

    /// Estimate number of corrections by counting word-level differences.
    private func countDifferences(original: String, corrected: String) -> Int {
        let origWords = original.split(separator: " ")
        let corrWords = corrected.split(separator: " ")
        if origWords == corrWords { return 0 }
        // Simple heuristic: count of differing words
        let maxLen = max(origWords.count, corrWords.count)
        var diffs = abs(origWords.count - corrWords.count)
        for i in 0..<min(origWords.count, corrWords.count) {
            if origWords[i] != corrWords[i] {
                diffs += 1
            }
        }
        return max(1, min(diffs, maxLen))
    }
}
