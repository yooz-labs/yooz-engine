// GrammarEngine.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "GrammarEngine")

/// Rust-based rule engine for grammar correction (English only).
///
/// Wraps the YoozTextCleanup Rust FFI via UniFFI bindings. Uses NLTagger
/// for POS tagging, then applies Rust-side grammar rules. Caches rule
/// counts on init and provides tier-based correction paths.
///
/// ## Tier System
/// - **Free**: ~200 rules (basic, grammar, articles, informal categories)
/// - **Pro**: 1,355+ rules including POS-based rules (all categories)
/// - **Premium**: Pro rules + LLM fallback (handled at TouchUp layer)
///
/// ## Correction Paths
/// - `check(text:categories:usePOS:)`: API-facing, category-based
/// - `correct(_:tier:usePOS:)`: Tier-based convenience for internal use
actor GrammarEngine {

    // MARK: - Singleton

    static let shared = GrammarEngine()

    // MARK: - Properties

    private let tagger = NLTaggerBridge()

    /// Whether grammar correction is available (FFI loaded and rules present).
    /// Set during initialization; remains false if Rust library fails to load.
    nonisolated private(set) var isAvailable: Bool = false

    /// Cached rule counts (populated on init for diagnostics and health reporting)
    nonisolated private(set) var simpleRuleCount: UInt32 = 0
    nonisolated private(set) var posRuleCount: UInt32 = 0
    nonisolated private(set) var totalRuleCount: UInt32 = 0
    nonisolated private(set) var programmaticRuleCount: UInt32 = 0

    /// Number of grammar rules available (cached total).
    nonisolated var ruleCount: Int {
        Int(totalRuleCount)
    }

    /// Library version string from Rust FFI.
    nonisolated var version: String {
        guard isAvailable else { return "unavailable" }
        return getVersion()
    }

    /// Available categories for English.
    nonisolated var availableCategories: [Category] {
        guard isAvailable else { return [] }
        return getAvailableCategoriesForLanguage(language: .english)
    }

    // MARK: - Initialization

    private init() {
        loadRuleCounts()

        #if DEBUG
        if isAvailable {
            verifyRulesWork()
        }
        #endif
    }

    /// Load rule counts from the Rust FFI with error handling.
    ///
    /// IMPORTANT: UniFFI-generated functions use `try!` internally, so FFI
    /// initialization failures will crash the app. If crashes occur in
    /// production, check: corrupted binary, platform incompatibility,
    /// memory issues.
    private nonisolated func loadRuleCounts() {
        // Safe defaults are already set by property initializers
        logger.info("Attempting FFI initialization...")

        // These FFI calls use try! internally and will crash if Rust library
        // fails to load. This is a UniFFI limitation.
        simpleRuleCount = getSimpleRuleCount(language: .english)
        posRuleCount = getPosRuleCount(language: .english)
        totalRuleCount = getRuleCount(language: .english)
        programmaticRuleCount = getProgrammaticRuleCount()

        if totalRuleCount > 0 {
            isAvailable = true
            logger.info(
                "Initialized with \(self.totalRuleCount) XML rules "
                + "(\(self.simpleRuleCount) simple + \(self.posRuleCount) POS) "
                + "+ \(self.programmaticRuleCount) programmatic rules"
            )
        } else {
            logger.error(
                "FFI succeeded but no rules loaded; grammar correction disabled. "
                + "simple=\(self.simpleRuleCount), pos=\(self.posRuleCount), "
                + "programmatic=\(self.programmaticRuleCount)"
            )
        }
    }

    // MARK: - Debug Verification

    #if DEBUG
    private nonisolated func verifyRulesWork() {
        let allCats = getAllCategories()

        // Subject-verb agreement (XML rules)
        let t1In = "I are happy"
        let t1Out = correctGrammarFullSimple(text: t1In, language: .english, categories: allCats)
        logger.debug("Test SVA: '\(t1In)' -> '\(t1Out)'")

        // Repeated consecutive words (programmatic rules)
        let t2In = "we need to to understand"
        let t2Out = correctGrammarFullSimple(text: t2In, language: .english, categories: allCats)
        logger.debug("Test WORD_REPEAT: '\(t2In)' -> '\(t2Out)'")

        // Number conversion
        let t3In = "I have twenty five items"
        let t3Out = correctGrammarFullSimple(text: t3In, language: .english, categories: [.numbers])
        logger.debug("Test NUMBERS: '\(t3In)' -> '\(t3Out)' [expected: 25]")

        let ruleNames = getProgrammaticRuleNames()
        logger.debug("Programmatic rules: \(ruleNames)")
    }
    #endif

    // MARK: - Grammar Checking (API-facing)

    /// Check and correct grammar in text (English only).
    ///
    /// This is the primary API-facing entry point. Clients send category names
    /// and an optional `usePOS` flag. When `usePOS` is true, NLTagger provides
    /// accurate POS tags enabling the full 1,355+ rule set.
    ///
    /// - Parameters:
    ///   - text: Input text to check.
    ///   - categories: Optional category names to restrict which rules apply.
    ///                 If nil or empty, all categories are used.
    ///   - usePOS: Whether to use NLTagger POS tagging (more accurate, slightly slower).
    ///             Defaults to true. When false, uses heuristic POS from Rust side.
    /// - Returns: Corrected text and number of corrections applied.
    func check(
        text: String,
        categories: [String]?,
        usePOS: Bool = true
    ) -> (result: String, correctionsApplied: Int) {
        guard !text.isEmpty else {
            return (result: text, correctionsApplied: 0)
        }
        guard isAvailable else {
            logger.warning("Grammar check skipped; rules not loaded")
            return (result: text, correctionsApplied: 0)
        }

        let resolvedCategories = resolveCategories(categories)
        let corrected: String

        if usePOS {
            let tokens = tagger.tokenize(text)
            corrected = correctGrammarFull(
                tokens: tokens,
                text: text,
                language: .english,
                categories: resolvedCategories
            )
        } else {
            corrected = correctGrammarFullSimple(
                text: text,
                language: .english,
                categories: resolvedCategories
            )
        }

        let corrections = text == corrected
            ? 0
            : countDifferences(original: text, corrected: corrected)
        logger.debug("Grammar check: \(corrections) correction(s) applied")
        return (result: corrected, correctionsApplied: corrections)
    }

    // MARK: - Tier-Based Correction (Internal Use)

    /// Tier identifiers for grammar rule gating.
    ///
    /// The engine itself is tier-agnostic at the API level (clients send
    /// categories). This enum exists for internal convenience when the
    /// engine needs tier-aware defaults (e.g., TouchUp pipeline integration).
    enum Tier: String, Sendable {
        /// ~200 rules (basic, grammar, articles, informal)
        case free
        /// All XML + POS rules (~1,355)
        case pro
        /// Pro + LLM fallback (handled at TouchUp layer, not here)
        case premium
    }

    /// Apply grammar correction based on tier.
    ///
    /// Convenience for internal callers (e.g., TouchUp pipeline) that know
    /// the user's tier but not the specific categories.
    ///
    /// - Parameters:
    ///   - text: Input text to correct.
    ///   - tier: Subscription tier controlling which rules apply.
    ///   - usePOS: Use NLTagger POS tagging. Defaults to true for pro/premium.
    /// - Returns: Corrected text.
    func correct(_ text: String, tier: Tier, usePOS: Bool = true) -> String {
        guard !text.isEmpty else { return text }
        guard isAvailable else {
            logger.warning("Grammar correction skipped; rules not loaded")
            return text
        }

        let categories = categoriesForTier(tier)

        if usePOS, tier != .free {
            let tokens = tagger.tokenize(text)
            return correctGrammarFull(
                tokens: tokens,
                text: text,
                language: .english,
                categories: categories
            )
        } else {
            return correctGrammarFullSimple(
                text: text,
                language: .english,
                categories: categories
            )
        }
    }

    /// Map tier to rule categories.
    func categoriesForTier(_ tier: Tier) -> [Category] {
        switch tier {
        case .free:
            return getFreeCategories()
        case .pro, .premium:
            return getAllCategories()
        }
    }

    // MARK: - Helpers

    private func resolveCategories(_ names: [String]?) -> [Category] {
        guard let names = names, !names.isEmpty else {
            return getAllCategories()
        }
        var resolved: [Category] = []
        for name in names {
            switch name.lowercased() {
            case "basic": resolved.append(.basic)
            case "grammar": resolved.append(.grammar)
            case "articles": resolved.append(.articles)
            case "informal": resolved.append(.informal)
            case "verbs": resolved.append(.verbs)
            case "numbers": resolved.append(.numbers)
            case "punctuation": resolved.append(.punctuation)
            case "style": resolved.append(.style)
            case "advanced": resolved.append(.advanced)
            default:
                logger.warning("Unknown grammar category: \(name)")
            }
        }
        if resolved.isEmpty {
            logger.warning("All \(names.count) requested categories were invalid, using all categories")
            return getAllCategories()
        }
        return resolved
    }

    /// Estimate number of corrections by counting word-level differences.
    private func countDifferences(original: String, corrected: String) -> Int {
        let origWords = original.split(separator: " ")
        let corrWords = corrected.split(separator: " ")
        if origWords == corrWords { return 0 }
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
