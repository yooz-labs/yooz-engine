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
/// - **Free**: Subset of rules (basic, grammar, articles, informal categories)
/// - **Pro**: All XML + POS-based rules (all categories)
/// - **Premium**: Pro rules + LLM fallback (handled at TouchUp layer)
/// Actual rule counts are cached in `counts` and reported via `ruleCount`.
///
/// ## Correction Paths
/// - `check(text:categories:usePOS:)`: API-facing, category-based
/// - `correct(_:tier:usePOS:)`: Tier-based convenience for internal use
public actor GrammarEngine {

    // MARK: - Singleton

    public static let shared = GrammarEngine()

    // MARK: - Properties

    private let tagger = NLTaggerBridge()

    /// Cached rule counts from Rust FFI, set once during init.
    private struct RuleCounts {
        let simple: UInt32
        let pos: UInt32
        let total: UInt32
        let programmatic: UInt32
        var isAvailable: Bool { total > 0 }
    }

    private nonisolated let counts: RuleCounts

    /// Whether grammar correction is available (FFI loaded and rules present).
    public nonisolated var isAvailable: Bool { counts.isAvailable }

    public nonisolated var simpleRuleCount: UInt32 { counts.simple }
    public nonisolated var posRuleCount: UInt32 { counts.pos }
    public nonisolated var totalRuleCount: UInt32 { counts.total }
    public nonisolated var programmaticRuleCount: UInt32 { counts.programmatic }

    /// Number of grammar rules available (cached total).
    public nonisolated var ruleCount: Int { Int(counts.total) }

    /// Library version string from Rust FFI.
    public nonisolated var version: String {
        guard isAvailable else { return "unavailable" }
        return getVersion()
    }

    /// Available categories for English.
    public nonisolated var availableCategories: [Category] {
        guard isAvailable else { return [] }
        return getAvailableCategoriesForLanguage(language: .english)
    }

    // MARK: - Initialization

    private init() {
        counts = Self.loadRuleCounts()

        #if DEBUG
        if isAvailable {
            verifyRulesWork()
        }
        #endif
    }

    /// Load rule counts from the Rust FFI.
    ///
    /// IMPORTANT: UniFFI-generated functions use `try!` internally, so FFI
    /// initialization failures will crash the app. If crashes occur in
    /// production, check: corrupted binary, platform incompatibility,
    /// memory issues.
    private static func loadRuleCounts() -> RuleCounts {
        logger.info("Attempting FFI initialization...")

        // These FFI calls use try! internally and will crash if Rust library
        // fails to load. This is a UniFFI limitation.
        let simple = getSimpleRuleCount(language: .english)
        let pos = getPosRuleCount(language: .english)
        let total = getRuleCount(language: .english)
        let programmatic = getProgrammaticRuleCount()

        let counts = RuleCounts(simple: simple, pos: pos, total: total, programmatic: programmatic)

        if counts.isAvailable {
            logger.info("Initialized with \(total) XML rules (\(simple) simple + \(pos) POS) + \(programmatic) programmatic rules")
        } else {
            logger.error("FFI succeeded but no rules loaded; grammar correction disabled. simple=\(simple), pos=\(pos), programmatic=\(programmatic)")
        }

        return counts
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
    public func check(
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

    /// Check grammar and return structured per-match data alongside the
    /// corrected text.
    ///
    /// Same correction behavior as ``check(text:categories:usePOS:)``; the
    /// `result` and `correctionsApplied` values are identical for the same
    /// inputs. Additionally returns `matches`: one ``GrammarMatch`` per
    /// contiguous edit, with `offset` / `length` in UTF-16 code units of the
    /// ORIGINAL text.
    ///
    /// Matches are derived from a token-aligned diff of original vs. corrected
    /// text (the matcher's FFI exposes only the corrected string). Offsets,
    /// lengths, original substrings, and replacements are exact; rule
    /// identity / category / message are best-effort. See ``GrammarMatch`` for
    /// the full rationale and limitations.
    ///
    /// - Parameters:
    ///   - text: Input text to check.
    ///   - categories: Optional category names restricting which rules apply.
    ///   - usePOS: Whether to use NLTagger POS tagging. Defaults to true.
    /// - Returns: Corrected text, correction count, and structured matches.
    public func checkDetailed(
        text: String,
        categories: [String]?,
        usePOS: Bool = true
    ) -> (result: String, correctionsApplied: Int, matches: [GrammarMatch]) {
        let base = check(text: text, categories: categories, usePOS: usePOS)
        let matches = GrammarMatchExtractor.matches(
            original: text,
            corrected: base.result
        )
        return (result: base.result, correctionsApplied: base.correctionsApplied, matches: matches)
    }

    // MARK: - Tier-Based Correction (Internal Use)

    /// Tier identifiers for grammar rule gating.
    ///
    /// The engine itself is tier-agnostic at the API level (clients send
    /// categories). This enum exists for internal convenience when the
    /// engine needs tier-aware defaults (e.g., TouchUp pipeline integration).
    public enum Tier: String, Sendable {
        /// Subset of rules (basic, grammar, articles, informal)
        case free
        /// All XML + POS rules (all categories)
        case pro
        /// Currently identical to pro; LLM fallback handled at TouchUp layer
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
    ///   - usePOS: Use NLTagger POS tagging. Defaults to true. Ignored for free tier (always uses simple rules).
    /// - Returns: Corrected text.
    public func correct(_ text: String, tier: Tier, usePOS: Bool = true) -> String {
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
    public func categoriesForTier(_ tier: Tier) -> [Category] {
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
