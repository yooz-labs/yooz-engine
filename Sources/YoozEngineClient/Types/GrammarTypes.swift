import Foundation

public struct GrammarCheckRequest: Codable, Sendable {
    public let text: String
    public let categories: [String]?
    /// Use NLTagger POS tagging for more accurate correction.
    /// Defaults to true server-side when nil.
    public let usePOS: Bool?

    public init(text: String, categories: [String]? = nil, usePOS: Bool? = nil) {
        self.text = text
        self.categories = categories
        self.usePOS = usePOS
    }
}

public struct GrammarCheckResponse: Codable, Sendable {
    public let result: String
    public let correctionsApplied: Int
    /// Total rule count used for this check (nil if server does not report it).
    public let ruleCount: Int?
}

/// Grammar tier identifiers matching engine-side tiers.
/// Clients use these to request tier-appropriate category sets.
public enum GrammarTier: String, Codable, Sendable {
    /// ~200 rules (basic, grammar, articles, informal)
    case free
    /// All XML + POS rules (~1,355)
    case pro
    /// Pro + LLM fallback
    case premium
}

/// Free-tier category names for client-side tier mapping.
public let grammarFreeCategories: [String] = [
    "basic", "grammar", "articles", "informal"
]

/// All category names (pro/premium tier).
public let grammarAllCategories: [String] = [
    "basic", "grammar", "articles", "informal",
    "verbs", "numbers", "punctuation", "style", "advanced"
]
