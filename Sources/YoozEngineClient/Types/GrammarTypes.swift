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

    public init(result: String, correctionsApplied: Int, ruleCount: Int?) {
        self.result = result
        self.correctionsApplied = correctionsApplied
        self.ruleCount = ruleCount
    }
}

/// Grammar tier identifiers matching engine-side tiers.
/// Clients use these to request tier-appropriate category sets.
/// Actual rule counts are reported by the server in GrammarCheckResponse.ruleCount.
public enum GrammarTier: String, Codable, Sendable {
    /// Subset of rules (basic, grammar, articles, informal categories)
    case free
    /// All XML + POS rules (all categories)
    case pro
    /// Pro rules + LLM fallback (currently identical to pro; LLM handled at TouchUp layer)
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
