import Foundation

// `GrammarCheckRequest` / `GrammarCheckResponse` moved to `YoozEngineWire`
// (#225) — visible here via `YoozEngineClient`'s `WireReexport.swift`.

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
