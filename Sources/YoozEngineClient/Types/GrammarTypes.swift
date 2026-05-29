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
    /// Structured per-match data, when the server reports it.
    ///
    /// `nil` when talking to an older server that predates this field, so old
    /// clients and old servers keep interoperating. Each match carries an
    /// `offset` / `length` in UTF-16 code units of the ORIGINAL request text.
    public let matches: [GrammarMatch]?

    public init(
        result: String,
        correctionsApplied: Int,
        ruleCount: Int?,
        matches: [GrammarMatch]? = nil
    ) {
        self.result = result
        self.correctionsApplied = correctionsApplied
        self.ruleCount = ruleCount
        self.matches = matches
    }
}

/// A single structured grammar correction anchored to the original text.
///
/// `offset` / `length` are **UTF-16 code units in the ORIGINAL text** (NSString
/// semantics), so macOS Accessibility consumers can map the range directly onto
/// an `NSRange`. `replacement` is empty for deletions; `original` is empty for
/// pure insertions.
public struct GrammarMatch: Codable, Sendable, Equatable {
    /// Start of the matched range, in UTF-16 code units of the original text.
    public let offset: Int
    /// Length of the matched range, in UTF-16 code units of the original text.
    /// Zero for pure insertions.
    public let length: Int
    /// The matched substring of the original text. Empty for pure insertions.
    public let original: String
    /// The suggested replacement text. Empty for deletions.
    public let replacement: String
    /// Rule identifier.
    public let ruleId: String
    /// Rule category.
    public let category: String
    /// Human-readable explanation of the suggestion.
    public let message: String
    /// Optional terse variant of `message`.
    public let shortMessage: String?

    public init(
        offset: Int,
        length: Int,
        original: String,
        replacement: String,
        ruleId: String,
        category: String,
        message: String,
        shortMessage: String? = nil
    ) {
        self.offset = offset
        self.length = length
        self.original = original
        self.replacement = replacement
        self.ruleId = ruleId
        self.category = category
        self.message = message
        self.shortMessage = shortMessage
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
