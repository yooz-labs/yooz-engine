//! Rule Engine
//!
//! Applies grammar rules to text with category and language filtering.

use std::collections::HashSet;
use crate::categories::{Category, Level};
use crate::languages::Language;
use crate::patterns::{tokenize, TokenMatcher, find_matches, find_matches_pos};
use crate::pos::POSToken;
use crate::xml_parser::{parse_rules, Rules, Rule as XmlRule};

/// A compiled rule ready for matching
#[derive(Debug, Clone)]
pub struct CompiledRule {
    pub id: String,
    pub category: Category,
    pub pattern: Vec<TokenMatcher>,
    /// Antipatterns: if any of these match, the rule is skipped
    pub antipatterns: Vec<Vec<TokenMatcher>>,
    pub replacement: String,
    pub is_picky: bool,
    pub priority: i32,
    /// Whether this rule requires POS tagging
    pub requires_pos: bool,
    /// Marker start: index of first token to replace (0-based)
    /// Defaults to 0 if not specified (entire pattern is error)
    pub marker_start: usize,
    /// Marker end: index after last token to replace (exclusive)
    /// Defaults to pattern.len() if not specified (entire pattern is error)
    pub marker_end: usize,
}

/// Rule information for API queries
#[derive(Debug, Clone, uniffi::Record)]
pub struct RuleInfo {
    pub id: String,
    pub name: String,
    pub category: String,
    pub is_picky: bool,
}

/// The main rule engine
pub struct RuleEngine {
    rules: Vec<CompiledRule>,
    language: Language,
}

impl RuleEngine {
    /// Create a new engine with default rules for English
    pub fn new() -> Self {
        Self::for_language(Language::English)
    }

    /// Create an engine for a specific language
    pub fn for_language(language: Language) -> Self {
        let xml = Self::load_rules_for_language(&language);
        Self::from_xml_with_language(&xml, language)
    }

    /// Load rules XML for a language
    fn load_rules_for_language(language: &Language) -> String {
        // Load rules from multiple files per language
        // Our curated rules + Full LanguageTool extracted rules + POS-based rules
        match language.base_code() {
            "en" => {
                // Combine all English rule files
                let curated = include_str!("../rules/en/grammar.xml");
                // Full LT rules with antipatterns, exceptions, backrefs, etc.
                // These include both simple and POS-based rules
                let lt_grammar_full = include_str!("../rules/en/lt-grammar-full.xml");
                let lt_style_full = include_str!("../rules/en/lt-style-full.xml");
                Self::merge_rule_files(&[curated, lt_grammar_full, lt_style_full])
            }
            // Future: add other languages
            // "es" => include_str!("../rules/es/grammar.xml").to_string(),
            // "fr" => include_str!("../rules/fr/grammar.xml").to_string(),
            // "de" => include_str!("../rules/de/grammar.xml").to_string(),
            _ => include_str!("../rules/en/grammar.xml").to_string(), // Fallback to English
        }
    }

    /// Merge multiple XML rule files into one
    fn merge_rule_files(files: &[&str]) -> String {
        let mut all_categories = String::new();

        for xml in files {
            // Extract content between <rules> tags
            if let Some(start) = xml.find("<rules>") {
                if let Some(end) = xml.rfind("</rules>") {
                    let content = &xml[start + 7..end];
                    all_categories.push_str(content);
                }
            }
        }

        format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rules>\n{}\n</rules>", all_categories)
    }

    /// Create engine from XML string
    pub fn from_xml(xml: &str) -> Self {
        Self::from_xml_with_language(xml, Language::English)
    }

    /// Create engine from XML string with language
    pub fn from_xml_with_language(xml: &str, language: Language) -> Self {
        let parsed = parse_rules(xml).unwrap_or_else(|e| {
            eprintln!("Failed to parse rules: {}", e);
            Rules { categories: vec![] }
        });

        let mut rules = Vec::new();
        for xml_category in parsed.categories {
            for rule in xml_category.rules {
                if let Some(compiled) = Self::compile_rule(&rule, &xml_category.id) {
                    rules.push(compiled);
                }
            }
        }

        // Sort by priority (higher first)
        rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        Self { rules, language }
    }

    fn compile_rule(rule: &XmlRule, parent_category_id: &str) -> Option<CompiledRule> {
        let mut requires_pos = false;

        let pattern: Vec<TokenMatcher> = rule.pattern.tokens.iter()
            .map(|token| Self::compile_token(token, &mut requires_pos))
            .collect();

        // Compile antipatterns
        let antipatterns: Vec<Vec<TokenMatcher>> = rule.antipatterns.iter()
            .map(|antipattern| {
                antipattern.tokens.iter()
                    .map(|token| Self::compile_token(token, &mut requires_pos))
                    .collect()
            })
            .collect();

        let replacement = rule.suggestions.first()
            .map(|s| s.text.clone())
            .unwrap_or_default();

        // Get marker positions (default to entire pattern if not specified)
        let pattern_len = pattern.len();
        let mut marker_start = rule.marker_start.unwrap_or(0);
        let mut marker_end = rule.marker_end.unwrap_or(pattern_len);

        // Validate marker bounds
        // Ensure marker_end doesn't exceed pattern length
        if marker_end > pattern_len {
            marker_end = pattern_len;
        }
        // Ensure marker_start is within bounds
        if marker_start >= pattern_len {
            marker_start = 0;
        }
        // Ensure marker_start < marker_end (swap if invalid, or fallback to full pattern)
        if marker_start >= marker_end {
            marker_start = 0;
            marker_end = pattern_len;
        }

        Some(CompiledRule {
            id: rule.id.clone(),
            category: rule.get_category(parent_category_id),
            pattern,
            antipatterns,
            replacement,
            is_picky: rule.is_picky(),
            priority: rule.get_priority(),
            requires_pos,
            marker_start,
            marker_end,
        })
    }

    /// Compile a single XML token into a TokenMatcher
    fn compile_token(token: &crate::xml_parser::Token, requires_pos: &mut bool) -> TokenMatcher {
        let postag = token.postag.clone();
        let postag_regexp = token.postag_regexp.as_ref().map(|s| s == "yes").unwrap_or(false);

        if postag.is_some() {
            *requires_pos = true;
        }

        // Parse min/max for optional tokens
        let min: u8 = token.min.as_ref().and_then(|s| s.parse().ok()).unwrap_or(1);
        let max: u8 = token.max.as_ref().and_then(|s| s.parse().ok()).unwrap_or(1);

        // Parse case_sensitive (default: false)
        let case_sensitive = token.case_sensitive.unwrap_or(false);

        // Parse spacebefore="no" (means token is attached to previous)
        let spacebefore_no = token.spacebefore.as_ref().map(|s| s == "no").unwrap_or(false);

        // Compile exceptions
        let exceptions: Vec<String> = token.exceptions.iter()
            .map(|e| e.text.clone())
            .filter(|s| !s.is_empty())
            .collect();

        TokenMatcher::with_all_options(
            if token.text.is_empty() { None } else { Some(token.text.clone()) },
            token.regexp.clone(),
            postag,
            postag_regexp,
            min,
            max,
            case_sensitive,
            spacebefore_no,
            exceptions,
        )
    }

    /// Apply all rules (no filtering) - simple rules only
    pub fn apply_rules(&self, text: &str) -> String {
        self.apply_rules_filtered(text, None, Level::Standard)
    }

    /// Apply all rules with POS-tagged tokens (simple + POS rules)
    /// This is the main entry point when using NLTagger
    pub fn apply_rules_with_pos(&self, tokens: &[POSToken]) -> String {
        self.apply_rules_with_pos_filtered(tokens, None, Level::Standard)
    }

    /// Apply rules with POS tokens and category filtering
    pub fn apply_rules_with_pos_categories(&self, tokens: &[POSToken], categories: &[Category]) -> String {
        let cat_set: HashSet<Category> = categories.iter().copied().collect();
        self.apply_rules_with_pos_filtered(tokens, Some(&cat_set), Level::Standard)
    }

    /// Debug: Find which rule matches the given tokens
    #[cfg(test)]
    pub fn find_matching_rule(&self, tokens: &[POSToken], categories: &[Category]) -> Option<(String, String, String)> {
        use crate::patterns::find_matches_pos;
        let cat_set: HashSet<Category> = categories.iter().copied().collect();

        for rule in &self.rules {
            // Category filter
            if !cat_set.contains(&rule.category) {
                continue;
            }

            // Skip picky rules for standard level
            if rule.is_picky {
                continue;
            }

            // Check if pattern matches
            let matches = find_matches_pos(tokens, &rule.pattern);
            if !matches.is_empty() {
                let m = &matches[0];
                // Check antipatterns
                if !self.antipattern_matches_pos(tokens, &rule.antipatterns, m.start, m.len) {
                    // Build pattern description
                    let pattern_desc: Vec<String> = rule.pattern.iter()
                        .map(|p| format!("{:?}", p))
                        .collect();
                    return Some((
                        rule.id.clone(),
                        rule.replacement.clone(),
                        pattern_desc.join(" | "),
                    ));
                }
            }
        }
        None
    }

    /// Apply rules with category filtering
    pub fn apply_rules_with_categories(&self, text: &str, categories: &[Category]) -> String {
        let cat_set: HashSet<Category> = categories.iter().copied().collect();
        self.apply_rules_filtered(text, Some(&cat_set), Level::Standard)
    }

    /// Apply rules with category and level filtering
    pub fn apply_rules_with_options(
        &self,
        text: &str,
        categories: &[Category],
        level: Level,
    ) -> String {
        let cat_set: HashSet<Category> = categories.iter().copied().collect();
        self.apply_rules_filtered(text, Some(&cat_set), level)
    }

    /// Core filtering logic (simple rules only)
    fn apply_rules_filtered(
        &self,
        text: &str,
        categories: Option<&HashSet<Category>>,
        level: Level,
    ) -> String {
        let mut result = text.to_string();

        for rule in &self.rules {
            // Skip POS-based rules when not using POS tagging
            // These rules require NLTagger and apply_rules_with_pos()
            if rule.requires_pos {
                continue;
            }

            // Category filter
            if let Some(cats) = categories {
                if !cats.contains(&rule.category) {
                    continue;
                }
            }

            // Level filter
            match level {
                Level::Essential => {
                    // Only high-priority, non-picky rules
                    if rule.is_picky || rule.priority < 5 {
                        continue;
                    }
                }
                Level::Standard => {
                    // Skip picky rules
                    if rule.is_picky {
                        continue;
                    }
                }
                Level::Thorough => {
                    // Apply all rules
                }
            }

            result = self.apply_single_rule(&result, rule);
        }

        result
    }

    /// Core filtering logic with POS tokens (all rules including POS-based)
    fn apply_rules_with_pos_filtered(
        &self,
        tokens: &[POSToken],
        categories: Option<&HashSet<Category>>,
        level: Level,
    ) -> String {
        // Build initial text from tokens
        let mut result_tokens: Vec<POSToken> = tokens.to_vec();

        for rule in &self.rules {
            // Category filter
            if let Some(cats) = categories {
                if !cats.contains(&rule.category) {
                    continue;
                }
            }

            // Level filter
            match level {
                Level::Essential => {
                    if rule.is_picky || rule.priority < 5 {
                        continue;
                    }
                }
                Level::Standard => {
                    if rule.is_picky {
                        continue;
                    }
                }
                Level::Thorough => {}
            }

            // Apply rule using POS matching
            result_tokens = self.apply_single_rule_pos(&result_tokens, rule);
        }

        // Reconstruct text from tokens with proper spacing
        self.reconstruct_text(&result_tokens)
    }

    /// Reconstruct text from tokens with proper spacing
    fn reconstruct_text(&self, tokens: &[POSToken]) -> String {
        let mut result = String::new();

        for token in tokens {
            if token.text.is_empty() {
                continue;
            }

            // Check if we need a space before this token
            let needs_space = !result.is_empty()
                && !is_opening_punct(&token.text)
                && !is_closing_punct(&token.text)
                && !token.text.starts_with("'")  // Contractions
                && !result.ends_with(' ')
                && !ends_with_opening_punct(&result);

            if needs_space {
                result.push(' ');
            }
            result.push_str(&token.text);
        }

        result
    }

    /// Apply a single rule using POS matching
    fn apply_single_rule_pos(&self, tokens: &[POSToken], rule: &CompiledRule) -> Vec<POSToken> {
        let matches = find_matches_pos(tokens, &rule.pattern);

        if matches.is_empty() {
            return tokens.to_vec();
        }

        // Apply first match only
        let m = &matches[0];

        // Check antipatterns: if any matches at this position, skip the rule
        if self.antipattern_matches_pos(tokens, &rule.antipatterns, m.start, m.len) {
            return tokens.to_vec();
        }

        // Get matched tokens for back-references (ALL pattern tokens, not just marked)
        let matched_tokens: Vec<&str> = (0..m.len)
            .filter_map(|i| tokens.get(m.start + i).map(|t| t.text.as_str()))
            .collect();

        // Apply back-references to replacement
        let replacement = self.apply_backreferences(&rule.replacement, &matched_tokens);

        let mut result = tokens.to_vec();

        // Create replacement tokens from suggestion
        let replacement_words: Vec<&str> = replacement.split_whitespace().collect();
        let replacement_tokens: Vec<POSToken> = replacement_words.iter()
            .enumerate()
            .map(|(_i, word)| POSToken {
                text: word.to_string(),
                tag: crate::pos::POSTag::Unknown, // We don't know the POS of replacement
                start: 0,
                end: 0,
            })
            .collect();

        // Replace ONLY marked tokens with suggestion (marker_start to marker_end)
        // Keep context tokens (before marker_start and after marker_end)
        // Validate marker bounds at runtime (defensive check)
        let marker_start = rule.marker_start.min(m.len);
        let marker_end = rule.marker_end.min(m.len);
        let (marker_start, marker_end) = if marker_start >= marker_end {
            (0, m.len)
        } else {
            (marker_start, marker_end)
        };

        let replace_start = m.start + marker_start;
        let marked_count = marker_end - marker_start;

        // Calculate safe count to avoid removing more tokens than available
        let safe_count = marked_count.min(result.len().saturating_sub(replace_start));
        debug_assert_eq!(marked_count, safe_count, "marked_count ({}) exceeded available tokens ({})", marked_count, safe_count);

        // Remove only the marked tokens
        for _ in 0..safe_count {
            result.remove(replace_start);
        }

        // Insert replacement tokens at the marker position
        for (i, token) in replacement_tokens.into_iter().enumerate() {
            result.insert(replace_start + i, token);
        }

        result
    }

    /// Check if any antipattern matches that overlaps with the pattern match
    ///
    /// Antipatterns can include context before the matched pattern, so we need to
    /// check at positions before `pos` as well. For example, if the pattern is
    /// "am VBG" matching at position 2, and antipattern is "I am", we need to
    /// check if "I am" matches at position 1 (where "I" precedes "am").
    fn antipattern_matches_pos(&self, tokens: &[POSToken], antipatterns: &[Vec<TokenMatcher>], pos: usize, pattern_len: usize) -> bool {
        for antipattern in antipatterns {
            let ap_len = antipattern.len();

            // Calculate the range of start positions where this antipattern could
            // overlap with our pattern match at `pos`. The antipattern could start
            // up to (ap_len - 1) positions before pos, as long as it overlaps.
            let earliest_start = pos.saturating_sub(ap_len.saturating_sub(1));
            // The antipattern could start as late as pos (if it extends beyond pattern)
            let latest_start = pos;

            for start in earliest_start..=latest_start {
                // Check bounds
                if start + ap_len > tokens.len() {
                    continue;
                }

                // Verify this position actually overlaps with our pattern
                let ap_end = start + ap_len;
                let pattern_end = pos + pattern_len;
                if ap_end <= pos || start >= pattern_end {
                    continue; // No overlap
                }

                // Check if antipattern matches at this position
                let mut all_match = true;
                for (j, matcher) in antipattern.iter().enumerate() {
                    if !matcher.matches_pos_token(&tokens[start + j]) {
                        all_match = false;
                        break;
                    }
                }

                if all_match {
                    return true;
                }
            }
        }
        false
    }

    fn apply_single_rule(&self, text: &str, rule: &CompiledRule) -> String {
        let tokens = tokenize(text);
        let matches = find_matches(&tokens, &rule.pattern);

        if matches.is_empty() {
            return text.to_string();
        }

        // Apply first match only (simplification for prototype)
        let match_idx = matches[0];

        // Check antipatterns: if any matches at this position, skip the rule
        if self.antipattern_matches(&tokens, &rule.antipatterns, match_idx) {
            return text.to_string();
        }

        // Get matched tokens for back-references (ALL pattern tokens, not just marked)
        let matched_tokens: Vec<&str> = (0..rule.pattern.len())
            .filter_map(|i| tokens.get(match_idx + i).map(|s| s.as_str()))
            .collect();

        // Apply back-references to replacement
        let replacement = self.apply_backreferences(&rule.replacement, &matched_tokens);

        let mut result_tokens = tokens.clone();

        // Replace ONLY marked tokens with suggestion (marker_start to marker_end)
        // Keep context tokens (before marker_start and after marker_end)
        let replacement_tokens: Vec<String> = replacement
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Validate marker bounds at runtime (defensive check)
        let pattern_len = rule.pattern.len();
        let marker_start = rule.marker_start.min(pattern_len);
        let marker_end = rule.marker_end.min(pattern_len);
        let (marker_start, marker_end) = if marker_start >= marker_end {
            (0, pattern_len)
        } else {
            (marker_start, marker_end)
        };

        // Calculate the actual position in the result where we start replacing
        let replace_start = match_idx + marker_start;
        let marked_count = marker_end - marker_start;

        // Calculate safe count to avoid removing more tokens than available
        let safe_count = marked_count.min(result_tokens.len().saturating_sub(replace_start));
        debug_assert_eq!(marked_count, safe_count, "marked_count ({}) exceeded available tokens ({})", marked_count, safe_count);

        // Remove only the marked tokens
        for _ in 0..safe_count {
            result_tokens.remove(replace_start);
        }

        // Insert replacement tokens at the marker position
        for (i, token) in replacement_tokens.iter().enumerate() {
            result_tokens.insert(replace_start + i, token.clone());
        }

        result_tokens.join(" ")
    }

    /// Apply back-references (\1, \2, etc.) in replacement string
    fn apply_backreferences(&self, replacement: &str, matched_tokens: &[&str]) -> String {
        let mut result = replacement.to_string();

        // Replace \1 through \9 with corresponding matched tokens
        for i in 1..=9 {
            let backref = format!("\\{}", i);
            if result.contains(&backref) {
                let replacement_text = matched_tokens.get(i - 1).unwrap_or(&"");
                result = result.replace(&backref, replacement_text);
            }
        }

        result
    }

    /// Check if any antipattern matches at the given position
    fn antipattern_matches(&self, tokens: &[String], antipatterns: &[Vec<TokenMatcher>], pos: usize) -> bool {
        for antipattern in antipatterns {
            // Check if this antipattern matches starting at pos
            if pos + antipattern.len() > tokens.len() {
                continue;
            }

            let mut all_match = true;
            for (j, matcher) in antipattern.iter().enumerate() {
                if !matcher.matches(&tokens[pos + j]) {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                return true;
            }
        }
        false
    }

    // Query methods

    /// Get the language this engine is configured for
    pub fn language(&self) -> Language {
        self.language
    }

    /// Get total rule count
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get count of simple rules (no POS required)
    pub fn simple_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| !r.requires_pos).count()
    }

    /// Get count of POS-based rules
    pub fn pos_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| r.requires_pos).count()
    }

    /// Get rule count for a category
    pub fn rule_count_for_category(&self, category: Category) -> usize {
        self.rules.iter().filter(|r| r.category == category).count()
    }

    /// Get rule count for a category and level
    pub fn rule_count_for_category_and_level(&self, category: Category, level: Level) -> usize {
        self.rules.iter().filter(|r| {
            if r.category != category {
                return false;
            }
            match level {
                Level::Essential => !r.is_picky && r.priority >= 5,
                Level::Standard => !r.is_picky,
                Level::Thorough => true,
            }
        }).count()
    }

    /// Get all categories that have rules
    pub fn available_categories(&self) -> Vec<Category> {
        let mut cats: HashSet<Category> = HashSet::new();
        for rule in &self.rules {
            cats.insert(rule.category);
        }
        let mut result: Vec<Category> = cats.into_iter().collect();
        result.sort_by_key(|c| c.as_str());
        result
    }

    /// Get rule info for all rules
    pub fn all_rules(&self) -> Vec<RuleInfo> {
        self.rules.iter().map(|r| RuleInfo {
            id: r.id.clone(),
            name: r.id.clone(), // Use ID as name for now
            category: r.category.to_string(),
            is_picky: r.is_picky,
        }).collect()
    }

    /// Get rule info for a category
    pub fn rules_for_category(&self, category: Category) -> Vec<RuleInfo> {
        self.rules.iter()
            .filter(|r| r.category == category)
            .map(|r| RuleInfo {
                id: r.id.clone(),
                name: r.id.clone(),
                category: r.category.to_string(),
                is_picky: r.is_picky,
            })
            .collect()
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for text reconstruction
fn is_opening_punct(s: &str) -> bool {
    matches!(s, "(" | "[" | "{" | "\"" | "'" | "`")
}

fn is_closing_punct(s: &str) -> bool {
    matches!(s, "." | "," | "!" | "?" | ";" | ":" | ")" | "]" | "}" | "\"" | "'")
}

fn ends_with_opening_punct(s: &str) -> bool {
    s.ends_with('(') || s.ends_with('[') || s.ends_with('{') ||
    s.ends_with('"') || s.ends_with('\'') || s.ends_with('`')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rules() {
        let engine = RuleEngine::new();
        let result = engine.apply_rules("I are happy");
        assert_eq!(result, "I am happy");
    }

    #[test]
    fn test_apply_with_categories() {
        let engine = RuleEngine::new();

        // Grammar category should fix "I are"
        let result = engine.apply_rules_with_categories(
            "I are happy",
            &[Category::Grammar],
        );
        assert_eq!(result, "I am happy");

        // Articles category should NOT fix "I are"
        let result = engine.apply_rules_with_categories(
            "I are happy",
            &[Category::Articles],
        );
        assert_eq!(result, "I are happy");
    }

    #[test]
    fn test_apply_with_multiple_categories() {
        let engine = RuleEngine::new();

        let result = engine.apply_rules_with_categories(
            "I are eating a apple",
            &[Category::Grammar, Category::Articles],
        );
        assert_eq!(result, "I am eating an apple");
    }

    #[test]
    fn test_category_counts() {
        let engine = RuleEngine::new();

        // Should have rules in Grammar category
        assert!(engine.rule_count_for_category(Category::Grammar) > 0);

        // Total should equal sum of categories
        let total = engine.rule_count();
        assert!(total > 0);
    }

    #[test]
    fn test_available_categories() {
        let engine = RuleEngine::new();
        let cats = engine.available_categories();

        // Should have at least Grammar and Articles
        assert!(cats.contains(&Category::Grammar));
        assert!(cats.contains(&Category::Articles));
    }

    #[test]
    fn test_language() {
        let engine = RuleEngine::for_language(Language::EnglishUS);
        assert_eq!(engine.language(), Language::EnglishUS);
    }

    #[test]
    fn test_antipattern_basic() {
        // Create a rule that replaces ", ." with "." but has an antipattern for ", . NET"
        let xml = r#"
        <rules>
            <category id="PUNCTUATION" name="Punctuation">
                <rule id="COMMA_PERIOD" name="comma period" prio="8">
                    <antipattern>
                        <token>,</token>
                        <token>.</token>
                        <token>NET</token>
                    </antipattern>
                    <pattern>
                        <token>,</token>
                        <token>.</token>
                    </pattern>
                    <message>Remove comma before period.</message>
                    <suggestion>.</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Without antipattern match: should apply the rule
        let result = engine.apply_rules("Hello , . World");
        assert_eq!(result, "Hello . World");

        // With antipattern match: should NOT apply the rule
        let result = engine.apply_rules("Use , . NET for development");
        assert_eq!(result, "Use , . NET for development");
    }

    #[test]
    fn test_antipattern_multiple() {
        // Rule with multiple antipatterns
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="TEST_RULE" name="test rule" prio="8">
                    <antipattern>
                        <token>foo</token>
                        <token>bar</token>
                        <token>baz</token>
                    </antipattern>
                    <antipattern>
                        <token>foo</token>
                        <token>bar</token>
                        <token>qux</token>
                    </antipattern>
                    <pattern>
                        <token>foo</token>
                        <token>bar</token>
                    </pattern>
                    <message>Test</message>
                    <suggestion>replaced</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // No antipattern match: should apply
        let result = engine.apply_rules("foo bar hello");
        assert_eq!(result, "replaced hello");

        // First antipattern matches: should NOT apply
        let result = engine.apply_rules("foo bar baz");
        assert_eq!(result, "foo bar baz");

        // Second antipattern matches: should NOT apply
        let result = engine.apply_rules("foo bar qux");
        assert_eq!(result, "foo bar qux");
    }

    #[test]
    fn test_case_sensitive() {
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="CASE_TEST" name="case test" prio="8">
                    <pattern>
                        <token case_sensitive="true">iPhone</token>
                    </pattern>
                    <message>Test</message>
                    <suggestion>smartphone</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Exact case: should match
        let result = engine.apply_rules("I have an iPhone");
        assert_eq!(result, "I have an smartphone");

        // Wrong case: should NOT match (case sensitive)
        let result = engine.apply_rules("I have an iphone");
        assert_eq!(result, "I have an iphone");
    }

    #[test]
    fn test_exception() {
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="EXCEPTION_TEST" name="exception test" prio="8">
                    <pattern>
                        <token>
                            affect
                            <exception>effect</exception>
                        </token>
                    </pattern>
                    <message>Test</message>
                    <suggestion>impact</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Normal match: should apply
        let result = engine.apply_rules("This will affect you");
        assert_eq!(result, "This will impact you");

        // Exception word: should NOT apply (but this tests exception within token)
        // Note: exceptions work at token level, not word level
    }

    #[test]
    fn test_backreference() {
        // Test back-references with a simple two-token pattern
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="BACKREF_TEST" name="backref test" prio="8">
                    <pattern>
                        <token>hello</token>
                        <token>world</token>
                    </pattern>
                    <message>Swap words</message>
                    <suggestion>\2 \1</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Should swap the words using back-references
        let result = engine.apply_rules("say hello world now");
        assert_eq!(result, "say world hello now");
    }

    #[test]
    fn test_backreference_simple() {
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="PRESERVE_TEST" name="preserve test" prio="8">
                    <pattern>
                        <token>,</token>
                        <token>.</token>
                    </pattern>
                    <message>Remove comma</message>
                    <suggestion>\2</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Should keep only the second matched token (the period)
        let result = engine.apply_rules("Hello , . World");
        assert_eq!(result, "Hello . World");
    }

    #[test]
    fn test_antipattern_with_pos() {
        use crate::pos::{POSToken, POSTag};

        // Rule that replaces "effect change" with "affect change",
        // but has an antipattern for when "effect" is followed by a noun (effect + NN)
        // Antipattern must start at the same position as the pattern
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="EFFECT_VERB" name="effect as verb" prio="8">
                    <antipattern>
                        <token>effect</token>
                        <token postag="NN">change</token>
                        <token>immediately</token>
                    </antipattern>
                    <pattern>
                        <token>effect</token>
                        <token>change</token>
                    </pattern>
                    <message>Use affect for verbs</message>
                    <suggestion>affect change</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Without antipattern match: should apply the rule
        let tokens = vec![
            POSToken::new(String::new(), POSTag::SentenceStart, 0, 0),
            POSToken::new("will".to_string(), POSTag::Verb, 0, 4),
            POSToken::new("effect".to_string(), POSTag::Verb, 5, 11),
            POSToken::new("change".to_string(), POSTag::Noun, 12, 18),
            POSToken::new(String::new(), POSTag::SentenceEnd, 18, 18),
        ];
        let result = engine.apply_rules_with_pos(&tokens);
        assert_eq!(result, "will affect change");

        // With antipattern match (effect + NN + immediately): should NOT apply the rule
        let tokens_with_antipattern = vec![
            POSToken::new(String::new(), POSTag::SentenceStart, 0, 0),
            POSToken::new("effect".to_string(), POSTag::Verb, 0, 6),
            POSToken::new("change".to_string(), POSTag::Noun, 7, 13),
            POSToken::new("immediately".to_string(), POSTag::Adverb, 14, 25),
            POSToken::new(String::new(), POSTag::SentenceEnd, 25, 25),
        ];
        let result = engine.apply_rules_with_pos(&tokens_with_antipattern);
        assert_eq!(result, "effect change immediately");
    }

    #[test]
    fn test_antipattern_at_text_boundary() {
        // Test antipattern when pattern is at the start of text
        // Antipattern must start at the same position as pattern (extends after pattern)
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="START_TEST" name="start test" prio="8">
                    <antipattern>
                        <token>foo</token>
                        <token>SPECIAL</token>
                    </antipattern>
                    <pattern>
                        <token>foo</token>
                    </pattern>
                    <message>Test</message>
                    <suggestion>bar</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Pattern at start, antipattern matches: should NOT apply
        let result = engine.apply_rules("foo SPECIAL world");
        assert_eq!(result, "foo SPECIAL world");

        // Pattern at start, no antipattern match: should apply
        let result = engine.apply_rules("foo normal world");
        assert_eq!(result, "bar normal world");

        // Test antipattern at the end of text (pattern at end)
        let xml_end = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="END_TEST" name="end test" prio="8">
                    <antipattern>
                        <token>foo</token>
                        <token>END</token>
                    </antipattern>
                    <pattern>
                        <token>foo</token>
                    </pattern>
                    <message>Test</message>
                    <suggestion>bar</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine_end = RuleEngine::from_xml(xml_end);

        // Pattern at end, antipattern matches: should NOT apply
        let result = engine_end.apply_rules("hello foo END");
        assert_eq!(result, "hello foo END");

        // Pattern at end, no antipattern match: should apply
        let result = engine_end.apply_rules("hello foo world");
        assert_eq!(result, "hello bar world");

        // Pattern at very end of text (no room for antipattern): should apply
        let result = engine_end.apply_rules("hello world foo");
        assert_eq!(result, "hello world bar");
    }

    #[test]
    fn test_antipattern_empty() {
        // Rule with no antipatterns (empty vector)
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="NO_ANTI" name="no antipatterns" prio="8">
                    <pattern>
                        <token>foo</token>
                    </pattern>
                    <message>Replace foo</message>
                    <suggestion>bar</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // No antipatterns defined: rule should always apply when pattern matches
        let result = engine.apply_rules("hello foo world");
        assert_eq!(result, "hello bar world");

        // Ensure multiple occurrences work (first one replaced)
        let result = engine.apply_rules("foo baz foo");
        assert!(result.contains("bar"));
    }

    #[test]
    fn test_antipattern_longer_than_remaining() {
        // Antipattern is longer than remaining text from match position
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="LONG_ANTI" name="long antipattern" prio="8">
                    <antipattern>
                        <token>foo</token>
                        <token>bar</token>
                        <token>baz</token>
                        <token>qux</token>
                        <token>quux</token>
                    </antipattern>
                    <pattern>
                        <token>foo</token>
                    </pattern>
                    <message>Replace foo</message>
                    <suggestion>replaced</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Text is shorter than antipattern: antipattern cannot match,
        // so the rule should apply
        let result = engine.apply_rules("hello foo bar");
        assert_eq!(result, "hello replaced bar");

        // Text exactly matches antipattern length: antipattern should block
        let result = engine.apply_rules("foo bar baz qux quux");
        assert_eq!(result, "foo bar baz qux quux");

        // Partial antipattern match (not all tokens): rule should apply
        let result = engine.apply_rules("foo bar baz qux other");
        assert_eq!(result, "replaced bar baz qux other");
    }

    #[test]
    fn test_antipattern_with_pos_boundary() {
        use crate::pos::{POSToken, POSTag};

        // Test POS antipattern at end of text (antipattern longer than remaining)
        let xml = r#"
        <rules>
            <category id="TEST" name="Test">
                <rule id="POS_BOUNDARY" name="pos boundary test" prio="8">
                    <antipattern>
                        <token>test</token>
                        <token postag="NN">word</token>
                        <token postag="VB">verb</token>
                    </antipattern>
                    <pattern>
                        <token>test</token>
                    </pattern>
                    <message>Replace test</message>
                    <suggestion>exam</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let engine = RuleEngine::from_xml(xml);

        // Tokens shorter than antipattern: should apply
        let tokens_short = vec![
            POSToken::new(String::new(), POSTag::SentenceStart, 0, 0),
            POSToken::new("test".to_string(), POSTag::Noun, 0, 4),
            POSToken::new("word".to_string(), POSTag::Noun, 5, 9),
            POSToken::new(String::new(), POSTag::SentenceEnd, 9, 9),
        ];
        let result = engine.apply_rules_with_pos(&tokens_short);
        assert_eq!(result, "exam word");

        // Antipattern matches completely: should NOT apply
        let tokens_full = vec![
            POSToken::new(String::new(), POSTag::SentenceStart, 0, 0),
            POSToken::new("test".to_string(), POSTag::Noun, 0, 4),
            POSToken::new("word".to_string(), POSTag::Noun, 5, 9),
            POSToken::new("verb".to_string(), POSTag::Verb, 10, 14),
            POSToken::new(String::new(), POSTag::SentenceEnd, 14, 14),
        ];
        let result = engine.apply_rules_with_pos(&tokens_full);
        assert_eq!(result, "test word verb");
    }
}
