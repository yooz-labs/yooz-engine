//! XML Rule Parser
//!
//! Parses LanguageTool-compatible XML grammar rules.

use serde::Deserialize;
use crate::categories::Category;
use std::str::FromStr;

#[derive(Debug, Deserialize, PartialEq)]
pub struct Rules {
    #[serde(rename = "category")]
    pub categories: Vec<XmlCategory>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct XmlCategory {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "rule", default)]
    pub rules: Vec<Rule>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Rule {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "@name")]
    pub name: String,
    /// Our category (basic, grammar, etc.)
    #[serde(rename = "@category", default)]
    pub category: Option<String>,
    /// Rule tags: "default" or "picky"
    #[serde(rename = "@tags", default)]
    pub tags: Option<String>,
    /// Priority (higher = more important)
    #[serde(rename = "@prio", default)]
    pub priority: Option<i32>,
    /// Default state: "on" or "off"
    #[serde(rename = "@default", default)]
    pub default: Option<String>,
    /// Marker start: index of first token to replace (0-based)
    /// If not specified, defaults to 0 (entire pattern is error)
    #[serde(rename = "@marker_start", default)]
    pub marker_start: Option<usize>,
    /// Marker end: index after last token to replace (exclusive)
    /// If not specified, defaults to pattern length (entire pattern is error)
    #[serde(rename = "@marker_end", default)]
    pub marker_end: Option<usize>,
    /// Antipatterns: patterns that prevent the rule from matching
    #[serde(rename = "antipattern", default)]
    pub antipatterns: Vec<Pattern>,
    #[serde(rename = "pattern")]
    pub pattern: Pattern,
    #[serde(rename = "message")]
    pub message: String,
    #[serde(rename = "suggestion", default)]
    pub suggestions: Vec<Suggestion>,
}

impl Rule {
    /// Get the rule's category, falling back to parent category or Grammar
    pub fn get_category(&self, parent_category_id: &str) -> Category {
        // First try rule's own category attribute
        if let Some(ref cat) = self.category {
            if let Ok(c) = Category::from_str(cat) {
                return c;
            }
        }

        // Fall back to mapping from parent XML category ID
        map_xml_category_to_category(parent_category_id)
    }

    /// Check if this is a "picky" (strict) rule
    pub fn is_picky(&self) -> bool {
        self.tags.as_ref().map(|t| t.contains("picky")).unwrap_or(false)
    }

    /// Check if this rule is enabled by default
    pub fn is_default_on(&self) -> bool {
        self.default.as_ref().map(|d| d != "off").unwrap_or(true)
    }

    /// Get priority (default 0)
    pub fn get_priority(&self) -> i32 {
        self.priority.unwrap_or(0)
    }
}

/// Map LanguageTool XML category IDs to our categories
fn map_xml_category_to_category(xml_category_id: &str) -> Category {
    match xml_category_id.to_uppercase().as_str() {
        // Direct mappings (our IDs + LanguageTool IDs)
        "GRAMMAR" | "SUBJECT_VERB_AGREEMENT" => Category::Grammar,
        "ARTICLES" => Category::Articles,
        "VERBS" | "VERB_TENSE" => Category::Verbs,
        "NUMBERS" => Category::Numbers,

        // Basic category
        "BASIC" | "TYPOS" | "COMMON_TYPOS" | "COMPOUNDING" => Category::Basic,

        // Informal/spoken
        "INFORMAL" | "COLLOQUIALISMS" => Category::Informal,

        // Punctuation
        "PUNCTUATION" | "CASING" | "TYPOGRAPHY" => Category::Punctuation,

        // Style
        "STYLE" | "REDUNDANCY" | "REPETITIONS_STYLE" => Category::Style,

        // Advanced
        "ADVANCED" | "CONFUSED_WORDS" | "SEMANTICS" => Category::Advanced,

        // Default
        _ => Category::Grammar,
    }
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Pattern {
    #[serde(rename = "token", default)]
    pub tokens: Vec<Token>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Token {
    #[serde(rename = "$value", default)]
    pub text: String,
    #[serde(rename = "@regexp", default)]
    pub regexp: Option<String>,
    #[serde(rename = "@postag", default)]
    pub postag: Option<String>,
    #[serde(rename = "@postag_regexp", default)]
    pub postag_regexp: Option<String>,
    #[serde(rename = "@case_sensitive", default)]
    pub case_sensitive: Option<bool>,
    #[serde(rename = "@min", default)]
    pub min: Option<String>,
    #[serde(rename = "@max", default)]
    pub max: Option<String>,
    /// Whether this token requires no space before it (attached to previous)
    #[serde(rename = "@spacebefore", default)]
    pub spacebefore: Option<String>,
    /// Exception elements - words that should NOT match
    #[serde(rename = "exception", default)]
    pub exceptions: Vec<Exception>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Exception {
    #[serde(rename = "$value", default)]
    pub text: String,
    #[serde(rename = "@regexp", default)]
    pub regexp: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
pub struct Suggestion {
    #[serde(rename = "$value", default)]
    pub text: String,
}

/// Parse XML rules from a string
pub fn parse_rules(xml: &str) -> Result<Rules, quick_xml::DeError> {
    quick_xml::de::from_str(xml)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let xml = r#"
        <rules>
            <category id="GRAMMAR" name="Grammar">
                <rule id="I_ARE" name="I are">
                    <pattern>
                        <token>I</token>
                        <token>are</token>
                    </pattern>
                    <message>Use 'am' instead of 'are' with 'I'</message>
                    <suggestion>I am</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let rules = parse_rules(xml).unwrap();
        assert_eq!(rules.categories.len(), 1);
        assert_eq!(rules.categories[0].rules.len(), 1);
        assert_eq!(rules.categories[0].rules[0].id, "I_ARE");
    }

    #[test]
    fn test_parse_rule_with_category() {
        let xml = r#"
        <rules>
            <category id="GRAMMAR" name="Grammar">
                <rule id="TEST" name="Test" category="articles" tags="picky" prio="5">
                    <pattern>
                        <token>a</token>
                        <token>apple</token>
                    </pattern>
                    <message>Test message</message>
                    <suggestion>an apple</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let rules = parse_rules(xml).unwrap();
        let rule = &rules.categories[0].rules[0];

        assert_eq!(rule.category, Some("articles".to_string()));
        assert_eq!(rule.tags, Some("picky".to_string()));
        assert_eq!(rule.priority, Some(5));
        assert!(rule.is_picky());
        assert_eq!(rule.get_category("GRAMMAR"), Category::Articles);
    }

    #[test]
    fn test_category_fallback() {
        let xml = r#"
        <rules>
            <category id="SUBJECT_VERB_AGREEMENT" name="Subject-Verb Agreement">
                <rule id="TEST" name="Test">
                    <pattern>
                        <token>test</token>
                    </pattern>
                    <message>Test</message>
                </rule>
            </category>
        </rules>
        "#;

        let rules = parse_rules(xml).unwrap();
        let rule = &rules.categories[0].rules[0];

        // No category attribute, should fall back to parent
        assert_eq!(rule.get_category("SUBJECT_VERB_AGREEMENT"), Category::Grammar);
    }

    #[test]
    fn test_map_xml_categories() {
        assert_eq!(map_xml_category_to_category("GRAMMAR"), Category::Grammar);
        assert_eq!(map_xml_category_to_category("ARTICLES"), Category::Articles);
        assert_eq!(map_xml_category_to_category("TYPOS"), Category::Basic);
        assert_eq!(map_xml_category_to_category("COLLOQUIALISMS"), Category::Informal);
        assert_eq!(map_xml_category_to_category("PUNCTUATION"), Category::Punctuation);
        assert_eq!(map_xml_category_to_category("STYLE"), Category::Style);
        assert_eq!(map_xml_category_to_category("CONFUSED_WORDS"), Category::Advanced);
    }

    #[test]
    fn test_parse_rule_with_antipattern() {
        let xml = r#"
        <rules>
            <category id="PUNCTUATION" name="Punctuation">
                <rule id="COMMA_PERIOD" name="comma before period">
                    <antipattern>
                        <token>,</token>
                        <token>.</token>
                        <token>NET</token>
                    </antipattern>
                    <antipattern>
                        <token>,</token>
                        <token>,</token>
                    </antipattern>
                    <pattern>
                        <token>,</token>
                        <token>.</token>
                    </pattern>
                    <message>Remove redundant comma before period.</message>
                    <suggestion>.</suggestion>
                </rule>
            </category>
        </rules>
        "#;

        let rules = parse_rules(xml).unwrap();
        let rule = &rules.categories[0].rules[0];

        assert_eq!(rule.id, "COMMA_PERIOD");
        assert_eq!(rule.antipatterns.len(), 2);
        assert_eq!(rule.antipatterns[0].tokens.len(), 3);
        assert_eq!(rule.antipatterns[0].tokens[2].text, "NET");
        assert_eq!(rule.antipatterns[1].tokens.len(), 2);
        assert_eq!(rule.pattern.tokens.len(), 2);
    }
}
