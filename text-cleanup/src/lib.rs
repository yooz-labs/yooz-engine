//! Yooz Text Cleanup
//!
//! Fast, rule-based grammar correction for spoken-to-written text.
//!
//! ## Features
//!
//! - **Categorized rules**: Filter by category (grammar, articles, informal, etc.)
//! - **Multi-language**: Support for multiple languages (English first)
//! - **Tiered access**: Free, Pro, Premium tiers with different rule sets
//! - **Query API**: Inspect available categories and rules

mod categories;
mod languages;
mod xml_parser;
mod rule_engine;
mod patterns;
mod pos;
mod java_rules;
#[cfg(feature = "spelling")]
mod spelling;

pub use categories::{Category, Level};
pub use languages::Language;
pub use rule_engine::{RuleEngine, RuleInfo};
pub use pos::{POSTag, POSToken, tokenize_with_pos_heuristic};

use std::sync::OnceLock;

// Cached engine for repeated calls (avoids re-parsing XML each time)
static CACHED_ENGINE: OnceLock<RuleEngine> = OnceLock::new();

fn get_cached_engine() -> &'static RuleEngine {
    CACHED_ENGINE.get_or_init(|| RuleEngine::new())
}
#[cfg(feature = "spelling")]
pub use spelling::{SpellChecker, SpellCheckResult, SpellingError, check_spelling, correct_spelling, correct_grammar_and_spelling};

// UniFFI proc-macro setup
uniffi::setup_scaffolding!();

// ============================================================================
// UniFFI Exports - Core API
// ============================================================================

/// Correct grammar with default settings (English, all categories, standard level)
/// Uses cached engine for performance (avoids re-parsing XML each call)
#[uniffi::export]
pub fn correct_grammar(text: String) -> String {
    get_cached_engine().apply_rules(&text)
}

/// Correct grammar for a specific language
#[uniffi::export]
pub fn correct_grammar_for_language(text: String, language: Language) -> String {
    let engine = RuleEngine::for_language(language);
    engine.apply_rules(&text)
}

// ============================================================================
// UniFFI Exports - POS Tagging API (for NLTagger integration)
// ============================================================================

/// Convert NLTagger tag string to POSTag enum
#[uniffi::export]
pub fn pos_tag_from_nltagger(tag: String) -> POSTag {
    POSTag::from_nltagger(&tag)
}

/// Create a POS token (called from Swift after NLTagger tagging)
#[uniffi::export]
pub fn create_pos_token(text: String, tag: POSTag, start: u32, end: u32) -> POSToken {
    POSToken { text, tag, start, end }
}

/// Correct grammar using POS-tagged tokens (for NLTagger integration)
/// Call this after tokenizing with NLTagger in Swift
/// Uses cached engine for performance
#[uniffi::export]
pub fn correct_grammar_with_pos(tokens: Vec<POSToken>) -> String {
    get_cached_engine().apply_rules_with_pos(&tokens)
}

/// Correct grammar using POS tokens with category filtering
#[uniffi::export]
pub fn correct_grammar_with_pos_categories(
    tokens: Vec<POSToken>,
    language: Language,
    categories: Vec<Category>,
) -> String {
    let engine = RuleEngine::for_language(language);
    engine.apply_rules_with_pos_categories(&tokens, &categories)
}

/// Tokenize text with heuristic POS tagging (fallback when NLTagger unavailable)
#[uniffi::export]
pub fn tokenize_with_heuristic_pos(text: String) -> Vec<POSToken> {
    crate::pos::tokenize_with_pos_heuristic(&text)
}

/// Correct grammar with category filtering
#[uniffi::export]
pub fn correct_grammar_with_categories(
    text: String,
    language: Language,
    categories: Vec<Category>,
) -> String {
    let engine = RuleEngine::for_language(language);
    engine.apply_rules_with_categories(&text, &categories)
}

/// Correct grammar with full options (categories and level)
#[uniffi::export]
pub fn correct_grammar_with_options(
    text: String,
    language: Language,
    categories: Vec<Category>,
    level: Level,
) -> String {
    let engine = RuleEngine::for_language(language);
    engine.apply_rules_with_options(&text, &categories, level)
}

// ============================================================================
// UniFFI Exports - Query API
// ============================================================================

/// Get library version
#[uniffi::export]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get all supported languages
#[uniffi::export]
pub fn get_available_languages() -> Vec<Language> {
    Language::all()
}

/// Get currently implemented languages
#[uniffi::export]
pub fn get_implemented_languages() -> Vec<Language> {
    Language::implemented()
}

/// Get all categories
#[uniffi::export]
pub fn get_all_categories() -> Vec<Category> {
    Category::all()
}

/// Get free tier categories
#[uniffi::export]
pub fn get_free_categories() -> Vec<Category> {
    Category::free_tier()
}

/// Get pro tier categories
#[uniffi::export]
pub fn get_pro_categories() -> Vec<Category> {
    Category::pro_tier()
}

/// Get premium tier categories
#[uniffi::export]
pub fn get_premium_categories() -> Vec<Category> {
    Category::premium_tier()
}

/// Get available categories for a language (categories that have rules)
#[uniffi::export]
pub fn get_available_categories_for_language(language: Language) -> Vec<Category> {
    let engine = RuleEngine::for_language(language);
    engine.available_categories()
}

/// Get rule count for a language (all rules including POS-based)
#[uniffi::export]
pub fn get_rule_count(language: Language) -> u32 {
    let engine = RuleEngine::for_language(language);
    engine.rule_count() as u32
}

/// Get count of simple rules (no POS tagging required)
#[uniffi::export]
pub fn get_simple_rule_count(language: Language) -> u32 {
    let engine = RuleEngine::for_language(language);
    engine.simple_rule_count() as u32
}

/// Get count of POS-based rules (require NLTagger)
#[uniffi::export]
pub fn get_pos_rule_count(language: Language) -> u32 {
    let engine = RuleEngine::for_language(language);
    engine.pos_rule_count() as u32
}

/// Get rule count for a language and category
#[uniffi::export]
pub fn get_rule_count_for_category(language: Language, category: Category) -> u32 {
    let engine = RuleEngine::for_language(language);
    engine.rule_count_for_category(category) as u32
}

/// Get all rule info for a language
#[uniffi::export]
pub fn get_all_rules(language: Language) -> Vec<RuleInfo> {
    let engine = RuleEngine::for_language(language);
    engine.all_rules()
}

/// Get rule info for a category
#[uniffi::export]
pub fn get_rules_for_category(language: Language, category: Category) -> Vec<RuleInfo> {
    let engine = RuleEngine::for_language(language);
    engine.rules_for_category(category)
}

// ============================================================================
// UniFFI Exports - Programmatic Rules (Java-style rules)
// ============================================================================

use std::sync::OnceLock as ProgrammaticOnceLock;
static CACHED_PROGRAMMATIC_ENGINE: ProgrammaticOnceLock<java_rules::ProgrammaticRuleEngine> = ProgrammaticOnceLock::new();

fn get_cached_programmatic_engine() -> &'static java_rules::ProgrammaticRuleEngine {
    CACHED_PROGRAMMATIC_ENGINE.get_or_init(|| java_rules::ProgrammaticRuleEngine::new())
}

/// Apply programmatic rules (Java-style rules like word repeat detection)
/// Requires POS-tagged tokens from NLTagger
#[uniffi::export]
pub fn correct_with_programmatic_rules(tokens: Vec<POSToken>, text: String) -> String {
    get_cached_programmatic_engine().correct(&tokens, &text)
}

/// Apply all corrections: XML rules + programmatic rules
/// This is the most comprehensive correction that handles both pattern-based
/// and programmatic rules (like repeated word detection)
#[uniffi::export]
pub fn correct_grammar_full(tokens: Vec<POSToken>, text: String, language: Language, categories: Vec<Category>) -> String {
    // First apply XML-based rules
    let xml_engine = RuleEngine::for_language(language);
    let after_xml = xml_engine.apply_rules_with_pos_categories(&tokens, &categories);

    // Then apply programmatic rules (need to re-tokenize since text changed)
    let new_tokens = crate::pos::tokenize_with_pos_heuristic(&after_xml);
    get_cached_programmatic_engine().correct(&new_tokens, &after_xml)
}

/// Apply all corrections with simple text input (no POS tokens required)
/// Uses heuristic POS tagging internally
#[uniffi::export]
pub fn correct_grammar_full_simple(text: String, language: Language, categories: Vec<Category>) -> String {
    // Tokenize with heuristic POS
    let tokens = crate::pos::tokenize_with_pos_heuristic(&text);

    // Apply XML-based rules
    let xml_engine = RuleEngine::for_language(language);
    let after_xml = xml_engine.apply_rules_with_pos_categories(&tokens, &categories);

    // Apply programmatic rules
    let new_tokens = crate::pos::tokenize_with_pos_heuristic(&after_xml);
    get_cached_programmatic_engine().correct(&new_tokens, &after_xml)
}

/// Get count of programmatic rules
#[uniffi::export]
pub fn get_programmatic_rule_count() -> u32 {
    get_cached_programmatic_engine().rule_count() as u32
}

/// Get names of programmatic rules
#[uniffi::export]
pub fn get_programmatic_rule_names() -> Vec<String> {
    get_cached_programmatic_engine().rule_names()
}

/// Get IDs of programmatic rules
#[uniffi::export]
pub fn get_programmatic_rule_ids() -> Vec<String> {
    get_cached_programmatic_engine().rule_ids()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_correction() {
        let input = "I are happy".to_string();
        let output = correct_grammar(input);
        assert_eq!(output, "I am happy");
    }

    #[test]
    fn test_no_change_needed() {
        let input = "I am happy".to_string();
        let output = correct_grammar(input);
        assert_eq!(output, "I am happy");
    }

    #[test]
    fn test_article_correction() {
        let input = "a apple".to_string();
        let output = correct_grammar(input);
        assert_eq!(output, "an apple");
    }

    #[test]
    fn test_with_language() {
        let input = "I are happy".to_string();
        let output = correct_grammar_for_language(input, Language::EnglishUS);
        assert_eq!(output, "I am happy");
    }

    #[test]
    fn test_with_categories() {
        // Grammar only - should fix "I are" but not "a apple"
        let input = "I are eating a apple".to_string();
        let output = correct_grammar_with_categories(
            input,
            Language::English,
            vec![Category::Grammar],
        );
        assert_eq!(output, "I am eating a apple");

        // Both Grammar and Articles
        let input = "I are eating a apple".to_string();
        let output = correct_grammar_with_categories(
            input,
            Language::English,
            vec![Category::Grammar, Category::Articles],
        );
        assert_eq!(output, "I am eating an apple");
    }

    #[test]
    fn test_tier_categories() {
        let free = get_free_categories();
        let pro = get_pro_categories();
        let premium = get_premium_categories();

        // Free = Grammar only (1 category)
        // Pro = All XML rules (9 categories)
        // Premium = Same as Pro for stt-engine (LLM features in yooz-whisper)
        assert_eq!(free.len(), 1);
        assert_eq!(pro.len(), 9);
        assert_eq!(premium.len(), 9);

        // Free should contain only Grammar
        assert!(free.contains(&Category::Grammar));

        // Free should be subset of pro
        for cat in &free {
            assert!(pro.contains(cat));
        }
    }

    #[test]
    fn test_query_api() {
        let languages = get_implemented_languages();
        assert!(languages.contains(&Language::English));

        let categories = get_available_categories_for_language(Language::English);
        assert!(categories.contains(&Category::Grammar));

        let rule_count = get_rule_count(Language::English);
        assert!(rule_count > 0);
    }

    #[test]
    fn test_rule_counts() {
        let total = get_rule_count(Language::English);
        let simple = get_simple_rule_count(Language::English);
        let pos = get_pos_rule_count(Language::English);

        // Total should equal simple + POS
        assert_eq!(total, simple + pos);

        // After filtering partial markers (rules that replace only marked tokens),
        // we have fewer but more reliable rules
        assert!(simple >= 500, "Expected at least 500 simple rules, got {}", simple);

        // We have POS rules from extraction
        assert!(pos >= 250, "Expected at least 250 POS rules, got {}", pos);

        // Print for visibility
        println!("Rule counts - Total: {}, Simple: {}, POS: {}", total, simple, pos);
    }

    #[test]
    fn test_pos_vs_simple_corrections() {
        // Test cases that benefit from POS tagging
        let test_cases = vec![
            // Possessive vs contraction
            "Its going to rain today",
            "The dog wagged it's tail",
            "Your the best person I know",
            "I think your right about that",
            "Their going to the store",
            "I saw there car parked outside",
            "Whose coming to the party",
            "Who's book is this",
            // Verb forms requiring context
            "He should of known better",
            "She would of helped us",
            "I could of done it myself",
            // Subject-verb with pronouns
            "Me and him went to the store",
            "Her and I are friends",
            // Common confusions
            "I seen him yesterday",
            "She done her homework",
            "They was at the party",
            "He don't know the answer",
            // Affect vs effect
            "The medicine will effect your sleep",
            "Climate change effects everyone",
            // Then vs than
            "She is smarter then me",
            "I would rather walk then drive",
        ];

        println!("\n=== POS vs Simple Correction Comparison ===\n");
        println!("{:<45} | {:<45} | {:<45}", "INPUT", "SIMPLE", "WITH POS");
        println!("{}", "-".repeat(140));

        for input in &test_cases {
            let simple_result = correct_grammar(input.to_string());

            // Use heuristic POS for testing (in real app, NLTagger provides better tags)
            let tokens = tokenize_with_heuristic_pos(input.to_string());
            let pos_result = correct_grammar_with_pos(tokens);

            let simple_changed = if simple_result != *input { "✓" } else { "" };
            let pos_changed = if pos_result != *input { "✓" } else { "" };
            let pos_better = if pos_result != simple_result && pos_result != *input { " ★" } else { "" };

            println!("{:<45} | {:<43}{} | {:<43}{}{}",
                input,
                simple_result, simple_changed,
                pos_result, pos_changed, pos_better
            );
        }
        println!();
    }

    #[test]
    fn test_its_going_bug() {
        // Debug the "Its going" → "it's to" bug
        let input = "Its going to rain today";
        let result = correct_grammar(input.to_string());
        println!("\nDEBUG: '{}' → '{}'", input, result);

        // The correction should be "It's going to rain today"
        // NOT "it's to rain today" (losing "going")
        assert!(
            result.contains("going"),
            "The word 'going' was lost! Result: '{}'", result
        );
    }

    #[test]
    fn test_debug_word_repeat_pipeline() {
        // Debug test to trace each step of the pipeline for "we need to to understand"
        let input = "we need to to understand";
        println!("\n=== DEBUG: Word Repeat Pipeline ===");
        println!("Input: '{}'", input);

        // Step 1: Tokenize
        let tokens = crate::pos::tokenize_with_pos_heuristic(input);
        println!("\nStep 1 - Tokens:");
        for t in &tokens {
            println!("  '{}' tag={:?} pos={}-{}", t.text, t.tag, t.start, t.end);
        }

        // Step 2: Apply XML rules only
        let xml_engine = RuleEngine::for_language(Language::English);
        let all_categories = Category::all();
        let after_xml = xml_engine.apply_rules_with_pos_categories(&tokens, &all_categories);
        println!("\nStep 2 - After XML rules: '{}'", after_xml);

        // Step 3: Re-tokenize after XML
        let new_tokens = crate::pos::tokenize_with_pos_heuristic(&after_xml);
        println!("\nStep 3 - Re-tokenized:");
        for t in &new_tokens {
            println!("  '{}' tag={:?} pos={}-{}", t.text, t.tag, t.start, t.end);
        }

        // Step 4: Apply programmatic rules only
        let prog_engine = java_rules::ProgrammaticRuleEngine::new();
        let after_prog = prog_engine.correct(&new_tokens, &after_xml);
        println!("\nStep 4 - After programmatic rules: '{}'", after_prog);

        // Step 5: Full pipeline
        let full_result = correct_grammar_full_simple(
            input.to_string(),
            Language::English,
            all_categories.clone(),
        );
        println!("\nStep 5 - Full pipeline result: '{}'", full_result);

        // Check if "need" is preserved
        assert!(
            full_result.contains("need"),
            "ERROR: 'need' was lost! Result: '{}'", full_result
        );

        // Expected result
        assert_eq!(full_result, "we need to understand");
    }

    #[test]
    fn test_xml_rules_only_on_to_to() {
        // Test what XML rules alone do to "we need to to understand"
        let input = "we need to to understand";
        let tokens = crate::pos::tokenize_with_pos_heuristic(input);
        let xml_engine = RuleEngine::for_language(Language::English);

        // Try each category separately
        println!("\n=== Testing each category separately ===");
        for cat in Category::all() {
            let result = xml_engine.apply_rules_with_pos_categories(&tokens, &[cat.clone()]);
            if result != input {
                println!("Category {:?}: '{}' → '{}'", cat, input, result);
            }
        }

        // Try with all categories
        let all_result = xml_engine.apply_rules_with_pos_categories(&tokens, &Category::all());
        println!("\nAll categories: '{}' → '{}'", input, all_result);

        // The XML rules should NOT modify this - word repeat is handled by programmatic rules
        assert_eq!(
            all_result, input,
            "XML rules should not modify 'we need to to understand'"
        );
    }

    #[test]
    fn test_find_matching_rule() {
        // Find which specific rule is matching "we need to to understand"
        let input = "we need to to understand";
        let tokens = crate::pos::tokenize_with_pos_heuristic(input);
        let xml_engine = RuleEngine::for_language(Language::English);

        println!("\n=== Finding matching rules for '{}' ===", input);
        println!("Tokens:");
        for (i, t) in tokens.iter().enumerate() {
            println!("  [{}] '{}' tag={:?}", i, t.text, t.tag);
        }

        // Find the first matching rule
        if let Some((rule_id, replacement, pattern)) = xml_engine.find_matching_rule(&tokens, &[Category::Grammar]) {
            println!("\n*** FOUND MATCHING RULE ***");
            println!("Rule ID: {}", rule_id);
            println!("Replacement: '{}'", replacement);
            println!("Pattern: {}", pattern);
        } else {
            println!("\nNo matching rule found");
        }

        // Apply rules iteratively to see the progression
        let mut current = input.to_string();
        let mut current_tokens = tokens.clone();

        for i in 0..5 {
            let before = current.clone();

            // Find which rule is about to fire
            if let Some((rule_id, replacement, _)) = xml_engine.find_matching_rule(&current_tokens, &[Category::Grammar]) {
                println!("\nIteration {}: Rule '{}' will fire (replacement: '{}')", i + 1, rule_id, replacement);
            }

            let result = xml_engine.apply_rules_with_pos_categories(&current_tokens, &[Category::Grammar]);
            if result == before {
                println!("\nNo more rules match.");
                break;
            }
            println!("  '{}' → '{}'", before, result);
            current = result.clone();
            current_tokens = crate::pos::tokenize_with_pos_heuristic(&result);
        }
    }

    #[test]
    fn test_combined_i_are_to_to() {
        // Test "I are going to to the store" → "I'm going to the store"
        // This tests both SVA correction (I are → I'm) and word repeat (to to → to)
        let input = "I are going to to the store";
        let result = correct_grammar_full_simple(
            input.to_string(),
            Language::English,
            Category::all(),
        );
        // Should be "I'm going to the store" (I are VBG → I'm VBG, then to to → to)
        assert_eq!(result, "I'm going to the store");
    }

    #[test]
    fn test_i_am_going_no_cascade() {
        // Test that "I am going" stays unchanged (AM_I rule should not fire due to antipattern)
        let input = "I am going to the store";
        let result = correct_grammar_full_simple(
            input.to_string(),
            Language::English,
            Category::all(),
        );
        // Should stay "I am going to the store" - AM_I rule should not fire
        assert_eq!(result, "I am going to the store");
    }

    #[test]
    fn test_i_is_going_no_cascade() {
        // Test "I is going" → "I am going" (not "I am I going")
        // I_IS rule converts "I is" → "I am", then AM_I antipattern prevents cascade
        let input = "I is going to the store";
        let result = correct_grammar_full_simple(
            input.to_string(),
            Language::English,
            Category::all(),
        );
        // Should be "I am going to the store" - not "I am I going..."
        assert_eq!(result, "I am going to the store");
    }
}
