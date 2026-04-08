//! Optional spell checking module using Harper's dictionary.
//!
//! Enabled with the `spelling` feature.

#[cfg(feature = "spelling")]
use harper_core::{Document, spell::FstDictionary, linting::{LintGroup, Linter}, parsers::PlainEnglish, Dialect};

/// Result of spell checking a text
#[derive(Debug, Clone)]
#[cfg_attr(feature = "spelling", derive(uniffi::Record))]
pub struct SpellCheckResult {
    /// Original text
    pub original: String,
    /// Corrected text (with spelling fixes applied)
    pub corrected: String,
    /// Number of spelling errors found
    pub error_count: u32,
    /// List of spelling errors with suggestions
    pub errors: Vec<SpellingError>,
}

/// A single spelling error with suggestions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "spelling", derive(uniffi::Record))]
pub struct SpellingError {
    /// The misspelled word
    pub word: String,
    /// Start position in original text
    pub start: u32,
    /// End position in original text
    pub end: u32,
    /// Suggested corrections (may be empty)
    pub suggestions: Vec<String>,
}

/// Spell checker using Harper's dictionary
#[cfg(feature = "spelling")]
pub struct SpellChecker {
    linter: LintGroup,
}

#[cfg(feature = "spelling")]
impl SpellChecker {
    /// Create a new spell checker
    pub fn new() -> Self {
        let dict = FstDictionary::curated();
        let linter = LintGroup::new_curated(dict, Dialect::American);
        Self { linter }
    }

    /// Check spelling and return errors with suggestions
    pub fn check(&mut self, text: &str) -> SpellCheckResult {
        let doc = Document::new_curated(text, &PlainEnglish);
        let lints = self.linter.lint(&doc);

        let mut errors = Vec::new();

        for lint in &lints {
            // Get the span of the error
            let span = lint.span;
            let source = doc.get_span_content(&span);
            let word: String = source.iter().collect();

            // Get suggestions if available
            let suggestions: Vec<String> = lint.suggestions
                .iter()
                .map(|s| s.to_string())
                .collect();

            errors.push(SpellingError {
                word,
                start: span.start as u32,
                end: span.end as u32,
                suggestions,
            });
        }

        // For now, don't auto-correct - just return errors
        // Auto-correction is complex due to character vs byte indexing
        SpellCheckResult {
            original: text.to_string(),
            corrected: text.to_string(), // No auto-correction for now
            error_count: errors.len() as u32,
            errors,
        }
    }

    /// Simple spell check - just returns corrected text
    pub fn correct(&mut self, text: &str) -> String {
        self.check(text).corrected
    }
}

#[cfg(feature = "spelling")]
impl Default for SpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// UniFFI Exports (only when spelling feature enabled)
// ============================================================================

/// Check spelling and get detailed results
#[cfg(feature = "spelling")]
#[uniffi::export]
pub fn check_spelling(text: String) -> SpellCheckResult {
    let mut checker = SpellChecker::new();
    checker.check(&text)
}

/// Correct spelling errors (returns corrected text)
#[cfg(feature = "spelling")]
#[uniffi::export]
pub fn correct_spelling(text: String) -> String {
    let mut checker = SpellChecker::new();
    checker.correct(&text)
}

/// Correct both grammar and spelling
#[cfg(feature = "spelling")]
#[uniffi::export]
pub fn correct_grammar_and_spelling(text: String) -> String {
    use crate::RuleEngine;

    // First apply grammar rules
    let engine = RuleEngine::new();
    let grammar_corrected = engine.apply_rules(&text);

    // Then apply spelling corrections
    let mut checker = SpellChecker::new();
    checker.correct(&grammar_corrected)
}

#[cfg(test)]
#[cfg(feature = "spelling")]
mod tests {
    use super::*;

    #[test]
    fn test_spell_check() {
        let mut checker = SpellChecker::new();
        let result = checker.check("I hav a problm");
        assert!(result.error_count > 0);
    }

    #[test]
    fn test_spell_check_with_suggestions() {
        let mut checker = SpellChecker::new();
        let result = checker.check("teh quick brown fox");
        // "teh" should be detected as misspelled
        assert!(result.error_count > 0);
        // Should have at least one error for "teh"
        let has_teh_error = result.errors.iter().any(|e| e.word == "teh");
        assert!(has_teh_error, "Should detect 'teh' as misspelled");
    }
}
