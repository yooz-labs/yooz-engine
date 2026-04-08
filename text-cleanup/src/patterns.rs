//! Pattern Matching
//!
//! Token-based pattern matching for grammar rules with optional POS support.

use regex::Regex;
use crate::pos::{POSTag, POSToken};

#[derive(Debug, Clone)]
pub struct TokenMatcher {
    text: Option<String>,
    regexp: Option<Regex>,
    /// POS tag pattern (e.g., "VB", "VB.*", "NN|VB")
    postag: Option<String>,
    /// Whether postag is a regexp
    postag_regexp: bool,
    /// Minimum occurrences (for optional tokens)
    min: u8,
    /// Maximum occurrences
    max: u8,
    /// Whether matching is case-sensitive (default: false for text, true for regexp)
    case_sensitive: bool,
    /// Whether this token requires no space before it (attached to previous token)
    spacebefore_no: bool,
    /// Exception words that should NOT match even if pattern matches
    exceptions: Vec<String>,
}

impl TokenMatcher {
    pub fn new(text: Option<String>, regexp: Option<String>) -> Self {
        let regexp = regexp.and_then(|r| {
            match Regex::new(&r) {
                Ok(re) => Some(re),
                Err(e) => {
                    eprintln!("Warning: Invalid regex pattern '{}': {}", r, e);
                    None
                }
            }
        });
        Self {
            text,
            regexp,
            postag: None,
            postag_regexp: false,
            min: 1,
            max: 1,
            case_sensitive: false,
            spacebefore_no: false,
            exceptions: Vec::new(),
        }
    }

    pub fn with_postag(
        text: Option<String>,
        regexp: Option<String>,
        postag: Option<String>,
        postag_regexp: bool,
    ) -> Self {
        let regexp = regexp.and_then(|r| {
            match Regex::new(&r) {
                Ok(re) => Some(re),
                Err(e) => {
                    eprintln!("Warning: Invalid regex pattern '{}': {}", r, e);
                    None
                }
            }
        });
        Self {
            text,
            regexp,
            postag,
            postag_regexp,
            min: 1,
            max: 1,
            case_sensitive: false,
            spacebefore_no: false,
            exceptions: Vec::new(),
        }
    }

    pub fn with_bounds(
        text: Option<String>,
        regexp: Option<String>,
        postag: Option<String>,
        postag_regexp: bool,
        min: u8,
        max: u8,
    ) -> Self {
        let regexp = regexp.and_then(|r| {
            match Regex::new(&r) {
                Ok(re) => Some(re),
                Err(e) => {
                    eprintln!("Warning: Invalid regex pattern '{}': {}", r, e);
                    None
                }
            }
        });
        Self {
            text,
            regexp,
            postag,
            postag_regexp,
            min,
            max,
            case_sensitive: false,
            spacebefore_no: false,
            exceptions: Vec::new(),
        }
    }

    /// Full constructor with all options
    pub fn with_all_options(
        text: Option<String>,
        regexp: Option<String>,
        postag: Option<String>,
        postag_regexp: bool,
        min: u8,
        max: u8,
        case_sensitive: bool,
        spacebefore_no: bool,
        exceptions: Vec<String>,
    ) -> Self {
        let regexp = regexp.and_then(|r| {
            match Regex::new(&r) {
                Ok(re) => Some(re),
                Err(e) => {
                    eprintln!("Warning: Invalid regex pattern '{}': {}", r, e);
                    None
                }
            }
        });
        Self {
            text,
            regexp,
            postag,
            postag_regexp,
            min,
            max,
            case_sensitive,
            spacebefore_no,
            exceptions,
        }
    }

    /// Check if this token requires no space before it
    pub fn requires_no_space_before(&self) -> bool {
        self.spacebefore_no
    }

    /// Check if this matcher requires POS tagging
    pub fn requires_pos(&self) -> bool {
        self.postag.is_some()
    }

    /// Check if this token is optional (min = 0)
    pub fn is_optional(&self) -> bool {
        self.min == 0
    }

    /// Simple text/regexp matching (no POS)
    pub fn matches(&self, word: &str) -> bool {
        self.matches_text(word)
    }

    /// Match text only (ignoring POS)
    fn matches_text(&self, word: &str) -> bool {
        // Check exceptions first - if word is in exceptions, don't match
        if !self.exceptions.is_empty() {
            for exception in &self.exceptions {
                if exception.eq_ignore_ascii_case(word) {
                    return false;
                }
            }
        }

        if let Some(text) = &self.text {
            return if self.case_sensitive {
                text == word
            } else {
                text.eq_ignore_ascii_case(word)
            };
        }
        if let Some(regexp) = &self.regexp {
            return regexp.is_match(word);
        }
        // If no text or regexp, match any word (for POS-only patterns)
        self.text.is_none() && self.regexp.is_none()
    }

    /// Match against a POS-tagged token
    pub fn matches_pos_token(&self, token: &POSToken) -> bool {
        // Check text/regexp first
        if !self.matches_text(&token.text) {
            return false;
        }

        // Check POS tag if specified
        if let Some(ref postag_pattern) = self.postag {
            return token.tag.matches_lt_pattern(postag_pattern, &token.text, self.postag_regexp);
        }

        true
    }
}

/// Simple tokenizer - splits on whitespace
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Find pattern matches in text (simple, no POS)
pub fn find_matches(tokens: &[String], pattern: &[TokenMatcher]) -> Vec<usize> {
    let mut matches = Vec::new();

    if pattern.is_empty() || tokens.is_empty() {
        return matches;
    }

    for i in 0..tokens.len() {
        if i + pattern.len() > tokens.len() {
            break;
        }

        let mut all_match = true;
        for (j, matcher) in pattern.iter().enumerate() {
            if !matcher.matches(&tokens[i + j]) {
                all_match = false;
                break;
            }
        }

        if all_match {
            matches.push(i);
        }
    }

    matches
}

/// Match result with token indices
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Start index in token array
    pub start: usize,
    /// End index (exclusive) in token array
    pub end: usize,
    /// Number of tokens matched
    pub len: usize,
}

/// Find pattern matches in POS-tagged tokens
/// Handles optional tokens (min=0) with backtracking
pub fn find_matches_pos(tokens: &[POSToken], pattern: &[TokenMatcher]) -> Vec<PatternMatch> {
    let mut matches = Vec::new();

    if pattern.is_empty() || tokens.is_empty() {
        return matches;
    }

    // Filter out sentence markers for matching
    let content_tokens: Vec<(usize, &POSToken)> = tokens.iter()
        .enumerate()
        .filter(|(_, t)| t.tag != POSTag::SentenceStart && t.tag != POSTag::SentenceEnd)
        .collect();

    for start_idx in 0..content_tokens.len() {
        if let Some(match_len) = try_match_pos(&content_tokens[start_idx..], pattern) {
            matches.push(PatternMatch {
                start: content_tokens[start_idx].0,
                end: content_tokens[start_idx].0 + match_len,
                len: match_len,
            });
        }
    }

    matches
}

/// Try to match pattern at current position
fn try_match_pos(tokens: &[(usize, &POSToken)], pattern: &[TokenMatcher]) -> Option<usize> {
    if tokens.is_empty() {
        // Check if remaining patterns are all optional
        return if pattern.iter().all(|m| m.is_optional()) {
            Some(0)
        } else {
            None
        };
    }

    if pattern.is_empty() {
        return Some(0);
    }

    let matcher = &pattern[0];
    let token = tokens[0].1;

    if matcher.matches_pos_token(token) {
        // Token matches, continue with rest of pattern
        if let Some(rest_len) = try_match_pos(&tokens[1..], &pattern[1..]) {
            return Some(1 + rest_len);
        }
    }

    // If optional, try skipping this matcher
    if matcher.is_optional() {
        return try_match_pos(tokens, &pattern[1..]);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "I are happy";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["I", "are", "happy"]);
    }

    #[test]
    fn test_token_matcher_exact() {
        let matcher = TokenMatcher::new(Some("I".to_string()), None);
        assert!(matcher.matches("I"));
        assert!(matcher.matches("i")); // case insensitive
        assert!(!matcher.matches("you"));
    }

    #[test]
    fn test_find_matches() {
        let tokens = vec!["I".to_string(), "are".to_string(), "happy".to_string()];
        let pattern = vec![
            TokenMatcher::new(Some("I".to_string()), None),
            TokenMatcher::new(Some("are".to_string()), None),
        ];

        let matches = find_matches(&tokens, &pattern);
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn test_no_match() {
        let tokens = vec!["I".to_string(), "am".to_string(), "happy".to_string()];
        let pattern = vec![
            TokenMatcher::new(Some("I".to_string()), None),
            TokenMatcher::new(Some("are".to_string()), None),
        ];

        let matches = find_matches(&tokens, &pattern);
        assert!(matches.is_empty());
    }
}
