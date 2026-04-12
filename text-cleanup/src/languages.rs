//! Supported Languages
//!
//! Languages for text cleanup rules.

use std::str::FromStr;

/// Supported languages for text cleanup
///
/// Each language has its own set of grammar rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, uniffi::Enum)]
pub enum Language {
    /// English (generic)
    #[default]
    English,
    /// English (US)
    EnglishUS,
    /// English (GB/UK)
    EnglishGB,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Portuguese
    Portuguese,
    /// Italian
    Italian,
    /// Dutch
    Dutch,
}

impl Language {
    /// Get all supported languages
    pub fn all() -> Vec<Language> {
        vec![
            Language::English,
            Language::EnglishUS,
            Language::EnglishGB,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Portuguese,
            Language::Italian,
            Language::Dutch,
        ]
    }

    /// Get currently implemented languages
    pub fn implemented() -> Vec<Language> {
        // Phase 1: Only English
        vec![Language::English, Language::EnglishUS, Language::EnglishGB]
    }

    /// Convert to ISO 639-1 code
    pub fn code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::EnglishUS => "en-US",
            Language::EnglishGB => "en-GB",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Portuguese => "pt",
            Language::Italian => "it",
            Language::Dutch => "nl",
        }
    }

    /// Get base language (for rule loading)
    /// e.g., en-US → en
    pub fn base_code(&self) -> &'static str {
        match self {
            Language::English | Language::EnglishUS | Language::EnglishGB => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Portuguese => "pt",
            Language::Italian => "it",
            Language::Dutch => "nl",
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::EnglishUS => "English (US)",
            Language::EnglishGB => "English (UK)",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Portuguese => "Portuguese",
            Language::Italian => "Italian",
            Language::Dutch => "Dutch",
        }
    }

    /// Check if this language is currently implemented
    pub fn is_implemented(&self) -> bool {
        Self::implemented().contains(self)
    }
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "en" | "english" => Ok(Language::English),
            "en-us" | "english-us" => Ok(Language::EnglishUS),
            "en-gb" | "english-gb" | "en-uk" => Ok(Language::EnglishGB),
            "es" | "spanish" => Ok(Language::Spanish),
            "fr" | "french" => Ok(Language::French),
            "de" | "german" => Ok(Language::German),
            "pt" | "portuguese" => Ok(Language::Portuguese),
            "it" | "italian" => Ok(Language::Italian),
            "nl" | "dutch" => Ok(Language::Dutch),
            _ => Err(format!("Unknown language: {}", s)),
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("en").unwrap(), Language::English);
        assert_eq!(Language::from_str("en-US").unwrap(), Language::EnglishUS);
        assert_eq!(Language::from_str("spanish").unwrap(), Language::Spanish);
        assert!(Language::from_str("unknown").is_err());
    }

    #[test]
    fn test_language_code() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::EnglishUS.code(), "en-US");
    }

    #[test]
    fn test_base_code() {
        assert_eq!(Language::EnglishUS.base_code(), "en");
        assert_eq!(Language::EnglishGB.base_code(), "en");
        assert_eq!(Language::Spanish.base_code(), "es");
    }

    #[test]
    fn test_default_language() {
        assert_eq!(Language::default(), Language::English);
    }

    #[test]
    fn test_implemented() {
        assert!(Language::English.is_implemented());
        // Spanish not implemented yet in Phase 1
        assert!(!Language::Spanish.is_implemented());
    }
}
