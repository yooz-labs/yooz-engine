//! Rule Categories
//!
//! Categories for organizing and filtering grammar rules.

use std::str::FromStr;

/// Rule categories for text cleanup
///
/// Categories allow selective application of rules based on:
/// - Performance requirements (fewer categories = faster)
/// - Monetization tiers (free/pro/premium)
/// - User preferences (keep informal speech, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum Category {
    /// Common typos and obvious fixes (e.g., alot → a lot)
    Basic,
    /// Subject-verb agreement (e.g., I are → I am)
    Grammar,
    /// Article corrections (e.g., a apple → an apple)
    Articles,
    /// Spoken to formal (e.g., gonna → going to)
    Informal,
    /// Irregular verb corrections (e.g., I seen → I saw)
    Verbs,
    /// Number formatting (e.g., twenty three → 23)
    Numbers,
    /// Capitalization and punctuation
    Punctuation,
    /// Redundancy and wordiness
    Style,
    /// Complex grammar and nuanced corrections
    Advanced,
}

impl Category {
    /// Get all categories
    pub fn all() -> Vec<Category> {
        vec![
            Category::Basic,
            Category::Grammar,
            Category::Articles,
            Category::Informal,
            Category::Verbs,
            Category::Numbers,
            Category::Punctuation,
            Category::Style,
            Category::Advanced,
        ]
    }

    /// Get free tier categories (basic grammar only)
    /// Note: Tiering decisions are made by yooz-whisper, not stt-engine
    pub fn free_tier() -> Vec<Category> {
        vec![Category::Grammar]
    }

    /// Get pro tier categories (all rule-based corrections)
    /// Note: Tiering decisions are made by yooz-whisper, not stt-engine
    pub fn pro_tier() -> Vec<Category> {
        Self::all()  // Pro gets all XML-based rules
    }

    /// Get premium tier categories (same as pro for stt-engine)
    /// Note: Premium features (LLM, dictionaries, memory) are in yooz-whisper
    pub fn premium_tier() -> Vec<Category> {
        Self::all()
    }

    /// Convert to string identifier
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Basic => "basic",
            Category::Grammar => "grammar",
            Category::Articles => "articles",
            Category::Informal => "informal",
            Category::Verbs => "verbs",
            Category::Numbers => "numbers",
            Category::Punctuation => "punctuation",
            Category::Style => "style",
            Category::Advanced => "advanced",
        }
    }
}

impl FromStr for Category {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "basic" => Ok(Category::Basic),
            "grammar" => Ok(Category::Grammar),
            "articles" => Ok(Category::Articles),
            "informal" => Ok(Category::Informal),
            "verbs" => Ok(Category::Verbs),
            "numbers" => Ok(Category::Numbers),
            "punctuation" => Ok(Category::Punctuation),
            "style" => Ok(Category::Style),
            "advanced" => Ok(Category::Advanced),
            _ => Err(format!("Unknown category: {}", s)),
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Rule strictness level
///
/// Controls which rules within a category are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, uniffi::Enum)]
pub enum Level {
    /// Only critical rules (~20% of rules)
    Essential,
    /// Default rules (~60% of rules)
    #[default]
    Standard,
    /// Include picky/strict rules (100% of rules)
    Thorough,
}

impl Level {
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Essential => "essential",
            Level::Standard => "standard",
            Level::Thorough => "thorough",
        }
    }
}

impl FromStr for Level {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "essential" => Ok(Level::Essential),
            "standard" => Ok(Level::Standard),
            "thorough" => Ok(Level::Thorough),
            _ => Err(format!("Unknown level: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_from_str() {
        assert_eq!(Category::from_str("grammar").unwrap(), Category::Grammar);
        assert_eq!(Category::from_str("BASIC").unwrap(), Category::Basic);
        assert!(Category::from_str("unknown").is_err());
    }

    #[test]
    fn test_tier_categories() {
        assert_eq!(Category::free_tier().len(), 1);  // Grammar only
        assert_eq!(Category::pro_tier().len(), 9);   // All categories
        assert_eq!(Category::premium_tier().len(), 9);
    }

    #[test]
    fn test_level_default() {
        assert_eq!(Level::default(), Level::Standard);
    }
}
