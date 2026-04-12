//! Part-of-Speech Tag System
//!
//! Maps between NLTagger tags and Penn Treebank tags used by LanguageTool.

use std::str::FromStr;

/// Coarse-grained POS tags (NLTagger compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum POSTag {
    // Content words
    Noun,
    Verb,
    Adjective,
    Adverb,

    // Function words
    Pronoun,
    Determiner,
    Preposition,
    Conjunction,
    Particle,

    // Other
    Number,
    Interjection,
    Punctuation,

    // Special
    SentenceStart,
    SentenceEnd,
    Unknown,
}

impl POSTag {
    /// Convert from NLTagger tag string
    pub fn from_nltagger(tag: &str) -> Self {
        match tag {
            "Noun" => POSTag::Noun,
            "Verb" => POSTag::Verb,
            "Adjective" => POSTag::Adjective,
            "Adverb" => POSTag::Adverb,
            "Pronoun" => POSTag::Pronoun,
            "Determiner" => POSTag::Determiner,
            "Preposition" => POSTag::Preposition,
            "Conjunction" => POSTag::Conjunction,
            "Particle" => POSTag::Particle,
            "Number" => POSTag::Number,
            "Interjection" => POSTag::Interjection,
            "SentenceTerminator" | "OtherPunctuation" | "OpenQuote" |
            "CloseQuote" | "OpenParenthesis" | "CloseParenthesis" |
            "WordJoiner" | "Dash" => POSTag::Punctuation,
            _ => POSTag::Unknown,
        }
    }

    /// Check if this tag matches a LanguageTool postag pattern
    /// Supports basic patterns like "VB", "VB.*", "NN|VB", etc.
    pub fn matches_lt_pattern(&self, pattern: &str, word: &str, is_regexp: bool) -> bool {
        if is_regexp {
            // Handle regex patterns like "VB.*", "NN|VB", etc.
            for part in pattern.split('|') {
                let part = part.trim();
                let (base, wildcard) = if part.ends_with(".*") {
                    (part.trim_end_matches(".*"), true)
                } else if part.ends_with("?") {
                    (part.trim_end_matches('?'), true) // e.g., "VBN?" means VB or VBN
                } else {
                    (part, false)
                };
                if self.matches_single_lt_tag(base, word, wildcard) {
                    return true;
                }
            }
            false
        } else {
            self.matches_single_lt_tag(pattern, word, false)
        }
    }

    /// Match a single LanguageTool tag
    fn matches_single_lt_tag(&self, tag: &str, word: &str, wildcard: bool) -> bool {
        // If wildcard, match any sub-tag that starts with the base
        if wildcard {
            return match tag {
                "VB" | "V" => *self == POSTag::Verb,
                "NN" | "N" => *self == POSTag::Noun,
                "JJ" | "J" => *self == POSTag::Adjective,
                "RB" | "R" => *self == POSTag::Adverb,
                "PRP" | "P" => *self == POSTag::Pronoun,
                _ => self.matches_single_lt_tag(tag, word, false), // Fall back to exact match
            };
        }

        match tag {
            // Verb forms
            "VB" => *self == POSTag::Verb && is_base_verb(word),
            "VBD" => *self == POSTag::Verb && is_past_tense(word),
            "VBG" => *self == POSTag::Verb && word.ends_with("ing"),
            "VBN" => *self == POSTag::Verb && is_past_participle(word),
            "VBP" => *self == POSTag::Verb && !word.ends_with('s'),
            "VBZ" => *self == POSTag::Verb && word.ends_with('s'),

            // Noun forms
            "NN" => *self == POSTag::Noun && !word.ends_with('s'),
            "NNS" => *self == POSTag::Noun && word.ends_with('s'),
            "NNP" => *self == POSTag::Noun && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
            "NNPS" => *self == POSTag::Noun && word.ends_with('s') && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false),
            "NN:UN" | "NN:UN?" => *self == POSTag::Noun, // Uncountable - treat as any noun
            "N" if wildcard => *self == POSTag::Noun,

            // Adjective forms
            "JJ" => *self == POSTag::Adjective,
            "JJR" => *self == POSTag::Adjective && word.ends_with("er"),
            "JJS" => *self == POSTag::Adjective && word.ends_with("est"),

            // Adverb
            "RB" => *self == POSTag::Adverb,
            "RBR" => *self == POSTag::Adverb && word.ends_with("er"),
            "RBS" => *self == POSTag::Adverb && word.ends_with("est"),

            // Pronouns
            "PRP" => *self == POSTag::Pronoun,
            "PRP$" | "PRP\\$" => *self == POSTag::Pronoun && is_possessive_pronoun(word),
            "WP" => *self == POSTag::Pronoun && is_wh_pronoun(word),
            "P" if wildcard => *self == POSTag::Pronoun,

            // Determiners
            "DT" => *self == POSTag::Determiner,
            "PDT" => *self == POSTag::Determiner, // predeterminer
            "WDT" => *self == POSTag::Determiner && is_wh_determiner(word),

            // Prepositions and particles
            "IN" => *self == POSTag::Preposition,
            "TO" => *self == POSTag::Preposition && word.eq_ignore_ascii_case("to"),
            "RP" => *self == POSTag::Particle,

            // Conjunctions
            "CC" => *self == POSTag::Conjunction,

            // Numbers
            "CD" => *self == POSTag::Number,

            // Modal
            "MD" => *self == POSTag::Verb && is_modal(word),

            // Existential there
            "EX" => word.eq_ignore_ascii_case("there"),

            // Punctuation
            "PCT" => *self == POSTag::Punctuation,
            "POS" => word == "'s" || word == "'",

            // Special markers
            "SENT_START" => *self == POSTag::SentenceStart,
            "SENT_END" => *self == POSTag::SentenceEnd,

            // Unknown
            "UNKNOWN" => *self == POSTag::Unknown,

            // Interjection
            "UH" => *self == POSTag::Interjection,

            _ => false,
        }
    }
}

impl FromStr for POSTag {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "noun" => Ok(POSTag::Noun),
            "verb" => Ok(POSTag::Verb),
            "adjective" | "adj" => Ok(POSTag::Adjective),
            "adverb" | "adv" => Ok(POSTag::Adverb),
            "pronoun" => Ok(POSTag::Pronoun),
            "determiner" | "det" => Ok(POSTag::Determiner),
            "preposition" | "prep" => Ok(POSTag::Preposition),
            "conjunction" | "conj" => Ok(POSTag::Conjunction),
            "particle" => Ok(POSTag::Particle),
            "number" | "num" => Ok(POSTag::Number),
            "interjection" | "interj" => Ok(POSTag::Interjection),
            "punctuation" | "punct" => Ok(POSTag::Punctuation),
            _ => Err(()),
        }
    }
}

// Morphological heuristics

fn is_base_verb(word: &str) -> bool {
    // Base verbs don't end in -s, -ed, -ing (usually)
    !word.ends_with('s') && !word.ends_with("ed") && !word.ends_with("ing")
}

fn is_past_tense(word: &str) -> bool {
    word.ends_with("ed")
}

fn is_past_participle(word: &str) -> bool {
    // Past participles often end in -ed, -en, -t, -n
    word.ends_with("ed") || word.ends_with("en") ||
    word.ends_with("wn") || word.ends_with("nt") ||
    is_irregular_past_participle(word)
}

fn is_irregular_past_participle(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "been" | "done" | "gone" | "seen" | "taken" | "given" | "known" |
        "shown" | "written" | "broken" | "spoken" | "chosen" | "frozen" |
        "stolen" | "driven" | "risen" | "fallen" | "beaten" | "eaten" |
        "forgotten" | "gotten" | "hidden" | "ridden" | "bitten" |
        "begun" | "drunk" | "rung" | "sung" | "swum" | "run" |
        "come" | "become" | "overcome" | "cut" | "put" | "set" | "hit" |
        "hurt" | "let" | "shut" | "split" | "spread" | "quit" | "cost" |
        "built" | "sent" | "spent" | "lent" | "bent" | "left" | "kept" |
        "slept" | "crept" | "swept" | "wept" | "felt" | "dealt" | "meant" |
        "burnt" | "learnt" | "dreamt" | "leant" | "spelt" | "spilt" |
        "made" | "paid" | "said" | "laid" | "had" | "sold" | "told" |
        "held" | "stood" | "understood" | "fed" | "led" | "read" | "shed" |
        "bred" | "wed" | "sped" | "thought" | "bought" | "brought" |
        "caught" | "taught" | "fought" | "sought" | "wrought" |
        "found" | "bound" | "wound" | "ground" |
        "sat" | "spat" | "lost" | "shot" | "met" | "got" |
        "won" | "hung" | "dug" | "clung" | "stung" | "swung" | "flung" |
        "slung" | "wrung" | "strung" | "spun" | "stuck" | "struck" |
        "shrunk" | "stunk" | "sunk" | "slunk" |
        "wore" | "tore" | "bore" | "swore" |
        "flew" | "grew" | "knew" | "threw" | "drew" | "blew" |
        "woke" | "broke" | "spoke" | "chose" | "froze" | "stole" |
        "rode" | "wrote" | "rose" | "drove" | "strove" | "wove"
    )
}

fn is_possessive_pronoun(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "my" | "your" | "his" | "her" | "its" | "our" | "their" |
        "mine" | "yours" | "hers" | "ours" | "theirs"
    )
}

fn is_wh_pronoun(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "who" | "whom" | "whose" | "what" | "which" | "whoever" |
        "whomever" | "whatever" | "whichever"
    )
}

fn is_wh_determiner(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "which" | "what" | "whose" | "whatever" | "whichever"
    )
}

fn is_modal(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "can" | "could" | "may" | "might" | "must" | "shall" | "should" |
        "will" | "would" | "ought" | "need" | "dare" |
        "can't" | "couldn't" | "won't" | "wouldn't" | "shouldn't" |
        "mustn't" | "needn't" | "shan't" | "cannot"
    )
}

/// A token with its POS tag
#[derive(Debug, Clone, uniffi::Record)]
pub struct POSToken {
    pub text: String,
    pub tag: POSTag,
    pub start: u32,
    pub end: u32,
}

impl POSToken {
    pub fn new(text: String, tag: POSTag, start: usize, end: usize) -> Self {
        Self { text, tag, start: start as u32, end: end as u32 }
    }
}

/// Tokenize text with POS tags (for testing without Swift)
/// Uses simple heuristics - real POS tagging should use NLTagger via Swift
pub fn tokenize_with_pos_heuristic(text: &str) -> Vec<POSToken> {
    let mut tokens = Vec::new();
    let mut start = 0;

    // Add sentence start marker
    tokens.push(POSToken::new(String::new(), POSTag::SentenceStart, 0, 0));

    for word in text.split_whitespace() {
        let end = start + word.len();
        let tag = infer_pos_heuristic(word);
        tokens.push(POSToken::new(word.to_string(), tag, start, end));
        start = end + 1; // +1 for space
    }

    // Add sentence end marker
    tokens.push(POSToken::new(String::new(), POSTag::SentenceEnd, start, start));

    tokens
}

/// Infer POS tag using simple heuristics (fallback when NLTagger unavailable)
fn infer_pos_heuristic(word: &str) -> POSTag {
    let lower = word.to_lowercase();

    // Check for punctuation
    if word.chars().all(|c| c.is_ascii_punctuation()) {
        return POSTag::Punctuation;
    }

    // Check for numbers
    if word.chars().all(|c| c.is_numeric() || c == ',' || c == '.') {
        return POSTag::Number;
    }

    // Common determiners
    if matches!(lower.as_str(), "a" | "an" | "the" | "this" | "that" | "these" |
                "those" | "some" | "any" | "no" | "every" | "each" | "all" |
                "both" | "half" | "either" | "neither" | "few" | "many" | "much") {
        return POSTag::Determiner;
    }

    // Common pronouns
    if matches!(lower.as_str(), "i" | "me" | "my" | "mine" | "myself" |
                "you" | "your" | "yours" | "yourself" | "yourselves" |
                "he" | "him" | "his" | "himself" |
                "she" | "her" | "hers" | "herself" |
                "it" | "its" | "itself" |
                "we" | "us" | "our" | "ours" | "ourselves" |
                "they" | "them" | "their" | "theirs" | "themselves" |
                "who" | "whom" | "whose" | "what" | "which" |
                "whoever" | "whatever" | "whichever") {
        return POSTag::Pronoun;
    }

    // Common prepositions
    if matches!(lower.as_str(), "in" | "on" | "at" | "to" | "for" | "from" |
                "with" | "by" | "about" | "of" | "into" | "through" |
                "during" | "before" | "after" | "above" | "below" |
                "between" | "under" | "over" | "out" | "up" | "down") {
        return POSTag::Preposition;
    }

    // Common conjunctions
    if matches!(lower.as_str(), "and" | "or" | "but" | "nor" | "yet" | "so" |
                "because" | "although" | "while" | "if" | "unless" |
                "until" | "since" | "when" | "where" | "whereas") {
        return POSTag::Conjunction;
    }

    // Common modals (subset of verbs)
    if is_modal(&lower) {
        return POSTag::Verb;
    }

    // Common be/have/do verbs
    if matches!(lower.as_str(), "be" | "am" | "is" | "are" | "was" | "were" |
                "been" | "being" | "have" | "has" | "had" | "having" |
                "do" | "does" | "did" | "done" | "doing") {
        return POSTag::Verb;
    }

    // Morphological hints
    if word.ends_with("ly") && word.len() > 3 {
        return POSTag::Adverb;
    }
    if word.ends_with("ing") && word.len() > 4 {
        return POSTag::Verb;
    }
    if word.ends_with("ed") && word.len() > 3 {
        return POSTag::Verb;
    }
    if matches!(word.chars().last(), Some('s') | Some('S')) &&
       word.len() > 2 &&
       !matches!(lower.as_str(), "is" | "was" | "has" | "does" | "this" | "thus" | "us") {
        // Could be plural noun or 3rd person verb
        // Prefer noun for common patterns
        return POSTag::Noun;
    }

    // Adjective suffixes
    if word.ends_with("ful") || word.ends_with("less") || word.ends_with("ous") ||
       word.ends_with("ive") || word.ends_with("able") || word.ends_with("ible") {
        return POSTag::Adjective;
    }

    // Default to unknown
    POSTag::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nltagger_conversion() {
        assert_eq!(POSTag::from_nltagger("Noun"), POSTag::Noun);
        assert_eq!(POSTag::from_nltagger("Verb"), POSTag::Verb);
        assert_eq!(POSTag::from_nltagger("Adjective"), POSTag::Adjective);
        assert_eq!(POSTag::from_nltagger("SentenceTerminator"), POSTag::Punctuation);
    }

    #[test]
    fn test_lt_pattern_matching() {
        // Verb matching
        assert!(POSTag::Verb.matches_lt_pattern("VBG", "running", false));
        assert!(POSTag::Verb.matches_lt_pattern("VBD", "walked", false));
        assert!(POSTag::Verb.matches_lt_pattern("VBZ", "runs", false));

        // Noun matching
        assert!(POSTag::Noun.matches_lt_pattern("NNS", "dogs", false));
        assert!(POSTag::Noun.matches_lt_pattern("NN", "dog", false));

        // Regex patterns
        assert!(POSTag::Verb.matches_lt_pattern("VB.*", "running", true));
        assert!(POSTag::Verb.matches_lt_pattern("VB|NN", "walk", true));
    }

    #[test]
    fn test_modal_detection() {
        assert!(is_modal("can"));
        assert!(is_modal("would"));
        assert!(is_modal("shouldn't"));
        assert!(!is_modal("walk"));
    }

    #[test]
    fn test_heuristic_pos() {
        let tokens = tokenize_with_pos_heuristic("I am running quickly");

        // Skip sentence start
        assert_eq!(tokens[1].tag, POSTag::Pronoun); // I
        assert_eq!(tokens[2].tag, POSTag::Verb);    // am
        assert_eq!(tokens[3].tag, POSTag::Verb);    // running
        assert_eq!(tokens[4].tag, POSTag::Adverb);  // quickly
    }

    #[test]
    fn test_past_participle() {
        assert!(is_irregular_past_participle("been"));
        assert!(is_irregular_past_participle("done"));
        assert!(is_irregular_past_participle("seen"));
        assert!(!is_irregular_past_participle("walked"));
    }
}
