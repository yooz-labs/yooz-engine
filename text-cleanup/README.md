# Yooz Text Cleanup

Fast, rule-based grammar correction for spoken-to-written text.

## Performance

| Engine | Per Sentence | Notes |
|--------|--------------|-------|
| **Yooz (grammar)** | 246µs | 2.1x faster than Harper |
| **Yooz (grammar+spell)** | 785µs | With optional spelling |
| Harper | 527µs | Grammar + spelling |
| nlprule | 725µs | LanguageTool port |

**1,560 rules total (919 simple + 641 POS-based), <1ms latency, ~100KB binary size**

## Quick Start

### Build & Test

```bash
cd text-cleanup
cargo test              # Run tests
cargo bench             # Run benchmarks
```

### Build XCFramework for App

```bash
./build-xcframework.sh  # Build XCFramework + Swift bindings
```

## API Usage

### Rust

```rust
use yooz_text_cleanup::{correct_grammar, Language, Category};

// Basic correction
let text = "I are happy";
let corrected = correct_grammar(text.to_string());
assert_eq!(corrected, "I am happy");

// With language and categories
let corrected = correct_grammar_with_categories(
    "I are eating a apple".to_string(),
    Language::English,
    vec![Category::Grammar, Category::Articles],
);
assert_eq!(corrected, "I am eating an apple");

// Spoken number conversion
let corrected = correct_grammar_with_categories(
    "I have twenty five items and one hundred dollars".to_string(),
    Language::English,
    vec![Category::Numbers],
);
assert_eq!(corrected, "I have 25 items and 100 dollars");
```

### Swift

```swift
import YoozTextCleanupFFI

// Basic correction
let corrected = correctGrammar(text: "I are happy")
// Returns: "I am happy"

// With categories
let result = correctGrammarWithCategories(
    text: "I are eating a apple",
    language: .english,
    categories: [.grammar, .articles]
)
// Returns: "I am eating an apple"
```

## Spell Checking (Optional)

Enable the `spelling` feature for spell checking:

```bash
cargo build --features spelling
```

### Rust

```rust
use yooz_text_cleanup::{check_spelling, correct_grammar_and_spelling};

// Check spelling (returns errors with suggestions)
let result = check_spelling("I hav a problm".to_string());
println!("Found {} errors", result.error_count);
for error in result.errors {
    println!("  '{}' at {}-{}: {:?}", error.word, error.start, error.end, error.suggestions);
}

// Correct grammar and spelling together
let corrected = correct_grammar_and_spelling("I are havng problms".to_string());
```

### Swift

```swift
// Check spelling
let result = checkSpelling(text: "I hav a problm")
print("Found \(result.errorCount) errors")

// Correct grammar and spelling
let corrected = correctGrammarAndSpelling(text: "I are havng problms")
```

## Categories & Tiers

| Category | Free | Pro | Description |
|----------|------|-----|-------------|
| Grammar | ✅ | ✅ | Subject-verb agreement |
| Articles | | ✅ | a/an corrections |
| Basic | | ✅ | Common typos (alot, etc) |
| Informal | | ✅ | gonna → going to |
| Verbs | | ✅ | Verb tense (I seen → I saw) |
| Numbers | | ✅ | Spoken numbers (twenty five → 25) |
| Punctuation | | ✅ | Punctuation fixes |
| Style | | ✅ | Style improvements |
| Advanced | | ✅ | Confused words |

**Tier Summary:**
- **Free**: Grammar only (~200 rules)
- **Pro**: All 9 categories (1,355 rules including POS-based)
- **Premium**: Pro + LLM features (handled by yooz-whisper, not this library)

## Query API

```rust
// Get available languages
let languages = get_implemented_languages();

// Get rule count
let count = get_rule_count(Language::English);

// Get categories for a language
let categories = get_available_categories_for_language(Language::English);

// Get rule info
let rules = get_all_rules(Language::English);
```

## Benchmarks

Run benchmarks:

```bash
# Basic benchmark
cargo run --release --features benchmark --bin tatoeba_benchmark

# With spelling
cargo run --release --features "benchmark,spelling" --bin spelling_benchmark

# POS tagger comparison
cargo run --release --features benchmark --bin pos_benchmark
```

See full benchmark results: [docs/benchmarks.md](../docs/benchmarks.md)

## Architecture

```
src/
├── lib.rs           # Public API (UniFFI exports)
├── xml_parser.rs    # Parse LanguageTool XML rules
├── rule_engine.rs   # Apply grammar rules
├── patterns.rs      # Token-based pattern matching
├── pos.rs           # POS tag system (NLTagger integration)
├── categories.rs    # Category definitions
├── languages.rs     # Language support
└── spelling.rs      # Optional spell checking

rules/
└── en/
    ├── grammar.xml           # 209 curated rules
    ├── lt-grammar-simple.xml # 567 LanguageTool grammar rules (no POS)
    ├── lt-style-simple.xml   # 117 LanguageTool style rules (no POS)
    ├── lt-grammar-pos.xml    # 649 POS-based grammar rules
    └── lt-style-pos.xml      # 18 POS-based style rules

swift-test/
└── NLTaggerBridge.swift  # Swift NLTagger wrapper for POS tagging
```

## XCFramework

Uses **UniFFI** (Mozilla) to create:
- Static library for macOS (Universal: arm64 + x86_64)
- Static library for iOS (arm64)
- Static library for iOS Simulator (Universal)
- Swift bindings (auto-generated)

Output: `build/YoozTextCleanup.xcframework`

## NLTagger Integration (POS-Based Rules)

The library includes 641 POS-based rules that require NLTagger for full functionality.
On Apple platforms, use the Swift bridge to get POS tags:

```swift
import YoozTextCleanupFFI
import NaturalLanguage

// Create NLTagger bridge
let bridge = NLTaggerBridge()

// Tokenize with POS tags
let tokens = bridge.tokenize("I am running quickly")
// Returns: [POSToken(text: "I", tag: .pronoun, ...),
//           POSToken(text: "am", tag: .verb, ...),
//           POSToken(text: "running", tag: .verb, ...),
//           POSToken(text: "quickly", tag: .adverb, ...)]

// Get rule counts
let total = getRuleCount(language: .english)     // 1560
let simple = getSimpleRuleCount(language: .english) // 919
let pos = getPosRuleCount(language: .english)      // 641
```

The POS-based rules unlock additional grammar checks for:
- Confused words (283 rules)
- Grammar (176 rules)
- Typos (158 rules)
- Collocations, compounding, and more

## Next Steps

- [x] 1,560 rules with <1ms latency
- [x] Optional spell checking API
- [x] Benchmark vs Harper, nlprule
- [x] POS-based rules using NLTagger (641 additional rules)
- [ ] Performance optimization for spelling
- [ ] POS-aware rule application function

## License

MIT
