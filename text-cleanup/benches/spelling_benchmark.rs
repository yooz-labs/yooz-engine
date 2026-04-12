//! Benchmark comparing Yooz (grammar + spelling) vs Harper vs nlprule
//! Uses Tatoeba 10k corpus

use std::time::Instant;
use std::fs;

// Our engine
use yooz_text_cleanup::RuleEngine;
#[cfg(feature = "spelling")]
use yooz_text_cleanup::SpellChecker;

// Harper
use harper_core::{Document, spell::FstDictionary, linting::{LintGroup, Linter}, parsers::PlainEnglish, Dialect};

// nlprule
use nlprule::{Tokenizer, Rules};

const CORPUS_PATH: &str = "bench/corpus/tatoeba_10k.tsv";
const NLPRULE_TOKENIZER: &str = "bench/nlprule/en_tokenizer.bin";
const NLPRULE_RULES: &str = "bench/nlprule/en_rules.bin";

fn load_sentences() -> Vec<String> {
    let content = fs::read_to_string(CORPUS_PATH)
        .expect("Failed to read Tatoeba corpus.");

    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(parts[2].to_string())
            } else {
                None
            }
        })
        .collect()
}

fn benchmark_yooz_grammar_only(sentences: &[String]) -> (std::time::Duration, usize) {
    let engine = RuleEngine::new();
    let mut total_corrections = 0;

    let start = Instant::now();
    for sentence in sentences {
        let result = engine.apply_rules(sentence);
        if result != *sentence {
            total_corrections += 1;
        }
    }
    let elapsed = start.elapsed();

    (elapsed, total_corrections)
}

#[cfg(feature = "spelling")]
fn benchmark_yooz_with_spelling(sentences: &[String]) -> (std::time::Duration, usize, usize) {
    let engine = RuleEngine::new();
    let mut checker = SpellChecker::new();
    let mut grammar_corrections = 0;
    let mut spelling_errors = 0;

    let start = Instant::now();
    for sentence in sentences {
        // Apply grammar rules first
        let grammar_result = engine.apply_rules(sentence);
        if grammar_result != *sentence {
            grammar_corrections += 1;
        }

        // Then check spelling
        let spell_result = checker.check(&grammar_result);
        spelling_errors += spell_result.error_count as usize;
    }
    let elapsed = start.elapsed();

    (elapsed, grammar_corrections, spelling_errors)
}

fn benchmark_harper(sentences: &[String]) -> (std::time::Duration, usize) {
    let dict = FstDictionary::curated();
    let mut linter = LintGroup::new_curated(dict, Dialect::American);
    let mut total_lints = 0;

    let start = Instant::now();
    for sentence in sentences {
        let doc = Document::new_curated(sentence, &PlainEnglish);
        let lints = linter.lint(&doc);
        total_lints += lints.len();
    }
    let elapsed = start.elapsed();

    (elapsed, total_lints)
}

fn benchmark_nlprule(sentences: &[String]) -> Result<(std::time::Duration, usize), String> {
    let tokenizer = Tokenizer::new(NLPRULE_TOKENIZER)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    let rules = Rules::new(NLPRULE_RULES)
        .map_err(|e| format!("Failed to load rules: {}", e))?;

    let mut total_suggestions = 0;

    let start = Instant::now();
    for sentence in sentences {
        let suggestions = rules.suggest(sentence, &tokenizer);
        total_suggestions += suggestions.len();
    }
    let elapsed = start.elapsed();

    Ok((elapsed, total_suggestions))
}

fn main() {
    println!("Loading Tatoeba corpus...");
    let sentences = load_sentences();
    println!("Loaded {} sentences\n", sentences.len());

    // Warm-up
    println!("Warming up...");
    let _ = benchmark_yooz_grammar_only(&sentences[..100]);
    #[cfg(feature = "spelling")]
    let _ = benchmark_yooz_with_spelling(&sentences[..100]);
    let _ = benchmark_harper(&sentences[..100]);
    let _ = benchmark_nlprule(&sentences[..100]);

    println!("\n=== Benchmark Results (10k sentences) ===\n");

    // Yooz grammar only
    println!("Running Yooz (grammar only, 893 rules)...");
    let (yooz_grammar_time, yooz_grammar_corrections) = benchmark_yooz_grammar_only(&sentences);
    let yooz_grammar_per = yooz_grammar_time.as_micros() as f64 / sentences.len() as f64;
    println!("  Total time: {:.2}s", yooz_grammar_time.as_secs_f64());
    println!("  Per sentence: {:.1}µs", yooz_grammar_per);
    println!("  Grammar corrections: {}", yooz_grammar_corrections);

    println!();

    // Yooz grammar + spelling
    #[cfg(feature = "spelling")]
    {
        println!("Running Yooz (grammar + spelling)...");
        let (yooz_full_time, grammar_corr, spell_errors) = benchmark_yooz_with_spelling(&sentences);
        let yooz_full_per = yooz_full_time.as_micros() as f64 / sentences.len() as f64;
        println!("  Total time: {:.2}s", yooz_full_time.as_secs_f64());
        println!("  Per sentence: {:.1}µs", yooz_full_per);
        println!("  Grammar corrections: {}", grammar_corr);
        println!("  Spelling errors found: {}", spell_errors);
    }

    println!();

    // Harper
    println!("Running Harper (grammar + spelling)...");
    let (harper_time, harper_lints) = benchmark_harper(&sentences);
    let harper_per = harper_time.as_micros() as f64 / sentences.len() as f64;
    println!("  Total time: {:.2}s", harper_time.as_secs_f64());
    println!("  Per sentence: {:.1}µs", harper_per);
    println!("  Total lints: {}", harper_lints);

    println!();

    // nlprule
    println!("Running nlprule (grammar, 3725 rules)...");
    match benchmark_nlprule(&sentences) {
        Ok((nlprule_time, nlprule_suggestions)) => {
            let nlprule_per = nlprule_time.as_micros() as f64 / sentences.len() as f64;
            println!("  Total time: {:.2}s", nlprule_time.as_secs_f64());
            println!("  Per sentence: {:.1}µs", nlprule_per);
            println!("  Suggestions: {}", nlprule_suggestions);

            println!("\n=== Comparison ===\n");

            println!("| Engine                | Per Sentence | Detections |");
            println!("|-----------------------|--------------|------------|");
            println!("| Yooz (grammar only)   | {:>8.1}µs   | {} corrections |", yooz_grammar_per, yooz_grammar_corrections);
            #[cfg(feature = "spelling")]
            {
                let (yooz_full_time, grammar_corr, spell_errors) = benchmark_yooz_with_spelling(&sentences);
                let yooz_full_per = yooz_full_time.as_micros() as f64 / sentences.len() as f64;
                println!("| Yooz (grammar+spell)  | {:>8.1}µs   | {} grammar, {} spelling |", yooz_full_per, grammar_corr, spell_errors);
            }
            println!("| Harper                | {:>8.1}µs   | {} lints |", harper_per, harper_lints);
            println!("| nlprule               | {:>8.1}µs   | {} suggestions |", nlprule_per, nlprule_suggestions);
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    println!("\n=== Notes ===");
    println!("- Yooz grammar: 893 curated spoken-to-written rules");
    println!("- Yooz spelling: Uses Harper's FstDictionary");
    println!("- Harper: Brill tagger + spelling dictionary");
    println!("- nlprule: LanguageTool port (3725 rules, no spelling)");
}
