//! Benchmark using Tatoeba corpus (10k sentences)
//! Same methodology as nlprule benchmark for fair comparison.
//!
//! Metrics:
//! - Total time to process 10k sentences
//! - Average time per sentence
//! - Corrections found

use std::time::Instant;
use std::fs;

// Our engine
use yooz_text_cleanup::RuleEngine;

// Harper
use harper_core::{Document, spell::FstDictionary, linting::{LintGroup, Linter}, parsers::PlainEnglish, Dialect};

// nlprule
use nlprule::{Tokenizer, Rules};

const CORPUS_PATH: &str = "bench/corpus/tatoeba_10k.tsv";
const NLPRULE_TOKENIZER: &str = "bench/nlprule/en_tokenizer.bin";
const NLPRULE_RULES: &str = "bench/nlprule/en_rules.bin";

fn load_sentences() -> Vec<String> {
    let content = fs::read_to_string(CORPUS_PATH)
        .expect("Failed to read Tatoeba corpus. Run download first.");

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

fn benchmark_yooz(sentences: &[String]) -> (std::time::Duration, usize) {
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

    // Warm-up runs
    println!("Warming up...");
    let _ = benchmark_yooz(&sentences[..100]);
    let _ = benchmark_harper(&sentences[..100]);
    let _ = benchmark_nlprule(&sentences[..100]);

    println!("\n=== Tatoeba 10k Benchmark ===\n");

    // Yooz benchmark
    let engine = RuleEngine::new();
    let simple_rules = engine.simple_rule_count();
    let pos_rules = engine.pos_rule_count();
    println!("Running Yooz Text Cleanup ({} simple rules, {} POS rules available)...", simple_rules, pos_rules);
    let (yooz_time, yooz_corrections) = benchmark_yooz(&sentences);
    println!("  Total time: {:.2}s", yooz_time.as_secs_f64());
    println!("  Per sentence: {:.2}µs", yooz_time.as_micros() as f64 / sentences.len() as f64);
    println!("  Corrections: {}", yooz_corrections);

    println!();

    // Harper benchmark
    println!("Running Harper (spell + grammar)...");
    let (harper_time, harper_lints) = benchmark_harper(&sentences);
    println!("  Total time: {:.2}s", harper_time.as_secs_f64());
    println!("  Per sentence: {:.2}µs", harper_time.as_micros() as f64 / sentences.len() as f64);
    println!("  Lints found: {}", harper_lints);

    println!();

    // nlprule benchmark
    println!("Running nlprule (3,725 LanguageTool rules)...");
    match benchmark_nlprule(&sentences) {
        Ok((nlprule_time, nlprule_suggestions)) => {
            println!("  Total time: {:.2}s", nlprule_time.as_secs_f64());
            println!("  Per sentence: {:.2}µs", nlprule_time.as_micros() as f64 / sentences.len() as f64);
            println!("  Suggestions: {}", nlprule_suggestions);

            println!("\n=== Comparison ===\n");

            let yooz_per_sentence = yooz_time.as_micros() as f64 / sentences.len() as f64;
            let harper_per_sentence = harper_time.as_micros() as f64 / sentences.len() as f64;
            let nlprule_per_sentence = nlprule_time.as_micros() as f64 / sentences.len() as f64;

            println!("| Engine     | Per Sentence | Ratio vs Yooz |");
            println!("|------------|--------------|---------------|");
            println!("| Yooz       | {:>8.1}µs   | 1.0x          |", yooz_per_sentence);
            println!("| Harper     | {:>8.1}µs   | {:.1}x slower |", harper_per_sentence, harper_per_sentence / yooz_per_sentence);
            println!("| nlprule    | {:>8.1}µs   | {:.1}x slower |", nlprule_per_sentence, nlprule_per_sentence / yooz_per_sentence);
        }
        Err(e) => {
            println!("  Error: {}", e);
            println!("  Make sure nlprule binaries are in bench/nlprule/");

            println!("\n=== Comparison (Yooz vs Harper only) ===\n");

            let ratio = harper_time.as_secs_f64() / yooz_time.as_secs_f64();
            println!("Yooz is {:.1}x faster than Harper", ratio);
        }
    }

    println!("\nNote: Different engines, different purposes:");
    println!("  - Yooz: {} simple rules now, {} POS rules when NLTagger enabled", simple_rules, pos_rules);
    println!("  - Harper: General spell/grammar checking (Brill tagger + BiLSTM)");
    println!("  - nlprule: LanguageTool port (3,725 rules, full NLP pipeline)");
    println!("\nRule gap analysis (why nlprule has more rules):");
    println!("  - Antipatterns (negative matching): ~1,247 rules we don't support");
    println!("  - Backreferences (dynamic replacement): ~505 rules we don't support");
    println!("  - Chunking (phrase analysis): ~356 rules we don't support");
}
