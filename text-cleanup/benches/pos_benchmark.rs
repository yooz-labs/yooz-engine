//! POS Rules Benchmark
//! Compares grammar correction performance between:
//! - Simple rules only (919 rules, no POS tagging)
//! - POS-enabled rules (1560 rules, with heuristic POS tagging)
//!
//! This shows the actual benefit of using POS tagging for more accurate corrections.

use std::time::Instant;
use std::fs;

// Our engine
use yooz_text_cleanup::{RuleEngine, tokenize_with_pos_heuristic};

const CORPUS_PATH: &str = "bench/corpus/tatoeba_10k.tsv";

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

/// Benchmark simple rules only (no POS tagging)
fn benchmark_simple_rules(sentences: &[String]) -> (std::time::Duration, usize) {
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

/// Benchmark POS-enabled rules (with heuristic POS tagging)
fn benchmark_pos_rules(sentences: &[String]) -> (std::time::Duration, usize) {
    let engine = RuleEngine::new();
    let mut total_corrections = 0;

    let start = Instant::now();
    for sentence in sentences {
        // Tokenize with heuristic POS
        let tokens = tokenize_with_pos_heuristic(sentence);
        // Apply all rules (simple + POS)
        let result = engine.apply_rules_with_pos(&tokens);
        if result != *sentence {
            total_corrections += 1;
        }
    }
    let elapsed = start.elapsed();

    (elapsed, total_corrections)
}

/// Benchmark just the heuristic POS tagging
fn benchmark_heuristic_pos(sentences: &[String]) -> (std::time::Duration, usize) {
    let mut total_tokens = 0;

    let start = Instant::now();
    for sentence in sentences {
        let tokens = tokenize_with_pos_heuristic(sentence);
        total_tokens += tokens.len();
    }
    let elapsed = start.elapsed();

    (elapsed, total_tokens)
}

fn main() {
    println!("Loading Tatoeba corpus...");
    let sentences = load_sentences();
    println!("Loaded {} sentences\n", sentences.len());

    // Get rule counts
    let engine = RuleEngine::new();
    let simple_count = engine.simple_rule_count();
    let pos_count = engine.pos_rule_count();
    let total_count = engine.rule_count();

    println!("Rule counts:");
    println!("  Simple rules (no POS): {}", simple_count);
    println!("  POS-based rules:       {}", pos_count);
    println!("  Total:                 {}", total_count);

    // Warm-up
    println!("\nWarming up...");
    let _ = benchmark_simple_rules(&sentences[..100]);
    let _ = benchmark_pos_rules(&sentences[..100]);
    let _ = benchmark_heuristic_pos(&sentences[..100]);

    println!("\n=== POS Rules Benchmark (10k sentences) ===\n");

    // Simple rules benchmark
    println!("1. Simple Rules Only ({} rules)...", simple_count);
    let (simple_time, simple_corrections) = benchmark_simple_rules(&sentences);
    println!("   Total time: {:.2}s", simple_time.as_secs_f64());
    println!("   Per sentence: {:.1}µs", simple_time.as_micros() as f64 / sentences.len() as f64);
    println!("   Corrections: {}", simple_corrections);

    println!();

    // Heuristic POS benchmark
    println!("2. Heuristic POS Tagging Only...");
    let (pos_time, token_count) = benchmark_heuristic_pos(&sentences);
    println!("   Total time: {:.2}s", pos_time.as_secs_f64());
    println!("   Per sentence: {:.1}µs", pos_time.as_micros() as f64 / sentences.len() as f64);
    println!("   Tokens: {}", token_count);

    println!();

    // POS-enabled rules benchmark
    println!("3. POS-Enabled Rules ({} total rules)...", total_count);
    let (full_time, full_corrections) = benchmark_pos_rules(&sentences);
    println!("   Total time: {:.2}s", full_time.as_secs_f64());
    println!("   Per sentence: {:.1}µs", full_time.as_micros() as f64 / sentences.len() as f64);
    println!("   Corrections: {}", full_corrections);

    println!("\n=== Summary ===\n");

    let simple_per = simple_time.as_micros() as f64 / sentences.len() as f64;
    let pos_per = pos_time.as_micros() as f64 / sentences.len() as f64;
    let full_per = full_time.as_micros() as f64 / sentences.len() as f64;

    println!("| Mode              | Rules | Per Sentence | Corrections | Overhead |");
    println!("|-------------------|-------|--------------|-------------|----------|");
    println!("| Simple only       | {:>5} | {:>8.1}µs   | {:>11} | baseline |",
             simple_count, simple_per, simple_corrections);
    println!("| Heuristic POS     |   n/a | {:>8.1}µs   |         n/a | +{:.0}µs   |",
             pos_per, pos_per);
    println!("| Simple + POS      | {:>5} | {:>8.1}µs   | {:>11} | {:.1}x    |",
             total_count, full_per, full_corrections, full_per / simple_per);

    println!();
    println!("Additional corrections with POS: {}",
             full_corrections.saturating_sub(simple_corrections));
    println!("POS overhead: {:.1}µs/sentence ({:.1}% of simple)",
             full_per - simple_per,
             ((full_per - simple_per) / simple_per) * 100.0);

    println!("\n=== Notes ===");
    println!("- Simple rules: Fast text matching, no NLP required");
    println!("- POS rules: Require part-of-speech tagging for context-aware matching");
    println!("- Heuristic POS: Uses morphological rules (fallback when NLTagger unavailable)");
    println!("- NLTagger (Apple): ~23µs/sentence, 7.7x faster than heuristic");
    println!("\nWith NLTagger on Apple platforms, expect ~{:.1}µs overhead instead of {:.1}µs",
             23.0 + (full_per - simple_per - pos_per), pos_per);
}
