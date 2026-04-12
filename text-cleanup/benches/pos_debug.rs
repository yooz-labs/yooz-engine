//! Debug POS corrections - show examples of what POS rules are correcting

use std::fs;
use yooz_text_cleanup::{RuleEngine, tokenize_with_pos_heuristic};

fn main() {
    let content = fs::read_to_string("bench/corpus/tatoeba_10k.tsv")
        .expect("Failed to read corpus");

    let engine = RuleEngine::new();
    let mut pos_only = Vec::new();
    let mut both = Vec::new();

    for line in content.lines().take(5000) {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 { continue; }
        let sentence = parts[2];

        // Simple rules
        let simple = engine.apply_rules(sentence);

        // POS rules
        let tokens = tokenize_with_pos_heuristic(sentence);
        let pos = engine.apply_rules_with_pos(&tokens);

        // Track all changes
        if pos != sentence && simple == sentence {
            pos_only.push((sentence.to_string(), pos.clone(), simple.clone()));
            if pos_only.len() >= 30 { break; }
        } else if pos != sentence && simple != sentence {
            both.push((sentence.to_string(), simple.clone(), pos.clone()));
        }
    }

    println!("=== POS-Only Corrections (first 30) ===\n");
    println!("These corrections are ONLY caught by POS rules, not simple rules.\n");

    for (i, (orig, fixed, _)) in pos_only.iter().enumerate() {
        if orig != fixed {
            println!("{}. ORIG:  {}", i + 1, orig);
            println!("   FIXED: {}", fixed);
            println!();
        }
    }

    println!("Total POS-only corrections shown: {}", pos_only.len());

    println!("\n=== Corrections by Both Simple and POS (first 10) ===\n");
    for (i, (orig, simple, pos)) in both.iter().take(10).enumerate() {
        println!("{}. ORIG:   {}", i + 1, orig);
        println!("   SIMPLE: {}", simple);
        println!("   POS:    {}", pos);
        println!();
    }

    // Show some POS tokens to debug
    println!("\n=== Sample POS Tagging ===\n");
    let test_sentences = [
        "I are going home",
        "She don't like it",
        "They was here yesterday",
        "He goed to school",
    ];

    for sent in test_sentences {
        let tokens = tokenize_with_pos_heuristic(sent);
        println!("\"{}\":", sent);
        for tok in &tokens {
            println!("  {:12} -> {:?}", tok.text, tok.tag);
        }
        let result = engine.apply_rules_with_pos(&tokens);
        println!("  Result: {}", result);
        println!();
    }
}
