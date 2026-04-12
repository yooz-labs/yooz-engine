//! Benchmark comparing grammar correction engines:
//! - Yooz Text Cleanup (our engine)
//! - Harper (Automattic's grammar checker)
//!
//! Note: nlprule requires separate binary files, tested separately.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

// Our engine
use yooz_text_cleanup::RuleEngine;

// Harper
use harper_core::{Document, spell::FstDictionary, linting::{LintGroup, Linter}, parsers::PlainEnglish, Dialect};

/// Test sentences for benchmarking
const TEST_SENTENCES: &[&str] = &[
    // Subject-verb agreement
    "I are happy today.",
    "He don't know the answer.",
    "She have a car.",

    // Articles
    "I want a apple.",
    "She is a honest person.",

    // Informal
    "I gonna go home.",
    "She wanna eat pizza.",

    // Typos
    "I need alot of help.",
    "He is definately coming.",

    // Verb tense
    "I seen it yesterday.",
    "She done her homework.",

    // Confused words
    "I could of done it.",
    "Their going to the store.",

    // Clean (no errors)
    "The quick brown fox jumps over the lazy dog.",
    "I am happy to see you today.",
];

/// Longer text for realistic benchmarking
const LONG_TEXT: &str = "So I was like gonna go to the store and um I seen this really cool thing. \
He don't know nothing about that alot of people was there. \
Its kinda like when your gonna do something but you wanna wait. \
I are happy to help you with a apple and a umbrella. \
She have went to the store to get some things that she could of bought yesterday. \
Their going to be there at this point in time because its really important.";

fn bench_yooz_engine(c: &mut Criterion) {
    let engine = RuleEngine::new();

    let mut group = c.benchmark_group("yooz");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark individual sentences
    for (i, sentence) in TEST_SENTENCES.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("sentence", i),
            sentence,
            |b, s| b.iter(|| engine.apply_rules(black_box(s))),
        );
    }

    // Benchmark long text
    group.bench_function("long_text", |b| {
        b.iter(|| engine.apply_rules(black_box(LONG_TEXT)))
    });

    // Benchmark all sentences combined
    group.bench_function("all_sentences", |b| {
        b.iter(|| {
            for sentence in TEST_SENTENCES {
                let _ = engine.apply_rules(black_box(sentence));
            }
        })
    });

    group.finish();
}

fn bench_harper(c: &mut Criterion) {
    let dict = FstDictionary::curated();
    let mut linter = LintGroup::new_curated(dict, Dialect::American);

    let mut group = c.benchmark_group("harper");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark individual sentences
    for (i, sentence) in TEST_SENTENCES.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("sentence", i),
            sentence,
            |b, s| {
                b.iter(|| {
                    let doc = Document::new_curated(black_box(s), &PlainEnglish);
                    let _lints = linter.lint(&doc);
                })
            },
        );
    }

    // Benchmark long text
    group.bench_function("long_text", |b| {
        b.iter(|| {
            let doc = Document::new_curated(black_box(LONG_TEXT), &PlainEnglish);
            let _lints = linter.lint(&doc);
        })
    });

    // Benchmark all sentences combined
    group.bench_function("all_sentences", |b| {
        b.iter(|| {
            for sentence in TEST_SENTENCES {
                let doc = Document::new_curated(black_box(sentence), &PlainEnglish);
                let _ = linter.lint(&doc);
            }
        })
    });

    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let yooz_engine = RuleEngine::new();
    let dict = FstDictionary::curated();
    let mut harper_linter = LintGroup::new_curated(dict, Dialect::American);

    let mut group = c.benchmark_group("comparison");
    group.measurement_time(Duration::from_secs(10));

    // Head-to-head on long text
    group.bench_function("yooz_long", |b| {
        b.iter(|| yooz_engine.apply_rules(black_box(LONG_TEXT)))
    });

    group.bench_function("harper_long", |b| {
        b.iter(|| {
            let doc = Document::new_curated(black_box(LONG_TEXT), &PlainEnglish);
            harper_linter.lint(&doc)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_yooz_engine, bench_harper, bench_comparison);
criterion_main!(benches);
