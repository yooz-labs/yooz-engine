//! Benchmark for rule engine performance
//!
//! Target: <10ms per sentence

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use yooz_text_cleanup::correct_grammar;

fn benchmark_simple_correction(c: &mut Criterion) {
    c.bench_function("simple_correction", |b| {
        b.iter(|| correct_grammar(black_box("I are happy".to_string())))
    });
}

fn benchmark_no_correction(c: &mut Criterion) {
    c.bench_function("no_correction", |b| {
        b.iter(|| correct_grammar(black_box("I am happy".to_string())))
    });
}

fn benchmark_multiple_errors(c: &mut Criterion) {
    let text = "I are happy and he don't know and she seen it".to_string();
    c.bench_function("multiple_errors", |b| {
        b.iter(|| correct_grammar(black_box(text.clone())))
    });
}

fn benchmark_long_text(c: &mut Criterion) {
    let text = "I are going to the store and I seen alot of people there and he don't know what to do about it".to_string();
    c.bench_function("long_text", |b| {
        b.iter(|| correct_grammar(black_box(text.clone())))
    });
}

criterion_group!(
    benches,
    benchmark_simple_correction,
    benchmark_no_correction,
    benchmark_multiple_errors,
    benchmark_long_text,
);
criterion_main!(benches);
