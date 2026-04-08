// NLTagger-only Benchmark
// Measures Apple's NLTagger POS tagging performance standalone

import Foundation
import NaturalLanguage

// Sample sentences for benchmarking
let sampleSentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I am happy to see you today.",
    "She has finished her work already.",
    "Power is a drug which few ever manage to relinquish.",
    "Many musicians consider Bach as simply the best composer of all time.",
    "Tom is getting ready to leave for Australia.",
    "I'm afraid of not succeeding with this deal.",
    "She said bad things about him.",
    "It's not necessary to do that now.",
    "Overwork caused her to be absent from work for a week.",
    "Let's try something new today.",
    "I have to go to sleep now.",
    "Today is a beautiful day for a walk.",
    "The password is secret.",
    "I will be back soon.",
    "I'm at a loss for words.",
    "This is never going to end.",
    "I just don't know what to say.",
    "That was an unexpected surprise.",
    "I was in the mountains last week."
]

// Expand to desired count
func generateSentences(count: Int) -> [String] {
    var sentences: [String] = []
    while sentences.count < count {
        sentences.append(contentsOf: sampleSentences)
    }
    return Array(sentences.prefix(count))
}

// Benchmark NLTagger POS tagging
func benchmarkNLTagger(sentences: [String]) -> (duration: TimeInterval, tokenCount: Int) {
    let tagger = NLTagger(tagSchemes: [.lexicalClass])
    var totalTokens = 0

    let start = Date()

    for sentence in sentences {
        tagger.string = sentence
        let range = sentence.startIndex..<sentence.endIndex

        tagger.enumerateTags(in: range, unit: .word, scheme: .lexicalClass, options: [.omitWhitespace]) { tag, tokenRange in
            if tag != nil {
                totalTokens += 1
            }
            return true
        }
    }

    let elapsed = Date().timeIntervalSince(start)
    return (elapsed, totalTokens)
}

// Main
print("=== NLTagger POS Benchmark ===\n")

for count in [1000, 10000] {
    print("Generating \(count) sentences...")
    let sentences = generateSentences(count: count)

    // Warm-up
    _ = benchmarkNLTagger(sentences: Array(sentences.prefix(100)))

    print("Running NLTagger (Apple POS) on \(count) sentences...")
    let (time, tokens) = benchmarkNLTagger(sentences: sentences)
    let perSentence = (time * 1_000_000) / Double(sentences.count)

    print("  Total time: \(String(format: "%.3f", time))s")
    print("  Per sentence: \(String(format: "%.1f", perSentence))µs")
    print("  Tokens: \(tokens)")
    print()
}

print("=== Comparison with Rust Engines (from benchmarks) ===\n")
print("| Engine        | Per Sentence | Notes                        |")
print("|---------------|--------------|------------------------------|")
print("| NLTagger      | see above    | tokenize + POS (Apple)       |")
print("| Harper Brill  |    180.7µs   | tokenize + POS (Rust)        |")
print("| nlprule       |    756.3µs   | tokenize + POS + lemma       |")
print("| Yooz          |    260.0µs   | rule matching (893 rules)    |")

print("\n=== Notes ===")
print("- NLTagger: Zero binary size, OS-optimized")
print("- Harper: Brill tagger, rule-based")
print("- nlprule: Full LanguageTool NLP pipeline")
print("- Yooz: Pattern matching without POS (for comparison)")
