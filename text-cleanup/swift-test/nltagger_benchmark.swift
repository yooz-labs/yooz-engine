// NLTagger + POS Rules Benchmark
// Tests the full pipeline: NLTagger POS tagging → Yooz POS-aware correction

import Foundation
import NaturalLanguage

// Sample sentences for benchmarking (mix of correct and incorrect)
let sampleSentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I are happy to see you today.",  // Grammar error
    "She has finished her work already.",
    "He don't like pizza.",  // Grammar error
    "Many musicians consider Bach as simply the best composer of all time.",
    "Tom is getting ready to leave for Australia.",
    "They was here yesterday.",  // Grammar error
    "She said bad things about him.",
    "It's not necessary to do that now.",
    "I seen that movie before.",  // Grammar error
    "Let's try something new today.",
    "I have to go to sleep now.",
    "She don't want to go.",  // Grammar error
    "Today is a beautiful day for a walk.",
    "The password is secret.",
    "I will be back soon.",
    "He goed to school.",  // Grammar error
    "This is never going to end.",
    "I just don't know what to say.",
    "That was an unexpected surprise."
]

// Load Tatoeba corpus if available
func loadTatoeba(path: String, limit: Int = 10000) -> [String]? {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        return nil
    }

    var sentences: [String] = []
    for line in content.split(separator: "\n").prefix(limit) {
        let parts = line.split(separator: "\t")
        if parts.count >= 3 {
            sentences.append(String(parts[2]))
        }
    }
    return sentences.isEmpty ? nil : sentences
}

// Expand sample sentences for testing
func generateSentences(count: Int) -> [String] {
    var sentences: [String] = []
    while sentences.count < count {
        sentences.append(contentsOf: sampleSentences)
    }
    return Array(sentences.prefix(count))
}

// Benchmark NLTagger POS tagging only
func benchmarkNLTaggerOnly(sentences: [String]) -> (duration: TimeInterval, tokenCount: Int) {
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

// Benchmark Yooz simple rules only (no POS)
func benchmarkYoozSimple(sentences: [String]) -> (duration: TimeInterval, corrections: Int) {
    var totalCorrections = 0

    let start = Date()

    for sentence in sentences {
        let result = correctGrammar(text: sentence)
        if result != sentence {
            totalCorrections += 1
        }
    }

    let elapsed = Date().timeIntervalSince(start)
    return (elapsed, totalCorrections)
}

// Benchmark full pipeline: NLTagger + Yooz POS rules
func benchmarkFullPipeline(sentences: [String]) -> (duration: TimeInterval, corrections: Int) {
    let bridge = NLTaggerBridge()
    var totalCorrections = 0

    let start = Date()

    for sentence in sentences {
        // Step 1: POS tag with NLTagger
        let tokens = bridge.tokenize(sentence)

        // Step 2: Apply POS-aware grammar correction
        let result = correctGrammarWithPos(tokens: tokens)

        if result != sentence {
            totalCorrections += 1
        }
    }

    let elapsed = Date().timeIntervalSince(start)
    return (elapsed, totalCorrections)
}

// Main benchmark
func runBenchmark() {
    print("=== NLTagger + POS Rules Benchmark ===\n")

    // Try to load Tatoeba corpus, fall back to sample sentences
    var sentences: [String]
    let corpusPath = "bench/corpus/tatoeba_10k.tsv"

    if let tatoeba = loadTatoeba(path: corpusPath, limit: 10000) {
        sentences = tatoeba
        print("Loaded \(sentences.count) sentences from Tatoeba corpus\n")
    } else {
        let sentenceCount = 1000
        sentences = generateSentences(count: sentenceCount)
        print("Generated \(sentences.count) sample sentences (Tatoeba not found)\n")
    }

    // Get rule counts
    let simpleCount = getSimpleRuleCount(language: .english)
    let posCount = getPosRuleCount(language: .english)
    let totalCount = getRuleCount(language: .english)

    print("Rule counts:")
    print("  Simple rules: \(simpleCount)")
    print("  POS rules:    \(posCount)")
    print("  Total:        \(totalCount)")

    // Warm-up
    print("\nWarming up...")
    _ = benchmarkNLTaggerOnly(sentences: Array(sentences.prefix(100)))
    _ = benchmarkYoozSimple(sentences: Array(sentences.prefix(100)))
    _ = benchmarkFullPipeline(sentences: Array(sentences.prefix(100)))

    print("\n=== Benchmark Results (\(sentences.count) sentences) ===\n")

    // 1. NLTagger only
    print("1. NLTagger POS Tagging Only...")
    let (nlTime, nlTokens) = benchmarkNLTaggerOnly(sentences: sentences)
    let nlPerSentence = (nlTime * 1_000_000) / Double(sentences.count)
    print("   Total time: \(String(format: "%.3f", nlTime))s")
    print("   Per sentence: \(String(format: "%.1f", nlPerSentence))µs")
    print("   Tokens: \(nlTokens)")

    print()

    // 2. Yooz simple rules only
    print("2. Yooz Simple Rules (\(simpleCount) rules)...")
    let (simpleTime, simpleCorrections) = benchmarkYoozSimple(sentences: sentences)
    let simplePerSentence = (simpleTime * 1_000_000) / Double(sentences.count)
    print("   Total time: \(String(format: "%.3f", simpleTime))s")
    print("   Per sentence: \(String(format: "%.1f", simplePerSentence))µs")
    print("   Corrections: \(simpleCorrections)")

    print()

    // 3. Full pipeline (NLTagger + POS rules)
    print("3. Full Pipeline: NLTagger + POS Rules (\(totalCount) rules)...")
    let (fullTime, fullCorrections) = benchmarkFullPipeline(sentences: sentences)
    let fullPerSentence = (fullTime * 1_000_000) / Double(sentences.count)
    print("   Total time: \(String(format: "%.3f", fullTime))s")
    print("   Per sentence: \(String(format: "%.1f", fullPerSentence))µs")
    print("   Corrections: \(fullCorrections)")

    print("\n=== Summary ===\n")

    print("| Mode                    | Rules | Per Sentence | Corrections |")
    print("|-------------------------|-------|--------------|-------------|")
    print("| NLTagger only           |   n/a | \(String(format: "%8.1f", nlPerSentence))µs   |         n/a |")
    print("| Yooz simple             | \(String(format: "%5d", simpleCount)) | \(String(format: "%8.1f", simplePerSentence))µs   | \(String(format: "%11d", simpleCorrections)) |")
    print("| NLTagger + Yooz POS     | \(String(format: "%5d", totalCount)) | \(String(format: "%8.1f", fullPerSentence))µs   | \(String(format: "%11d", fullCorrections)) |")

    print()

    let posOverhead = fullPerSentence - simplePerSentence
    let nlPct = (nlPerSentence / fullPerSentence) * 100

    print("POS overhead: \(String(format: "%.1f", posOverhead))µs/sentence")
    print("NLTagger portion: \(String(format: "%.1f", nlPct))% of full pipeline")
    print("Additional corrections with POS: \(fullCorrections - simpleCorrections)")

    print("\n=== Notes ===")
    print("- NLTagger: Apple's built-in NLP (zero binary size, OS-optimized)")
    print("- Simple rules: Fast text matching, no POS required")
    print("- POS rules: Context-aware patterns using NLTagger tags")
    print("- Full pipeline combines NLTagger + simple + POS rules")

    // Show some example corrections
    print("\n=== Sample Corrections ===\n")
    let bridge = NLTaggerBridge()
    let testCases = [
        "I are going home.",
        "She don't like it.",
        "They was here yesterday.",
        "He goed to school."
    ]

    for test in testCases {
        let simple = correctGrammar(text: test)
        let tokens = bridge.tokenize(test)
        let pos = correctGrammarWithPos(tokens: tokens)

        print("Input:  \(test)")
        print("Simple: \(simple)")
        print("POS:    \(pos)")
        print()
    }
}

// Entry point
@main
struct NLTaggerBenchmarkApp {
    static func main() {
        runBenchmark()
    }
}
