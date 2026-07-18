// TouchUpFidelityEvalTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Real-model generation-fidelity harness for the Quality backend
// (engine #277 / whisper #313, Phase 2 of the touch-up faithfulness epic).
// Gated behind YOOZ_LLM_LOAD_MODELS=1 like the rest of the model-dependent
// suite in LLMModuleTests.swift — no mocks, the real Yooz-Quality-v2
// weights, per project policy.
//
// Fixture provenance: the corruption inputs are the SAME real, log-id-
// anchored dictation phrases Phase 1 pinned in Tests/AlignmentGuardTests.swift
// (whisper repo), which trace back to
// `.context/research-touchup-faithfulness-2026-07-18.md`'s Finding 2 table
// (build-64 Air debug logs). Reusing the exact wording — rather than
// re-deriving approximate phrasing from that table's crude fragments —
// keeps the two phases' fixtures byte-for-byte comparable: Phase 1 measures
// whether the whisper-side guard repairs a corrupted output back to the
// input; this harness measures whether the engine's raw Quality-model
// output avoids the corruption in the first place, so Phase 1's
// `guardRepairedSentences` firing rate is the live metric for whether
// Stage 1 here actually worked in production.
//
// All corruption evidence in that research doc is from FULL touch-up mode
// (Finding 1: "all 274 Air full-mode dictations ran the LLM"), so every
// fixture below runs against `qualityFull` — the prompt tier the real
// corruption was observed under.

import XCTest
@testable import LLMModule

final class TouchUpFidelityEvalTests: XCTestCase {

    private var shouldLoadRealModels: Bool {
        ProcessInfo.processInfo.environment["YOOZ_LLM_LOAD_MODELS"] == "1"
    }

    // MARK: - Fixture data (reusable across assertions and before/after runs)

    /// One dictation input evaluated against a touch-up system prompt.
    /// `isClean` marks an already-correct input, which additionally must
    /// come back near-identical (assertion e).
    struct Fixture {
        let id: String
        let input: String
        let systemPrompt: String
        let isClean: Bool
    }

    static let fixtures: [Fixture] = [
        // Corruption-pattern fixtures — verbatim inputs from
        // AlignmentGuardTests.swift's "must fire and repair" section.
        Fixture(
            id: "weNeedMoreSpace",
            input: "we need more space.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "mightEndUpNeedingSome",
            input: "Please note that we might end up needing some.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "bankStatementClause",
            input: "I need to check my credit card statement before I decide, or something like that.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "possessiveDropped",
            input: "Don't change other people's code without asking.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "oneIsDigitized",
            input: "One is correct.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "oneYearHyphenArtifact",
            input: "It took one year to finish.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "notNormalizedButDropped",
            input: "The data is not normalized, but it is usable.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        Fixture(
            id: "intelDropped",
            input: "I use a Linux Intel system for development.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: false
        ),
        // Clean / already-correct fixtures (sourced from AlignmentGuardTests'
        // "must NOT fire" examples, used here as standalone inputs).
        Fixture(
            id: "cleanFineAsIs",
            input: "This is fine as is.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: true
        ),
        Fixture(
            id: "cleanSystemWorking",
            input: "The system is working correctly.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: true
        ),
        Fixture(
            id: "cleanRobinHood",
            input: "We went to see Robin Hood and the main event.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: true
        ),
        Fixture(
            id: "cleanCleaningHouse",
            input: "I've been cleaning the house all day.",
            systemPrompt: YoozPrompts.qualityFull,
            isClean: true
        )
    ]

    // MARK: - Per-fixture outcome (reusable across before/after runs)

    struct FixtureOutcome {
        let fixtureID: String
        let output: String
        let noNewClauseClone: Bool
        let wordCountRatioOK: Bool
        let singleDigitsPreserved: Bool
        let validJSON: Bool
        /// nil unless the fixture is `isClean`.
        let similarityOK: Bool?

        var allPass: Bool {
            noNewClauseClone && wordCountRatioOK && singleDigitsPreserved
                && validJSON && (similarityOK ?? true)
        }
    }

    // MARK: - The harness (real model, gated)

    /// Runs every fixture through the real Yooz-Quality backend and checks
    /// all five faithfulness assertions. Run this once against the
    /// pre-Stage-1 checkout and once against the post-Stage-1 checkout
    /// (`git stash` / checkout the parent commit for the "before" run) to
    /// produce the before/after tables for the PR body — the printed
    /// result table is deliberately unconditional (not gated behind a
    /// verbose flag) so a captured stdout log doubles as that table.
    func testQualityFullFidelityAgainstFixtures() async throws {
        try XCTSkipUnless(
            shouldLoadRealModels,
            "Set YOOZ_LLM_LOAD_MODELS=1 to run the Quality generation-fidelity eval"
        )

        let backend = MLXLLMBackend.createQuality()
        try await backend.load()

        var outcomes: [FixtureOutcome] = []
        for fixture in Self.fixtures {
            let raw = try await backend.generate(prompt: fixture.input, systemPrompt: fixture.systemPrompt)
            let (parsedText, success) = parseProofreadResponse(raw, fallback: fixture.input)
            let checkedText = success ? parsedText : raw

            outcomes.append(
                FixtureOutcome(
                    fixtureID: fixture.id,
                    output: checkedText,
                    noNewClauseClone: !Self.introducesDuplicateClause(
                        input: fixture.input, output: checkedText
                    ),
                    wordCountRatioOK: Self.wordCountRatio(
                        input: fixture.input, output: checkedText
                    ) >= 0.7,
                    singleDigitsPreserved: !Self.introducesDigitizedSingleDigit(
                        input: fixture.input, output: checkedText
                    ),
                    validJSON: success,
                    similarityOK: fixture.isClean
                        ? Self.similarity(fixture.input, checkedText) >= 0.9
                        : nil
                )
            )
        }

        Self.logResultsTable(outcomes)

        for outcome in outcomes {
            XCTAssertTrue(
                outcome.noNewClauseClone,
                "[\(outcome.fixtureID)] output introduces a duplicated clause not present in the input: \(outcome.output)"
            )
            XCTAssertTrue(
                outcome.wordCountRatioOK,
                "[\(outcome.fixtureID)] output dropped too much content (< 0.7x input words): \(outcome.output)"
            )
            XCTAssertTrue(
                outcome.singleDigitsPreserved,
                "[\(outcome.fixtureID)] LLM digitized a single-digit number word: \(outcome.output)"
            )
            XCTAssertTrue(
                outcome.validJSON,
                "[\(outcome.fixtureID)] response did not parse via parseProofreadResponse: \(outcome.output)"
            )
            if let similarityOK = outcome.similarityOK {
                XCTAssertTrue(
                    similarityOK,
                    "[\(outcome.fixtureID)] clean input diverged too much from output: \(outcome.output)"
                )
            }
        }
    }

    // MARK: - Assertion helpers (reusable, pure functions — no model dependency)

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    private static func ngramCounts(_ words: [String], n: Int) -> [String: Int] {
        guard words.count >= n else { return [:] }
        var counts: [String: Int] = [:]
        for i in 0...(words.count - n) {
            let gram = words[i..<(i + n)].joined(separator: " ")
            counts[gram, default: 0] += 1
        }
        return counts
    }

    /// (a) Clause-clone detector, modeled after `AlignmentGuard`'s
    /// `duplicationIntroduced` (see its doc in AlignmentGuardTests.swift):
    /// a word n-gram (3 to 8 words) that occurs MORE often in the output
    /// than in the input is a clone the LLM introduced. Comparing counts
    /// (rather than flagging any repeat, or requiring the repeat to be
    /// immediately back-to-back) avoids false positives on transcripts
    /// that legitimately repeat a short phrase across sentences, and on
    /// inputs that already contained a repeated clause the model
    /// legitimately deduplicated (repeated count going DOWN is fine).
    static func introducesDuplicateClause(input: String, output: String, minWords: Int = 3) -> Bool {
        let inputWords = tokenize(input)
        let outputWords = tokenize(output)
        guard outputWords.count >= minWords else { return false }
        let maxN = min(8, outputWords.count)
        guard minWords <= maxN else { return false }
        for n in minWords...maxN {
            let outputCounts = ngramCounts(outputWords, n: n)
            let inputCounts = ngramCounts(inputWords, n: n)
            for (gram, count) in outputCounts where count > (inputCounts[gram] ?? 0) {
                if count > 1 { return true }
            }
        }
        return false
    }

    /// (b) Drop detector: output word count relative to input word count,
    /// after removing "uh"/"um" fillers from the input side so legitimate
    /// filler removal is never mistaken for a content drop.
    static func wordCountRatio(input: String, output: String) -> Double {
        let fillers: Set<String> = ["uh", "um"]
        let inputWords = tokenize(input).filter { !fillers.contains($0) }
        let outputWords = tokenize(output)
        guard !inputWords.isEmpty else { return 1.0 }
        return Double(outputWords.count) / Double(inputWords.count)
    }

    /// (c) Number-convention guard: the Rust `.numbers` grammar pass
    /// (`java_rules.rs:1275-1299`) deliberately keeps single digits
    /// (zero-nine) spelled out; only IT converts words -> digits, and only
    /// for multi-digit / measurement contexts. If a standalone single-digit
    /// numeral (0-9) shows up in the output but that literal digit wasn't
    /// already present in the input, the LLM did the digitizing itself —
    /// exactly the behavior the training prompts never asked for.
    static func introducesDigitizedSingleDigit(input: String, output: String) -> Bool {
        let inputTokens = Set(tokenize(input))
        let outputTokens = Set(tokenize(output))
        for digit in 0...9 {
            let token = String(digit)
            if outputTokens.contains(token) && !inputTokens.contains(token) {
                return true
            }
        }
        return false
    }

    /// (e) Word-level Ratcliff/Obershelp-style similarity: 2 * (longest
    /// common subsequence length) / (len(a) + len(b)) — the same shape as
    /// Python's difflib.SequenceMatcher.ratio(), implemented here as a
    /// small test-only helper rather than a new production dependency.
    static func similarity(_ first: String, _ second: String) -> Double {
        let wordsA = tokenize(first)
        let wordsB = tokenize(second)
        if wordsA.isEmpty && wordsB.isEmpty { return 1.0 }
        if wordsA.isEmpty || wordsB.isEmpty { return 0.0 }

        var dp = Array(repeating: Array(repeating: 0, count: wordsB.count + 1), count: wordsA.count + 1)
        for i in 1...wordsA.count {
            for j in 1...wordsB.count {
                if wordsA[i - 1] == wordsB[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        let longestCommonSubsequence = dp[wordsA.count][wordsB.count]
        return (2.0 * Double(longestCommonSubsequence)) / Double(wordsA.count + wordsB.count)
    }

    private static func logResultsTable(_ outcomes: [FixtureOutcome]) {
        print("=== TouchUpFidelityEvalTests results ===")
        print("fixture | clauseClone | wordRatio | digits | json | similarity | PASS")
        for outcome in outcomes {
            let similarityColumn = outcome.similarityOK.map { $0 ? "pass" : "FAIL" } ?? "n/a"
            print(
                "\(outcome.fixtureID) | \(outcome.noNewClauseClone ? "pass" : "FAIL")"
                    + " | \(outcome.wordCountRatioOK ? "pass" : "FAIL")"
                    + " | \(outcome.singleDigitsPreserved ? "pass" : "FAIL")"
                    + " | \(outcome.validJSON ? "pass" : "FAIL")"
                    + " | \(similarityColumn)"
                    + " | \(outcome.allPass ? "PASS" : "FAIL")"
            )
        }
    }
}
