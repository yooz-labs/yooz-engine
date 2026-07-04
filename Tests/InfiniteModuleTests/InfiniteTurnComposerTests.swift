// InfiniteTurnComposerTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import InfiniteModule

/// Unit coverage (no model) for the turn-commit composers (engine#267):
/// `splitThinking`'s text-shape contract, mirrored exactly from
/// `d1_cache/turns.py`'s `split_thinking` (infinite repo), plus the
/// fragment-shape guarantees both composers must hold for
/// `MLXInfiniteBackend.generateTurn` to compose correctly.
final class InfiniteTurnComposerTests: XCTestCase {

    // MARK: - Qwen35ChatMLComposer.splitThinking — exhaustive

    func testSplitThinkingClosedThinkBlock() {
        let composer = Qwen35ChatMLComposer()
        let (reasoning, answer) = composer.splitThinking(
            "<think>\nlet me reason\n</think>\n\nThe answer."
        )
        XCTAssertEqual(reasoning, "let me reason")
        XCTAssertEqual(answer, "The answer.")
    }

    func testSplitThinkingNoThinkAtAll() {
        let composer = Qwen35ChatMLComposer()
        let (reasoning, answer) = composer.splitThinking("Just a direct answer, no think block.")
        XCTAssertEqual(reasoning, "")
        XCTAssertEqual(answer, "Just a direct answer, no think block.")
    }

    /// No `</think>` at all (budget ended mid-thought): the whole text is
    /// the "answer" per the Python contract's no-close branch — the caller
    /// (`MLXInfiniteBackend.generateTurn`) is responsible for recognizing
    /// this as unclosed and never actually committing it as an answer
    /// (`finishReason == "length_in_think"`), not this split itself.
    func testSplitThinkingUnterminatedThinkReturnsWholeTextAsAnswer() {
        let composer = Qwen35ChatMLComposer()
        let (reasoning, answer) = composer.splitThinking("<think>\nstill reasoning, never closes")
        XCTAssertEqual(reasoning, "")
        XCTAssertEqual(answer, "<think>\nstill reasoning, never closes")
    }

    /// Multiple think/answer cycles in one decode: reasoning splits on the
    /// FIRST `</think>`, answer splits on the LAST — everything in between
    /// (a second reasoning cycle plus whatever text preceded it) is
    /// silently dropped, exactly matching `text.split("</think>")[0]` vs
    /// `text.split("</think>")[-1]` in the Python reference.
    func testSplitThinkingMultipleThinkBlocksLastCloseWinsForAnswer() {
        let composer = Qwen35ChatMLComposer()
        let (reasoning, answer) = composer.splitThinking(
            "<think>R1</think>A1<think>R2</think>A2"
        )
        XCTAssertEqual(reasoning, "R1")
        XCTAssertEqual(answer, "A2")
    }

    func testSplitThinkingEmptyAnswerAfterThink() {
        let composer = Qwen35ChatMLComposer()
        let (reasoning, answer) = composer.splitThinking("<think>reasoning here</think>")
        XCTAssertEqual(reasoning, "reasoning here")
        XCTAssertEqual(answer, "")
    }

    // MARK: - Gemma4Composer.splitThinking — identity

    func testGemma4SplitThinkingIsIdentity() {
        let composer = Gemma4Composer()
        let (reasoning, answer) = composer.splitThinking("<think>looks like reasoning but isn't</think>done")
        XCTAssertEqual(reasoning, "")
        XCTAssertEqual(answer, "<think>looks like reasoning but isn't</think>done")
    }

    // MARK: - Fragment shape guarantees

    func testQwenComposerFragmentsAreNonEmpty() {
        let composer = Qwen35ChatMLComposer()
        XCTAssertFalse(composer.userOpen.isEmpty)
        XCTAssertFalse(composer.userClose.isEmpty)
        XCTAssertFalse(composer.generationPrompt(thinking: true).isEmpty)
        XCTAssertFalse(composer.generationPrompt(thinking: false).isEmpty)
        XCTAssertFalse(composer.stableAssistantWrap(answer: "hi").isEmpty)
    }

    func testGemma4ComposerFragmentsAreNonEmpty() {
        let composer = Gemma4Composer()
        XCTAssertFalse(composer.userOpen.isEmpty)
        XCTAssertFalse(composer.userClose.isEmpty)
        XCTAssertFalse(composer.generationPrompt(thinking: true).isEmpty)
        XCTAssertFalse(composer.generationPrompt(thinking: false).isEmpty)
        XCTAssertFalse(composer.stableAssistantWrap(answer: "hi").isEmpty)
    }

    /// The historical (durable-commit) rendering must never carry a think
    /// marker, regardless of what the model actually produced — turn-commit's
    /// entire point is quarantining reasoning out of the durable session.
    func testStableAssistantWrapNeverContainsThinkMarkers() {
        for composer in [Qwen35ChatMLComposer() as any InfiniteTurnComposer, Gemma4Composer()] {
            let wrapped = composer.stableAssistantWrap(answer: "the final answer")
            XCTAssertFalse(wrapped.contains("<think>"))
            XCTAssertFalse(wrapped.contains("</think>"))
        }
    }

    func testStableAssistantWrapTrimsAnswer() {
        let composer = Qwen35ChatMLComposer()
        XCTAssertEqual(
            composer.stableAssistantWrap(answer: "  padded  \n"),
            "<|im_start|>assistant\npadded<|im_end|>\n"
        )
    }

    /// Gemma4's generation prompt ignores the `thinking` argument entirely —
    /// the template has no per-turn think toggle (see `Gemma4Composer`'s doc).
    func testGemma4GenerationPromptIgnoresThinkingArgument() {
        let composer = Gemma4Composer()
        XCTAssertEqual(composer.generationPrompt(thinking: true), composer.generationPrompt(thinking: false))
    }

    /// Qwen's generation prompt DOES vary with `thinking` — this is what lets
    /// `MLXInfiniteBackend.generateTurn` detect "does this composer support
    /// thinking at all" generically (`generationPrompt(thinking: true) !=
    /// generationPrompt(thinking: false)`).
    func testQwenGenerationPromptVariesWithThinkingArgument() {
        let composer = Qwen35ChatMLComposer()
        XCTAssertNotEqual(composer.generationPrompt(thinking: true), composer.generationPrompt(thinking: false))
    }

    // MARK: - InfiniteModelSelection.turnComposer routing

    func testTurnComposerRoutingByModelFamily() {
        XCTAssertTrue(InfiniteModelSelection.qwen35B1M.turnComposer is Qwen35ChatMLComposer)
        XCTAssertTrue(InfiniteModelSelection.gemma4E4B1M.turnComposer is Gemma4Composer)
        XCTAssertTrue(InfiniteModelSelection.gemma4_26B_A4B1M.turnComposer is Gemma4Composer)
        XCTAssertTrue(InfiniteModelSelection.gemma4_12B1M.turnComposer is Gemma4Composer)
    }
}
