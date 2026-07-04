// InfiniteTurnCommitLiveTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import XCTest
@testable import InfiniteModule

/// Live gate for turn-commit (engine#267, epic#263 phase 4) — real weights,
/// real decode, gated behind `YOOZ_INFINITE_LIVE=1` (downloads the pinned
/// `Qwen3.5-0.8B-MLX-4bit` on first run, same pin as
/// `InfiniteSessionRuntimeTests`/`InfiniteCheckpointResumeLiveTests`).
final class InfiniteTurnCommitLiveTests: XCTestCase {

    private static let descriptor = InfiniteBackendDescriptor(
        selection: .qwen35B1M,
        repository: InfiniteModelRepository(
            id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            revision: "5d894f8cc4ef3e6c88537bf3746ed262f549da6a"
        ),
        backendKind: .pagedKV,
        adapterKind: .pagedKVMLX,
        nativeContextTokens: 32_768,
        targetContextTokens: 32_768,
        requiredRAMTier: .reduced
    )

    private static func skipUnlessLive() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INFINITE_LIVE"] == "1",
            "set YOOZ_INFINITE_LIVE=1 to run the live turn-commit gate (downloads a model)"
        )
    }

    // MARK: - (a) Template parity: composer fragments == apply_chat_template

    /// Renders a system-less 2-turn conversation (user / assistant-historical
    /// / user) both via `tokenizer.applyChatTemplate` and via composer
    /// fragments (stable historical framing for the closed turn, `userOpen`
    /// for the new one, plus the non-thinking generation prompt matching the
    /// template's own default when `enable_thinking` is left unset) — the
    /// token streams must be IDENTICAL. This is what catches template drift;
    /// if it ever fails, the fix is in the composer's fragment strings, never
    /// in this test.
    func testTemplateParityTwoTurns() async throws {
        try Self.skipUnlessLive()
        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let composer = Qwen35ChatMLComposer()

        let question0 = "What is the capital of France?"
        let answer1 = "The capital of France is Paris."
        let question2 = "What about Japan?"

        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": question0],
            ["role": "assistant", "content": answer1],
            ["role": "user", "content": question2],
        ]
        let templateIds = try await backend.applyChatTemplateForTesting(messages: messages)

        // Encoded per-turn — three separate `encode` calls concatenated,
        // exactly how `generateTurn` itself tokenizes (never one `encode` of
        // the whole joined string, which a BPE tokenizer is not guaranteed
        // to reproduce token-for-token across fragment boundaries).
        let composedIds =
            (await backend.encodeForTesting(composer.userOpen + question0 + composer.userClose))
            + (await backend.encodeForTesting(composer.stableAssistantWrap(answer: answer1)))
            + (await backend.encodeForTesting(composer.userOpen + question2 + composer.userClose))
            + (await backend.encodeForTesting(composer.generationPrompt(thinking: false)))

        XCTAssertEqual(composedIds, templateIds)
    }

    // MARK: - (b) Bit-exactness: the branch decode never touches the durable cache

    /// Session A goes through a real `.turnCommit` branch decode; session B
    /// never branches at all — it is built by directly appending the exact
    /// same committed text (`U + stableAssistantWrap(answer)`, read back
    /// from A's own outcome) onto an identically-primed durable cache. If
    /// `branchCaches` ever aliased the durable cache's storage instead of
    /// copying it, A's branch decode would leave extra state behind that B
    /// never accumulates, and this comparison would diverge.
    func testBranchDecodeLeavesDurableBitExact() async throws {
        try Self.skipUnlessLive()
        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let composer = Qwen35ChatMLComposer()

        let priorText = "The capital of France is Paris. The capital of Japan is Tokyo. "
        let question = "Name one of the two capitals mentioned above in one word."
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)

        let sessionA = "bitexact-a-\(UUID().uuidString)"
        let sessionB = "bitexact-b-\(UUID().uuidString)"
        await backend.openSession(id: sessionA)
        _ = try await backend.appendTokens(id: sessionA, text: priorText)
        await backend.openSession(id: sessionB)
        _ = try await backend.appendTokens(id: sessionB, text: priorText)

        // Sanity precondition: two sessions primed with the same text from
        // fresh caches must already be bit-identical before either does
        // anything turn-commit-specific.
        let primedEqual = await backend.liveCacheStateEqualsForTesting(id: sessionA, matches: sessionB)
        XCTAssertTrue(primedEqual)

        let outcome = try await backend.generateTurn(
            id: sessionA, prompt: question, maxTokens: 64, temperature: 0, policy: .turnCommit
        )

        // Reproduce exactly what the commit step inside generateTurn fed the
        // durable cache: U (the framed user turn) + S (the stable-framed
        // answer) — via the plain `appendTokens` raw-continuation path, never
        // touching a branch.
        let committedText =
            composer.userOpen + trimmedQuestion + composer.userClose
            + composer.stableAssistantWrap(answer: outcome.text)
        _ = try await backend.appendTokens(id: sessionB, text: committedText)

        let finalEqual = await backend.liveCacheStateEqualsForTesting(id: sessionA, matches: sessionB)
        XCTAssertTrue(
            finalEqual,
            "a turn-commit branch decode must leave the durable cache bit-identical to a control "
                + "session that only ever appended the committed text directly"
        )

        // The durable tokenRecord must end with exactly U+S — none of the
        // reasoning tokens the branch spent leaking in.
        let (tokenRecordA, _) = await backend.tokenRecordForTesting(id: sessionA)
        let (tokenRecordB, _) = await backend.tokenRecordForTesting(id: sessionB)
        XCTAssertEqual(tokenRecordA, tokenRecordB)
    }

    // MARK: - (c) Committed-session equivalence: cold prefill of tokenRecord matches the live session

    /// After one `generateTurn(.turnCommit)`, a fresh cache prefilled with
    /// the session's exact `tokenRecord` must produce an identical greedy
    /// continuation to the durable session itself — the Python reference's
    /// committed-equivalence gate. Proves the commit step leaves the cache
    /// in exactly the state a plain prefill of the same tokens would, no
    /// chunk-boundary or cache-type drift.
    ///
    /// The continuation deliberately uses `.thinkingInSession` (thinking
    /// OFF), not a raw `generateSession` continuation: empirically (see
    /// `testThreeTurnConversationReportsThinkingVsCommittedTokens`) this
    /// pinned 0.8B never closes `<think>` for simple factual QA within any
    /// tested budget up to 2048 tokens, so the initial turn's committed
    /// answer is the honest forced-empty `length_in_think` form. A raw-text
    /// continuation right after that empty, untagged-role turn is enough
    /// out-of-distribution context to make the model emit an immediate stop
    /// token (observed directly). A properly role-framed follow-up turn
    /// avoids that and isolates exactly the property under test: does the
    /// committed cache continue like a plain prefill of its own tokenRecord.
    func testCommittedSessionEqualsAnswersOnlyPrefill() async throws {
        try Self.skipUnlessLive()
        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let composer = Qwen35ChatMLComposer()

        let sessionID = "committed-equiv-\(UUID().uuidString)"
        await backend.openSession(id: sessionID)
        _ = try await backend.appendTokens(
            id: sessionID, text: "The capital of France is Paris. The capital of Japan is Tokyo. "
        )
        // A small think budget keeps this fast: the equivalence property
        // under test holds regardless of whether the committed answer is
        // real text or the forced-empty length_in_think form.
        let turnOutcome = try await backend.generateTurn(
            id: sessionID,
            prompt: "Name one of the two capitals mentioned above in one word.",
            maxTokens: 48,
            temperature: 0,
            policy: .turnCommit
        )

        let (tokenRecordBeforeContinuation, _) = await backend.tokenRecordForTesting(id: sessionID)

        let continuationPrompt = "And the other one?"
        let live = try await backend.generateTurn(
            id: sessionID, prompt: continuationPrompt, maxTokens: 32, temperature: 0,
            policy: .thinkingInSession
        )

        // Mirrors generateTurnThinkingInSession's own construction exactly:
        // two SEPARATE encode calls (userOpen+prompt+userClose, then the
        // generation prompt) concatenated, never one encode of the joined
        // string.
        let continuationFragmentIds =
            (await backend.encodeForTesting(composer.userOpen + continuationPrompt + composer.userClose))
            + (await backend.encodeForTesting(composer.generationPrompt(thinking: false)))
        let cold = try await backend.coldDecodeForTesting(
            seedIds: tokenRecordBeforeContinuation + continuationFragmentIds, maxTokens: 32, temperature: 0
        )

        XCTAssertEqual(
            live.text, cold.text,
            "continuing the committed session via .thinkingInSession must match a cold cache "
                + "prefilled with its exact tokenRecord plus the same framed continuation"
        )
        print(
            "[InfiniteTurnCommitLiveTests] committed-equivalence turn: "
                + "finishReason=\(turnOutcome.finishReason) answer=\"\(turnOutcome.text)\" "
                + "thinkingTokens=\(turnOutcome.thinkingTokens ?? -1); continuation=\"\(live.text)\""
        )
    }

    // MARK: - (d) Aggregate: thinking vs committed tokens over a 3-turn conversation

    /// Reports thinking-vs-committed token totals over a 3-turn scripted
    /// conversation on the pinned 0.8B model. NOT a strict gate on think
    /// closure: empirically, this pinned model never closes `<think>` for
    /// simple factual QA at temperature 0 within ANY tested budget up to
    /// 2048 tokens (confirmed directly — see the deviation noted in the PR
    /// description). Every turn must still produce internally-consistent
    /// stats and a non-empty COMMITTED token count (U + stableAssistantWrap,
    /// which is never empty even when the answer itself is forced empty).
    func testThreeTurnConversationReportsThinkingVsCommittedTokens() async throws {
        try Self.skipUnlessLive()
        let backend = try await MLXInfiniteBackend.load(Self.descriptor)

        let sessionID = "aggregate-\(UUID().uuidString)"
        await backend.openSession(id: sessionID)
        _ = try await backend.appendTokens(
            id: sessionID,
            text: "The capital of France is Paris. The capital of Japan is Tokyo. "
                + "The capital of Egypt is Cairo. "
        )

        let questions = [
            "What is the capital of France?",
            "What is the capital of Japan?",
            "What is the capital of Egypt?",
        ]

        var totalThinking = 0
        var totalCommitted = 0
        var unclosedTurns = 0
        var perTurnSummaries: [String] = []

        for (index, question) in questions.enumerated() {
            let outcome = try await backend.generateTurn(
                id: sessionID, prompt: question, maxTokens: 128, temperature: 0, policy: .turnCommit
            )
            if outcome.finishReason == "length_in_think" {
                unclosedTurns += 1
            }
            let thinkingTokens = try XCTUnwrap(outcome.thinkingTokens, "turnCommit always reports thinkingTokens")
            let committedTokens = try XCTUnwrap(outcome.committedTokens, "turnCommit always reports committedTokens")
            // Never empty: U + stableAssistantWrap always includes the
            // open/close role markers, even when the answer itself is
            // forced empty by an unclosed think block.
            XCTAssertGreaterThan(committedTokens, 0)
            totalThinking += thinkingTokens
            totalCommitted += committedTokens
            perTurnSummaries.append(
                "turn \(index + 1): finishReason=\(outcome.finishReason) think=\(thinkingTokens) "
                    + "committed=\(committedTokens) commit=\(String(format: "%.3f", outcome.commitSeconds ?? 0))s "
                    + "answer=\"\(outcome.text)\""
            )
        }

        let closureNote = unclosedTurns == questions.count
            ? "this pinned 0.8B never closed <think> for any turn within the 128-token budget - "
                + "reported honestly, matches the engine's documented v1 deviation from the Python "
                + "reference's forced-close fallback, not a gate failure"
            : "at least one turn closed its think block normally"
        print(
            "[InfiniteTurnCommitLiveTests] 3-turn aggregate - "
                + "thinkingTokens=\(totalThinking), committedTokens=\(totalCommitted), "
                + "unclosedTurns=\(unclosedTurns)/\(questions.count) (\(closureNote)):\n"
                + perTurnSummaries.joined(separator: "\n")
        )
    }
}
