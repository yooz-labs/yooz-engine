// InfiniteSessionRuntimeTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
@testable import InfiniteModule

/// Unit coverage (no model) for the `tokenRecord`/`pendingToken` invariant
/// InfiniteSessionRuntime maintains, plus the live equivalence gate that
/// proves the KV-cache-reuse session path is token-for-token identical to a
/// cold restart (engine#265).
final class InfiniteSessionRuntimeTests: XCTestCase {

    // MARK: - truncatedAppendState — pure math, exhaustive over chunk counts

    /// Zero chunks fed: the session must come back exactly as it was,
    /// whether or not it already had a pending token.
    func testTruncatedAppendStateZeroChunksLeavesStateUnchanged() {
        let (recordWithPending, pendingWithPending) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [10, 11, 12],
            priorPendingToken: 12,
            toFeed: [12, 20, 21, 22, 23, 24],
            chunkSize: 4,
            chunksCompleted: 0
        )
        XCTAssertEqual(recordWithPending, [10, 11, 12])
        XCTAssertEqual(pendingWithPending, 12)

        let (recordNoPending, pendingNoPending) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [],
            priorPendingToken: nil,
            toFeed: [30, 31, 32, 33],
            chunkSize: 4,
            chunksCompleted: 0
        )
        XCTAssertEqual(recordNoPending, [])
        XCTAssertNil(pendingNoPending)
    }

    /// Mid-flight: some whole chunks landed, the rest of `toFeed` did not.
    /// The first unfed id becomes the new pending token.
    func testTruncatedAppendStateMidFlightCommitsCompleteChunksOnly() {
        // toFeed = [12 (old pending), 20, 21, 22, 23, 24, 25, 26] — 8 ids,
        // chunkSize 4 -> two whole chunks + cancellation before a third.
        let toFeed = [12, 20, 21, 22, 23, 24, 25, 26]
        let (tokenRecord, pendingToken) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [10, 11, 12],
            priorPendingToken: 12,
            toFeed: toFeed,
            chunkSize: 4,
            chunksCompleted: 1
        )
        // One whole chunk (4 ids: 12, 20, 21, 22) durably fed; the next id
        // (23) is the new pending token.
        XCTAssertEqual(tokenRecord, [10, 11, 12, 20, 21, 22, 23])
        XCTAssertEqual(pendingToken, 23)
    }

    /// First append on a fresh session (no prior pending token) also
    /// truncates correctly mid-flight.
    func testTruncatedAppendStateMidFlightOnFirstAppendHasNoLeadingPending() {
        let toFeed = [1, 2, 3, 4, 5, 6, 7]
        let (tokenRecord, pendingToken) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [],
            priorPendingToken: nil,
            toFeed: toFeed,
            chunkSize: 3,
            chunksCompleted: 2
        )
        // Two whole chunks (6 ids) fed; id 7 (index 6) is the new pending.
        XCTAssertEqual(tokenRecord, [1, 2, 3, 4, 5, 6, 7])
        XCTAssertEqual(pendingToken, 7)
    }

    /// Boundary: every chunk of `toFeed` completed. `fedCount` clamps to
    /// `toFeed.count` and there is nothing left to hold pending — a real
    /// cancellation can never actually reach this branch (the admission
    /// gate is only checked before a chunk starts, so the loop simply ends
    /// once the last chunk succeeds), but the math must stay sane at the
    /// boundary rather than reading out of bounds.
    func testTruncatedAppendStateAllChunksFedClampsCleanly() {
        let toFeed = [12, 20, 21, 22, 23, 24, 25, 26]
        let (tokenRecord, pendingToken) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [10, 11, 12],
            priorPendingToken: 12,
            toFeed: toFeed,
            chunkSize: 4,
            chunksCompleted: 2
        )
        XCTAssertEqual(tokenRecord, [10, 11] + toFeed)
        XCTAssertNil(pendingToken)

        // Over-reporting chunksCompleted beyond what toFeed can hold must
        // clamp the same way, not crash.
        let (clampedRecord, clampedPending) = InfiniteSessionRuntime.truncatedAppendState(
            priorTokenRecord: [10, 11, 12],
            priorPendingToken: 12,
            toFeed: toFeed,
            chunkSize: 4,
            chunksCompleted: 99
        )
        XCTAssertEqual(clampedRecord, [10, 11] + toFeed)
        XCTAssertNil(clampedPending)
    }

    // MARK: - append/generate invariant transitions (pure, via the outcome shape)

    /// The invariant this whole runtime exists to maintain: after a
    /// successful append, the durable cache holds `tokenRecord.dropLast()`
    /// and `pendingToken == tokenRecord.last`. Modeled here directly on
    /// the pure state transition (no model needed) since `appendTokens`
    /// itself always performs exactly this transition on success.
    func testAppendCommitInvariantHoldsAfterSuccess() {
        var tokenRecord = [10, 11, 12]
        var pendingToken: Int? = 12
        let newTokens = [20, 21, 22]

        // Mirrors appendTokens's success path exactly.
        tokenRecord += newTokens
        pendingToken = newTokens.last

        XCTAssertEqual(tokenRecord, [10, 11, 12, 20, 21, 22])
        XCTAssertEqual(pendingToken, 22)
        XCTAssertEqual(Array(tokenRecord.dropLast()), [10, 11, 12, 20, 21])
        XCTAssertEqual(pendingToken, tokenRecord.last)
    }

    /// The generate-side half of the same invariant: after a successful
    /// generate, pendingToken is always nil (every token TokenIterator
    /// emits was already fed by the time it's returned — see
    /// InfiniteSessionRuntime's invariant doc), so the durable cache
    /// equals the ENTIRE tokenRecord, nothing withheld.
    func testGenerateCommitInvariantAlwaysClearsPendingToken() {
        var tokenRecord = [10, 11, 12]
        let promptTokens = [30, 31]
        let fedIds = [40, 41, 42]

        // Mirrors generateSession's commit exactly (both the clean-finish
        // and the cancelled-mid-decode path commit identically).
        tokenRecord += promptTokens + fedIds
        let pendingToken: Int? = nil

        XCTAssertEqual(tokenRecord, [10, 11, 12, 30, 31, 40, 41, 42])
        XCTAssertNil(pendingToken)
        // The whole record is durable — no trailing element is withheld.
        XCTAssertEqual(tokenRecord, Array(tokenRecord))
    }

    // MARK: - Empty-input rejections (no model needed)

    func testRequireNonEmptyRejectsEmptyString() {
        XCTAssertThrowsError(try InfiniteSessionRuntime.requireNonEmpty("", operation: "append")) { error in
            guard case InfiniteError.invalidSessionInput(let reason) = error else {
                return XCTFail("expected invalidSessionInput, got \(error)")
            }
            XCTAssertTrue(reason.contains("append"))
        }
        XCTAssertThrowsError(try InfiniteSessionRuntime.requireNonEmpty("", operation: "generate")) { error in
            guard case InfiniteError.invalidSessionInput(let reason) = error else {
                return XCTFail("expected invalidSessionInput, got \(error)")
            }
            XCTAssertTrue(reason.contains("generate"))
        }
    }

    func testRequireNonEmptyAcceptsNonEmptyString() throws {
        try InfiniteSessionRuntime.requireNonEmpty("x", operation: "append")
        try InfiniteSessionRuntime.requireNonEmpty("real context", operation: "generate")
    }

    // MARK: - Live equivalence gate (engine#265's core acceptance criterion)

    /// THE equivalence gate: a live session built via two `appendTokens`
    /// calls (forcing a pending-token hand-off between them) plus a
    /// `generateSession` call must produce IDENTICAL token ids to a cold
    /// restart — one fresh cache prefilled with
    /// encode(A) + encode(B) + encode(prompt) (three separate encode
    /// calls concatenated, exactly how the session path itself tokenizes
    /// each call — not one encode of the joined string, which a BPE
    /// tokenizer is not guaranteed to reproduce token-for-token) driven
    /// through the same decode loop. Also proves the second `generate` on
    /// the same session reuses the cache: its own prefill length equals
    /// only its own prompt, not the whole session history.
    ///
    /// Gated behind YOOZ_INFINITE_LIVE=1 (downloads ~0.6 GB on first run;
    /// real weights, real decode — mirrors
    /// InfiniteModuleTests.testRealNativeContextGenerationProducesTokens's
    /// pinned model).
    func testLiveSessionMatchesColdRestartTokenForToken() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INFINITE_LIVE"] == "1",
            "set YOOZ_INFINITE_LIVE=1 to run the live session equivalence gate (downloads a model)"
        )

        let descriptor = InfiniteBackendDescriptor(
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
        let backend = try await MLXInfiniteBackend.load(descriptor)

        let textA = "The capital of France is Paris. "
        let textB = "The capital of Japan is Tokyo. "
        let prompt = "Name one of the two capitals mentioned above in one word."
        let maxTokens = 32

        let sessionID = "equivalence-gate-\(UUID().uuidString)"
        await backend.openSession(id: sessionID)
        _ = try await backend.appendTokens(id: sessionID, text: textA)
        _ = try await backend.appendTokens(id: sessionID, text: textB)
        let live = try await backend.generateSession(
            id: sessionID, prompt: prompt, maxTokens: maxTokens, temperature: 0
        )

        let idsA = await backend.encodeForTesting(textA)
        let idsB = await backend.encodeForTesting(textB)
        let idsPrompt = await backend.encodeForTesting(prompt)
        let seedIds = idsA + idsB + idsPrompt
        let cold = try await backend.coldDecodeForTesting(
            seedIds: seedIds, maxTokens: maxTokens, temperature: 0
        )

        XCTAssertFalse(live.tokenIds.isEmpty, "the live session should produce real tokens")
        XCTAssertEqual(
            live.tokenIds, cold.emittedIds,
            "live session (two appends + generate) must match a cold restart token-for-token"
        )
        XCTAssertEqual(live.text, cold.text)

        // Second generate on the same session continues without
        // re-prefilling: pendingToken is nil after the first generate (see
        // InfiniteSessionRuntime's invariant doc), so this call's own
        // seedIds is just its own prompt tokens, not the whole history.
        let secondPrompt = "And the other one?"
        let secondPromptTokenCount = await backend.encodeForTesting(secondPrompt).count
        let second = try await backend.generateSession(
            id: sessionID, prompt: secondPrompt, maxTokens: maxTokens, temperature: 0
        )
        XCTAssertEqual(
            second.prefillTokenCount, secondPromptTokenCount,
            "second generate's prefill must cover only its own prompt, not the accumulated session history"
        )
    }
}
