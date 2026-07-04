// InfiniteCheckpointResumeLiveTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import XCTest
@testable import InfiniteModule

/// Live checkpoint/park/resume/fork gate (engine#266) — real weights, real
/// decode, gated behind `YOOZ_INFINITE_LIVE=1` since it downloads the pinned
/// `Qwen3.5-0.8B-MLX-4bit` model on first run (mirrors
/// `InfiniteSessionRuntimeTests`'s live equivalence gate, same model pin).
/// Qwen3.5 is the hybrid architecture (`MambaCache` GDN layers interleaved
/// with `KVCacheSimple` full-attention layers, see `Qwen35.swift.newCache`)
/// — the hardest case for a KV-cache save/load round trip.
///
/// These exercise `InfiniteEngine`'s real `append`/`checkpoint`/`resume`/
/// `fork` orchestration (not just the backend primitives), via
/// `installBackendForTesting` — a test-only seam that installs the tiny
/// pinned model as the resident backend. Unlike `MLXInfiniteBackend`'s own
/// "the selection label is cosmetic" pattern (safe there because nothing
/// re-derives the label from the backend), the label given to
/// `installBackendForTesting` MUST equal the backend's own `.selection`
/// (`.qwen35B1M`, matching `Self.descriptor.selection` below): `loadBackend`'s
/// fast-path cache check is `loaded.selection == selection` — the backend's
/// OWN baked-in identity, not the engine's separately-tracked `loadedModel`
/// — so a mismatched label makes every `append`/`checkpoint`/`resume` call
/// think it's a cache miss and silently kick off a REAL reload of that
/// label's actual (multi-GB) weights in the background, racing the test
/// against a wrong model. `.qwen35B1M` requires the `.full` RAM tier, so
/// these tests skip on a `.reduced`-tier machine.
final class InfiniteCheckpointResumeLiveTests: XCTestCase {

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

    /// Must match `descriptor.selection` — see the class doc's
    /// `installBackendForTesting` caveat. `createSession(modelId:)` wants
    /// the wire id.
    private static let engineModelId = InfiniteModelSelection.qwen35B1M.rawValue

    private var tempRoot: URL!
    private var store: InfiniteSessionStore!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("InfiniteCheckpointResumeLiveTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempRoot = dir
        store = InfiniteSessionStore(root: dir)
    }

    override func tearDownWithError() throws {
        if let tempRoot, FileManager.default.fileExists(atPath: tempRoot.path) {
            try FileManager.default.removeItem(at: tempRoot)
        }
        tempRoot = nil
        store = nil
    }

    private static func skipUnlessLive() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INFINITE_LIVE"] == "1",
            "set YOOZ_INFINITE_LIVE=1 to run the live checkpoint/resume/fork gate (downloads a model)"
        )
        // `installBackendForTesting`'s label must equal the injected
        // backend's own `.selection` (`.qwen35B1M`, see the class doc) —
        // that selection's real catalog descriptor requires `.full`.
        try XCTSkipUnless(
            InfiniteRAMTier.current == .full,
            "this live gate needs the .full (64 GiB) RAM tier; current tier is \(InfiniteRAMTier.current.rawValue)"
        )
    }

    /// ~1000+ tokens per call — two appends puts a session around ~2K
    /// tokens, forcing real GDN recurrent-state growth on the hybrid
    /// MambaCache/KVCacheSimple layers (mirrors
    /// `InfiniteSessionRuntimeTests`'s equivalence-gate style, scaled up).
    private func longContextText(sentences: Int, seed: String) -> String {
        (0..<sentences)
            .map { "Fact \(seed)-\($0): the capital of country \(seed)-\($0) is city \(seed)-\($0). " }
            .joined()
    }

    // MARK: - 1. Checkpoint(park) + resume greedy equivalence

    func testCheckpointResumeGreedyEquivalence() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let textA = longContextText(sentences: 130, seed: "a")
        let textB = longContextText(sentences: 130, seed: "b")
        let prompt = "Name one of the cities mentioned above in one word."
        let maxTokens = 64

        // (a) the resumed path: two appends, checkpoint(park: true), resume.
        let session = try await engine.createSession(
            request: InfiniteCreateSessionRequest(modelId: Self.engineModelId, label: "resume-path")
        )
        _ = try await engine.append(sessionID: session.id, request: InfiniteAppendSessionRequest(text: textA))
        _ = try await engine.append(sessionID: session.id, request: InfiniteAppendSessionRequest(text: textB))

        let checkpointResponse = try await engine.checkpoint(
            sessionID: session.id, request: InfiniteCheckpointSessionRequest(label: "before-resume", park: true)
        )
        let parkedInfo = checkpointResponse.session
        XCTAssertEqual(parkedInfo.state, "parked")

        let resumeStart = Date.timeIntervalSinceReferenceDate
        let resumedInfo = try await engine.resume(
            sessionID: session.id, request: InfiniteResumeSessionRequest(checkpointId: nil)
        )
        let resumeDuration = Date.timeIntervalSinceReferenceDate - resumeStart
        XCTAssertEqual(resumedInfo.state, "open")

        // The resumed MambaCache (GDN) layers must keep their fp32
        // recurrent state across the safetensors round trip
        // (GatedDelta.swift: "state kept in fp32 to match Python mlx-lm").
        // Each MambaCache holds two state arrays (`ArraysCache(size: 2)`):
        // [0] the short-conv cache, in the model's compute dtype
        // (bfloat16); [1] the GDN recurrent state, which must stay fp32 —
        // confirmed against a live (never-checkpointed) session via a
        // throwaway diagnostic before writing this assertion.
        let layerDTypes = await backend.liveCacheLayerDTypesForTesting(id: session.id)
        let mambaLayers = layerDTypes.filter { $0.className.contains("MambaCache") }
        XCTAssertFalse(mambaLayers.isEmpty, "expected at least one MambaCache (GDN) layer on Qwen3.5")
        for layer in mambaLayers {
            XCTAssertEqual(layer.dtypes.count, 2, "expected [convState, recurrentState], got \(layer.dtypes)")
            XCTAssertEqual(
                layer.dtypes.last, .float32,
                "resumed MambaCache recurrent state must stay .float32, got \(layer.dtypes)"
            )
        }

        // Token-exact comparison needs the backend's own tokenIds (the
        // engine's wire response is text-only); the resumed session's live
        // state now lives under the same id in `backend`, so calling it
        // directly continues the very state `resume` just installed.
        let resumed = try await backend.generateSession(
            id: session.id, prompt: prompt, maxTokens: maxTokens, temperature: 0
        )

        // (b) the same two appends, generated WITHOUT any park/resume cycle.
        let uninterruptedID = "uninterrupted-\(session.id)"
        await backend.openSession(id: uninterruptedID)
        _ = try await backend.appendTokens(id: uninterruptedID, text: textA)
        _ = try await backend.appendTokens(id: uninterruptedID, text: textB)
        let uninterrupted = try await backend.generateSession(
            id: uninterruptedID, prompt: prompt, maxTokens: maxTokens, temperature: 0
        )

        // (c) a cold restart control prefilled from the same id stream.
        let idsA = await backend.encodeForTesting(textA)
        let idsB = await backend.encodeForTesting(textB)
        let idsPrompt = await backend.encodeForTesting(prompt)
        let cold = try await backend.coldDecodeForTesting(
            seedIds: idsA + idsB + idsPrompt, maxTokens: maxTokens, temperature: 0
        )

        XCTAssertFalse(resumed.tokenIds.isEmpty, "resumed generation should produce real tokens")
        XCTAssertEqual(
            resumed.tokenIds, uninterrupted.tokenIds,
            "checkpoint(park)+resume must not change the session's generated output"
        )
        XCTAssertEqual(
            resumed.tokenIds, cold.emittedIds,
            "checkpoint(park)+resume must match a cold restart token-for-token"
        )

        print(
            "[InfiniteCheckpointResumeLiveTests] checkpoint sizeBytes=\(checkpointResponse.sizeBytes), "
                + "tokenCount=\(checkpointResponse.tokenCount), "
                + "save=\(String(format: "%.3f", checkpointResponse.durationSeconds))s, "
                + "resume=\(String(format: "%.3f", resumeDuration))s "
                + "(reference anchors: Python save 1.2-4.8s for 1.8-7.0 GiB @ 262K-1M tokens on the 35B; "
                + "this 0.8B/~\(checkpointResponse.tokenCount)-token checkpoint is far smaller — reported for the record, not as a gate)"
        )
    }

    // MARK: - 2. Fork diverges, parent stays intact

    func testForkDivergesParentIntact() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let textA = longContextText(sentences: 40, seed: "shared")
        let parent = try await engine.createSession(
            request: InfiniteCreateSessionRequest(modelId: Self.engineModelId, label: "fork-parent")
        )
        _ = try await engine.append(sessionID: parent.id, request: InfiniteAppendSessionRequest(text: textA))

        let parentCheckpoint = try await engine.checkpoint(
            sessionID: parent.id, request: InfiniteCheckpointSessionRequest(park: true)
        )
        let parentCacheURL = try store
            .checkpointDirectory(session: parent.id, checkpoint: parentCheckpoint.checkpoint.id)
            .appendingPathComponent("cache.safetensors", isDirectory: false)
        let parentBytesBefore = try Data(contentsOf: parentCacheURL)

        let forked = try await engine.fork(sessionID: parent.id, request: InfiniteForkSessionRequest())
        XCTAssertNotEqual(forked.id, parent.id)
        XCTAssertEqual(forked.state, "parked", "fork must not auto-resume the new session")

        _ = try await engine.resume(sessionID: parent.id, request: InfiniteResumeSessionRequest())
        _ = try await engine.resume(sessionID: forked.id, request: InfiniteResumeSessionRequest())

        let promptParent = "Reply with exactly one word describing the topic above."
        let promptForked = "Reply with exactly one different word describing the topic above."

        let parentContinuation = try await backend.generateSession(
            id: parent.id, prompt: promptParent, maxTokens: 32, temperature: 0
        )
        let forkedContinuation = try await backend.generateSession(
            id: forked.id, prompt: promptForked, maxTokens: 32, temperature: 0
        )

        let idsA = await backend.encodeForTesting(textA)
        let coldParent = try await backend.coldDecodeForTesting(
            seedIds: idsA + (await backend.encodeForTesting(promptParent)), maxTokens: 32, temperature: 0
        )
        let coldForked = try await backend.coldDecodeForTesting(
            seedIds: idsA + (await backend.encodeForTesting(promptForked)), maxTokens: 32, temperature: 0
        )

        XCTAssertEqual(
            parentContinuation.tokenIds, coldParent.emittedIds,
            "resumed parent must match its own uninterrupted control"
        )
        XCTAssertEqual(
            forkedContinuation.tokenIds, coldForked.emittedIds,
            "resumed fork must match its own uninterrupted control from the same shared seed"
        )

        // Diverge the fork further, then confirm the parent's ORIGINAL
        // checkpoint file bytes are untouched — different session id,
        // different directory tree entirely.
        _ = try await engine.append(
            sessionID: forked.id, request: InfiniteAppendSessionRequest(text: "One more fact after the fork. ")
        )
        _ = try await engine.checkpoint(sessionID: forked.id, request: InfiniteCheckpointSessionRequest())

        let parentBytesAfter = try Data(contentsOf: parentCacheURL)
        XCTAssertEqual(parentBytesBefore, parentBytesAfter, "forking + diverging must not mutate the parent's checkpoint")
    }

    // MARK: - 3. Integrity gate rejects a tampered checkpoint

    func testIntegrityGateRejectsTamperedTokens() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let session = try await engine.createSession(
            request: InfiniteCreateSessionRequest(modelId: Self.engineModelId, label: "integrity-gate")
        )
        _ = try await engine.append(
            sessionID: session.id,
            request: InfiniteAppendSessionRequest(text: longContextText(sentences: 20, seed: "tamper"))
        )
        let checkpointResponse = try await engine.checkpoint(
            sessionID: session.id, request: InfiniteCheckpointSessionRequest(park: true)
        )
        let checkpointID = checkpointResponse.checkpoint.id

        let tokensURL = try store.checkpointDirectory(session: session.id, checkpoint: checkpointID)
            .appendingPathComponent("tokens.bin", isDirectory: false)
        var tokensData = try Data(contentsOf: tokensURL)
        XCTAssertFalse(tokensData.isEmpty)
        tokensData[0] ^= 0xFF
        try tokensData.write(to: tokensURL)

        do {
            _ = try await engine.resume(
                sessionID: session.id, request: InfiniteResumeSessionRequest(checkpointId: checkpointID)
            )
            XCTFail("expected checkpointIntegrity for a tampered tokens.bin")
        } catch InfiniteError.checkpointIntegrity {
            // expected
        }

        // The failed resume must not have installed a live session: the
        // record reverts to its pre-resume (`.parked`) state.
        let status = try await engine.session(id: session.id)
        XCTAssertEqual(status.state, "parked")
    }

    // MARK: - 4. Model switch refused while a session is generating

    /// A model switch that would orphan the loaded backend while one of its
    /// sessions is mid-generate must REFUSE with `sessionBusy` (engine#266
    /// review): the orphaned instance would deallocate after the in-flight
    /// call and take the session's live KV state with it. The refusal fires
    /// in `loadBackend` BEFORE any weights for the requested model are
    /// touched, so this test never downloads the other catalog row.
    func testModelSwitchRefusedWhileGenerating() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let generating = try await engine.createSession(
            request: InfiniteCreateSessionRequest(modelId: Self.engineModelId, label: "mid-generate")
        )
        _ = try await engine.append(
            sessionID: generating.id, request: InfiniteAppendSessionRequest(text: "warm me up")
        )
        await engine.setGeneratingForTesting(id: generating.id)

        // A session on a DIFFERENT catalog row: its append must load that
        // row's backend, which would orphan the qwen backend mid-generate.
        let other = try await engine.createSession(
            request: InfiniteCreateSessionRequest(
                modelId: InfiniteModelSelection.gemma4E4B1M.rawValue, label: "switcher"
            )
        )
        do {
            _ = try await engine.append(
                sessionID: other.id, request: InfiniteAppendSessionRequest(text: "hello")
            )
            XCTFail("model switch while a session is generating must throw sessionBusy")
        } catch let error as InfiniteError {
            guard case .sessionBusy(let id) = error else {
                XCTFail("expected sessionBusy, got \(error)")
                return
            }
            XCTAssertEqual(id, generating.id, "the refusal must name the generating session")
        }

        // The refusal is backpressure, not a load failure: status must not
        // report a lastError. (status.modelId reflects the PICKER's active
        // row, which createSession(modelId:) legitimately moved to the
        // gemma row — it says nothing about which backend is resident, so
        // it is not asserted here. Residency is evidenced by the refusal
        // itself: sessionBusy fired before any gemma weights were touched.)
        let status = await engine.status()
        XCTAssertNil(status.lastError)
        // The refused switcher session reverts to `.open` (its true
        // pre-operation state — it was freshly created, never parked, so
        // `.open` is correct here, not `.parked`; see the `becameLive`
        // doc comment on `append` for the bug this specifically guards
        // against), and the mid-generate session is untouched by the
        // refusal.
        let switcher = try await engine.session(id: other.id)
        XCTAssertEqual(switcher.state, "open")
        let untouched = try await engine.session(id: generating.id)
        XCTAssertEqual(untouched.state, "generating")

        // Once the busy session clears, the identical switch must succeed
        // and park it — the existing (unchanged) backend-eviction-parking
        // behavior, now reachable because the refusal above no longer
        // silently skips it. This loads the real (cached) gemma4-e4b
        // weights, evicting the qwen backend.
        await engine.setOpenForTesting(id: generating.id)

        _ = try await engine.append(
            sessionID: other.id, request: InfiniteAppendSessionRequest(text: "hello again")
        )
        let parkedAfterSwitch = try await engine.session(id: generating.id)
        XCTAssertEqual(
            parkedAfterSwitch.state, "parked",
            "once no longer busy, the model switch must proceed and park the outgoing session"
        )
    }
}
