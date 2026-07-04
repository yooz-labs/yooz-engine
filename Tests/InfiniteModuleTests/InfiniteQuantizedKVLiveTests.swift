// InfiniteQuantizedKVLiveTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import XCTest
@testable import InfiniteModule

/// Live gate for the quantized-KV session knob (engine#268, epic#263
/// phase 5) — real weights, real decode, gated behind `YOOZ_INFINITE_LIVE=1`
/// (downloads the pinned `Qwen3.5-0.8B-MLX-4bit` on first run, same pin as
/// `InfiniteSessionRuntimeTests`/`InfiniteCheckpointResumeLiveTests`/
/// `InfiniteTurnCommitLiveTests`). Qwen3.5 is the hybrid architecture
/// (`MambaCache` GDN layers interleaved with `KVCacheSimple` full-attention
/// layers, see `Qwen35.swift.newCache`) — exactly the case
/// `openSession(kvBits:kvGroupSize:)` must handle correctly: only the
/// full-attention layers convert to `QuantizedKVCache`, the GDN layers are
/// left untouched.
final class InfiniteQuantizedKVLiveTests: XCTestCase {

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

    /// Must match `descriptor.selection` — see
    /// `InfiniteCheckpointResumeLiveTests`'s class doc for why
    /// `installBackendForTesting`'s label must equal the injected backend's
    /// own `.selection`.
    private static let engineModelId = InfiniteModelSelection.qwen35B1M.rawValue

    private var tempRoot: URL!
    private var store: InfiniteSessionStore!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("InfiniteQuantizedKVLiveTests-\(UUID().uuidString)", isDirectory: true)
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
            "set YOOZ_INFINITE_LIVE=1 to run the live quantized-KV gate (downloads a model)"
        )
        try XCTSkipUnless(
            InfiniteRAMTier.current == .full,
            "this live gate needs the .full (64 GiB) RAM tier; current tier is \(InfiniteRAMTier.current.rawValue)"
        )
    }

    private func longContextText(sentences: Int, seed: String) -> String {
        (0..<sentences)
            .map { "Fact \(seed)-\($0): the capital of country \(seed)-\($0) is city \(seed)-\($0). " }
            .joined()
    }

    // MARK: - (a) Greedy equivalence across checkpoint(park) + resume

    /// A `kvBits: 8` session — append, checkpoint(park), resume, generate
    /// greedy — must produce IDENTICAL token ids to the same quantized
    /// config decoded with no park/resume cycle at all. NOT compared
    /// against an unquantized (f16) reference: quantized-vs-f16 outputs may
    /// legitimately differ (lossy quantization); this gate's equivalence is
    /// strictly within-quantized-config, mirroring
    /// `InfiniteCheckpointResumeLiveTests.testCheckpointResumeGreedyEquivalence`'s
    /// shape for the unquantized case.
    func testQuantizedSessionGreedyEquivalence() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let text = longContextText(sentences: 80, seed: "q")
        let prompt = "Name one of the cities mentioned above in one word."
        let maxTokens = 32

        // (a) the resumed path: create a kvBits=8 session, append,
        // checkpoint(park: true), resume, generate.
        let session = try await engine.createSession(
            request: InfiniteCreateSessionRequest(
                modelId: Self.engineModelId, label: "quantized-resume", kvBits: 8
            )
        )
        _ = try await engine.append(sessionID: session.id, request: InfiniteAppendSessionRequest(text: text))

        let checkpointResponse = try await engine.checkpoint(
            sessionID: session.id, request: InfiniteCheckpointSessionRequest(label: "before-resume", park: true)
        )
        XCTAssertEqual(checkpointResponse.session.state, "parked")

        let resumedInfo = try await engine.resume(
            sessionID: session.id, request: InfiniteResumeSessionRequest(checkpointId: nil)
        )
        XCTAssertEqual(resumedInfo.state, "open")

        // The resumed cache must still be quantized post-round-trip
        // (loadPromptCache reconstructs QuantizedKVCache from the class
        // name baked into cache.safetensors' metadata — no extra wiring
        // needed on the resume path beyond what checkpoint/resume already do).
        let resumedLayers = await backend.liveCacheLayerDTypesForTesting(id: session.id)
        let resumedFullAttention = resumedLayers.filter { !$0.className.contains("MambaCache") }
        XCTAssertFalse(resumedFullAttention.isEmpty, "expected at least one full-attention layer on Qwen3.5")
        for layer in resumedFullAttention {
            XCTAssertEqual(
                layer.className, "QuantizedKVCache",
                "resumed full-attention layer must still be QuantizedKVCache after the safetensors round trip"
            )
        }

        let resumed = try await backend.generateSession(
            id: session.id, prompt: prompt, maxTokens: maxTokens, temperature: 0
        )

        // (b) the same kvBits=8 config, generated with no park/resume cycle.
        let uninterruptedID = "quantized-uninterrupted-\(session.id)"
        await backend.openSession(id: uninterruptedID, kvBits: 8)
        _ = try await backend.appendTokens(id: uninterruptedID, text: text)
        let uninterrupted = try await backend.generateSession(
            id: uninterruptedID, prompt: prompt, maxTokens: maxTokens, temperature: 0
        )

        XCTAssertFalse(resumed.tokenIds.isEmpty, "quantized generation should produce real tokens")
        XCTAssertEqual(
            resumed.tokenIds, uninterrupted.tokenIds,
            "checkpoint(park)+resume on a quantized session must match the same quantized config " +
                "decoded with no park/resume cycle"
        )
    }

    // MARK: - (b) Cache class stability across append/generate/turn-commit

    /// Guards the `maybeQuantizeKVCache` entry-replacement hazard forever
    /// (see `InfiniteSessionRuntime.swift`'s doc comment on
    /// `runSessionDecodeLoop`): after append, a plain (non-turn-framed)
    /// generate, and a turn-commit generate, every full-attention cache
    /// entry must STILL be `QuantizedKVCache` — never silently swapped back
    /// to a `KVCacheSimple` copy inside some iterator's own private array —
    /// and every `MambaCache` (GDN) entry must be untouched (still present,
    /// never converted). Exercises two independent decode-loop callers
    /// against the same durable quantized cache: `generateSession` (raw
    /// continuation, no branching) and `generateTurn(.turnCommit)` (branches
    /// the durable cache — `KVCacheBranching`'s allowlist already includes
    /// `QuantizedKVCache` — decodes on the branch, discards it, then commits
    /// via the same chunked-prefill primitive `append` uses).
    func testQuantizedSessionCacheClassStability() async throws {
        try Self.skipUnlessLive()

        let backend = try await MLXInfiniteBackend.load(Self.descriptor)
        let engine = InfiniteEngine(store: store)
        await engine.installBackendForTesting(backend, as: .qwen35B1M)

        let session = try await engine.createSession(
            request: InfiniteCreateSessionRequest(
                modelId: Self.engineModelId, label: "quantized-stability", kvBits: 8
            )
        )
        _ = try await engine.append(
            sessionID: session.id,
            request: InfiniteAppendSessionRequest(text: longContextText(sentences: 30, seed: "stab"))
        )

        @discardableResult
        func assertStillQuantized(_ label: String) async -> Int {
            let layers = await backend.liveCacheLayerDTypesForTesting(id: session.id)
            let fullAttention = layers.filter { !$0.className.contains("MambaCache") }
            let mamba = layers.filter { $0.className.contains("MambaCache") }
            XCTAssertFalse(fullAttention.isEmpty, "\(label): expected at least one full-attention layer")
            XCTAssertFalse(mamba.isEmpty, "\(label): expected at least one MambaCache (GDN) layer")
            for layer in fullAttention {
                XCTAssertEqual(
                    layer.className, "QuantizedKVCache",
                    "\(label): full-attention layer must stay QuantizedKVCache " +
                        "(guards the maybeQuantizeKVCache entry-replacement hazard)"
                )
            }
            return mamba.count
        }

        let mambaCountAfterAppend = await assertStillQuantized("after append")

        _ = try await backend.generateSession(
            id: session.id, prompt: "Name one fact above in one word.", maxTokens: 16, temperature: 0
        )
        let mambaCountAfterGenerate = await assertStillQuantized("after plain generate")
        XCTAssertEqual(mambaCountAfterGenerate, mambaCountAfterAppend, "MambaCache layer count must not change")

        _ = try await engine.generate(
            sessionID: session.id,
            request: InfiniteGenerateSessionRequest(
                prompt: "Name a different fact above in one word.", maxTokens: 16, temperature: 0
            )
        )
        let mambaCountAfterTurnCommit = await assertStillQuantized("after turn-commit generate")
        XCTAssertEqual(mambaCountAfterTurnCommit, mambaCountAfterAppend, "MambaCache layer count must not change")
    }
}
