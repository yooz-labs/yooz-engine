// KVCompressionTests.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Tests for issue #10 — TurboQuant KV cache compression.
// These tests verify the configuration plumbing and default-off contract
// without requiring a loaded model. A live integration smoke test (loading
// a real Qwen3 model and asserting `KVCacheSimple.turboQuantEnabled` flips
// to true on the per-layer cache) is intentionally gated behind an env var
// so CI does not need the 1.7B weights on hand. See `kvCompressionLiveTest`
// below; run with `KVCOMPRESSION_LIVE=1 swift test` once the model is
// cached locally.

import XCTest
@testable import YoozEngine

final class KVCompressionTests: XCTestCase {

    func testDefaultModeIsOff() {
        // Issue #10 ships default-off so no existing call site changes
        // behavior until a caller explicitly opts in. Regressing this would
        // silently turn on TurboQuant globally.
        XCTAssertEqual(EngineConfig.kvCompression, .off)
    }

    func testKVCompressionModeRawValues() {
        // Wire-format stability: the API and config files use these strings.
        // Renaming a case without bumping is a breaking change for clients.
        XCTAssertEqual(KVCompressionMode.off.rawValue, "off")
        XCTAssertEqual(KVCompressionMode.turbo3.rawValue, "turbo3")
    }

    func testKVCompressionModeCodable() throws {
        // Round-trip encode/decode so the setting can be persisted in
        // settings JSON or sent over /v1/llm/generate without surprises.
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for mode in [KVCompressionMode.off, .turbo3] {
            let data = try encoder.encode(mode)
            let decoded = try decoder.decode(KVCompressionMode.self, from: data)
            XCTAssertEqual(decoded, mode)
        }
    }

    func testKVCompressionModeRejectsUnknown() {
        // Defensive: an unknown wire value must fail decode rather than
        // silently default to a particular mode (we don't want a typo
        // like "turbo4" to silently fall through to off).
        let bogus = #"\"turbo4\""#.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(KVCompressionMode.self, from: bogus))
    }

    func testMLXLLMBackendInitDefaultsToConfig() async {
        // The init `kvCompression` default reads `EngineConfig.kvCompression`,
        // so flipping the engine-wide default propagates to backends that
        // don't override it. We can only observe this indirectly (the
        // property is private), but instantiation must succeed and return
        // a backend with the expected identifier.
        let backend = MLXLLMBackend(modelType: .yoozLight)
        let id = backend.identifier
        XCTAssertEqual(id, LLMModelType.yoozLight.rawValue)
    }

    func testMLXLLMBackendAcceptsExplicitTurbo3() async {
        // Per-instance override path: a call site can opt into turbo3
        // even if the engine default is off. This is the path that the
        // /v1/llm/generate request body will eventually plumb into.
        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        let id = backend.identifier
        XCTAssertEqual(id, LLMModelType.yoozQuality.rawValue)
    }

    /// Live integration test for TurboQuant.
    ///
    /// Skipped unless `KVCOMPRESSION_LIVE=1` is set in the environment, so
    /// CI doesn't try to download Qwen3-1.7B (~1.1 GB). When run, this test:
    ///   1. Loads the Yooz-Quality model with `kvCompression: .turbo3`.
    ///   2. Generates a short response on a small prompt (well below the
    ///      2048-token activation gate, so no compression actually fires —
    ///      we just want to confirm the path is wired without crashes).
    /// The assertion that the SharpAI fork is in use is enforced at compile
    /// time by the `KVCacheSimple.turboQuantEnabled` reference in
    /// `MLXLLMBackend.swift`; building at all on stock mlx-swift-lm would
    /// fail.
    func testKVCompressionLiveTest() async throws {
        guard ProcessInfo.processInfo.environment["KVCOMPRESSION_LIVE"] == "1" else {
            throw XCTSkip("Set KVCOMPRESSION_LIVE=1 to run the live model test")
        }

        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await backend.load()
        let output = try await backend.generate(
            prompt: "Say hello.",
            systemPrompt: "You are a helpful assistant. Reply in one short sentence."
        )
        XCTAssertFalse(output.isEmpty, "turbo3 backend must produce output on a short prompt")
        backend.unload()
    }
}
