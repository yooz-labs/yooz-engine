// KVCompressionTests.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Tests for issue #10 — TurboQuant KV cache compression.
//
// Tier 1 (always run): config plumbing + default-off contract + wire-format
// pinning + per-instance override accessor + factory propagation +
// /v1/llm/generate request-body decode.
//
// Tier 2 (gated, `KVCOMPRESSION_LIVE=1`): live tests that load a real
// Qwen3-1.7B model and exercise the turbo3 codepath end-to-end. CI does
// not run these (no model weights on the runner). Run locally with:
//
//   KVCOMPRESSION_LIVE=1 xcodebuild -project YoozEngine.xcodeproj \
//     -scheme YoozEngine -skipMacroValidation \
//     -destination 'platform=macOS' test
//
// Tier 3 (static fixture): loads `results/phase2_prototype/touchup_quality.json`
// and asserts the byte-equal ratio against a floor. Catches a SharpAI
// rebase that drops Phase 2 quality numbers without needing the model.

import XCTest
import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class KVCompressionTests: XCTestCase {

    // MARK: - Tier 1: config plumbing (always-run)

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

    // MARK: - Tier 1: backend init / accessor

    func testMLXLLMBackendInitDefaultsToConfig() async {
        // The init `kvCompression` default reads `EngineConfig.kvCompression`,
        // and the actor exposes a read-only accessor so we can assert the
        // configured value without reaching into private state.
        let backend = MLXLLMBackend(modelType: .yoozLight)
        let id = await backend.identifier
        let mode = await backend.currentKVCompression
        XCTAssertEqual(id, LLMModelType.yoozLight.rawValue)
        XCTAssertEqual(mode, EngineConfig.kvCompression)
    }

    func testMLXLLMBackendAcceptsExplicitTurbo3() async {
        // Per-instance override path: a call site can opt into turbo3
        // even if the engine default is off. The actor accessor confirms
        // the value flowed all the way to the property — passing this
        // assertion means a future refactor that drops the assignment
        // (e.g., a typo on the constructor argument label) fails the test.
        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        let id = await backend.identifier
        let mode = await backend.currentKVCompression
        XCTAssertEqual(id, LLMModelType.yoozQuality.rawValue)
        XCTAssertEqual(mode, .turbo3)
    }

    func testMLXLLMBackendCounterStartsAtZero() async {
        // Pre-generate, the observability counters must be zero so a test
        // that asserts `> 0` after a generate call cannot be satisfied by
        // stale state.
        let backend = MLXLLMBackend(
            modelType: .yoozLight,
            kvCompression: .turbo3
        )
        let enabled = await backend.lastTurboLayersEnabled
        let total = await backend.lastTurboLayersTotal
        let gate = await backend.lastActivationGatePassed
        XCTAssertEqual(enabled, 0)
        XCTAssertEqual(total, 0)
        XCTAssertFalse(gate)
    }

    // MARK: - Tier 1: factory propagation

    func testFactoryCreateLightPropagatesTurbo3() async {
        // `createLight` is the primary factory used by `TouchUpEngine`.
        // If a future refactor drops the `kvCompression` argument from
        // the factory or hard-codes the default, this test fails.
        let backend = MLXLLMBackend.createLight(kvCompression: .turbo3)
        let mode = await backend.currentKVCompression
        XCTAssertEqual(mode, .turbo3)
    }

    func testFactoryCreateQualityPropagatesTurbo3() async {
        let backend = MLXLLMBackend.createQuality(kvCompression: .turbo3)
        let mode = await backend.currentKVCompression
        XCTAssertEqual(mode, .turbo3)
    }

    func testFactoryCreateForPropagatesTurbo3() async {
        let backend = MLXLLMBackend.create(for: .yoozLight, kvCompression: .turbo3)
        let mode = await backend.currentKVCompression
        XCTAssertEqual(mode, .turbo3)
    }

    func testFactoryNilFallsBackToConfigDefault() async {
        // `kvCompression: nil` (or omitted) must use the engine-wide
        // default. This is the path TouchUp's cached models take.
        let backend = MLXLLMBackend.createLight(kvCompression: nil)
        let mode = await backend.currentKVCompression
        XCTAssertEqual(mode, EngineConfig.kvCompression)
    }

    // MARK: - Tier 1: /v1/llm/generate request-body decode

    func testGenerateRequestDecodesKVCompressionCamelCase() throws {
        let json = #"""
        {
          "prompt": "hi",
          "kvCompression": "turbo3"
        }
        """#.data(using: .utf8)!
        let req = try JSONDecoder().decode(LLMGenerateServerRequest.self, from: json)
        XCTAssertEqual(req.prompt, "hi")
        XCTAssertEqual(req.kvCompression, .turbo3)
    }

    func testGenerateRequestDecodesKVCompressionSnakeCase() throws {
        // Wire-format alias: REST callers may use kv_compression. Both
        // forms must decode identically.
        let json = #"""
        {
          "prompt": "hi",
          "kv_compression": "turbo3"
        }
        """#.data(using: .utf8)!
        let req = try JSONDecoder().decode(LLMGenerateServerRequest.self, from: json)
        XCTAssertEqual(req.kvCompression, .turbo3)
    }

    func testGenerateRequestKVCompressionAbsentIsNil() throws {
        // Field omitted -> nil (engine uses EngineConfig.kvCompression).
        let json = #"""
        { "prompt": "hi" }
        """#.data(using: .utf8)!
        let req = try JSONDecoder().decode(LLMGenerateServerRequest.self, from: json)
        XCTAssertNil(req.kvCompression)
    }

    func testGenerateRequestKVCompressionUnknownFails() {
        // Typo protection: kv_compression: "turbo4" must throw, not
        // silently degrade to off.
        let json = #"""
        {
          "prompt": "hi",
          "kv_compression": "turbo4"
        }
        """#.data(using: .utf8)!
        XCTAssertThrowsError(try JSONDecoder().decode(LLMGenerateServerRequest.self, from: json))
    }

    // MARK: - Tier 3: static-fixture quality regression

    func testTouchUpQualityFixtureMeetsByteEqualFloor() throws {
        // Loads results/phase2_prototype/touchup_quality.json from the
        // repo root and asserts the recorded byte-equal ratio is at
        // least 80%. A SharpAI rebase that introduces a quality drop
        // (centroid table change, hot-window tuning) would update this
        // fixture and fail the floor here. No model weights needed.
        let fixtureURL = Self.repoRootURL()
            .appendingPathComponent("results/phase2_prototype/touchup_quality.json")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("touchup_quality.json fixture missing at \(fixtureURL.path); run from the repo root.")
        }
        let data = try Data(contentsOf: fixtureURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let summary = json?["summary"] as? [String: Any]
        guard let pct = summary?["byte_equal_pct"] as? Double else {
            XCTFail("touchup_quality.json missing summary.byte_equal_pct")
            return
        }
        XCTAssertGreaterThanOrEqual(
            pct, 80.0,
            "TouchUp byte-equal ratio dropped below floor; investigate before merging."
        )
    }

    // MARK: - Tier 2: live integration tests (KVCOMPRESSION_LIVE=1)

    /// Live integration test for TurboQuant — proves the turbo3 path
    /// produces non-empty output without crashing.
    ///
    /// Skipped unless `KVCOMPRESSION_LIVE=1` is set in the environment.
    /// Note that on a short prompt (well below the 2048-token activation
    /// gate) no compression actually fires — `testKVCompressionTurbo3CountsLayersOnGate`
    /// below uses a >2048-token prompt to verify the encode path.
    func testKVCompressionLiveSmoke() async throws {
        try Self.requireLiveEnv()

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
        await backend.unload()
    }

    /// Verifies that the runtime `turboQuantEnabled` cast actually
    /// succeeds on at least one cache layer when `.turbo3` is requested.
    /// This is the test the silent-failure-hunter, code-reviewer, and
    /// pr-test-analyzer all flagged as missing in the original PR — a
    /// SharpAI rebase that renames `KVCacheSimple` would silently
    /// degrade to FP16 without it.
    func testKVCompressionTurbo3FlipsAtLeastOneLayer() async throws {
        try Self.requireLiveEnv()

        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await backend.load()
        _ = try await backend.generate(
            prompt: "Say hello.",
            systemPrompt: "You are a helpful assistant. Reply in one short sentence."
        )
        let enabled = await backend.lastTurboLayersEnabled
        let total = await backend.lastTurboLayersTotal
        XCTAssertGreaterThan(
            total, 0,
            "Generate produced zero cache layers — model load probably failed silently."
        )
        XCTAssertGreaterThan(
            enabled, 0,
            "turbo3 was requested but no cache layer was TurboQuantCapable; this means a SharpAI fork rename or the model uses a non-Simple cache. Check engine logs."
        )
        await backend.unload()
    }

    /// Verifies that the activation gate is observable: a short prompt
    /// must report `lastActivationGatePassed == false`. Triage signal
    /// for "I enabled turbo3 and nothing happened."
    func testKVCompressionShortPromptDoesNotPassGate() async throws {
        try Self.requireLiveEnv()

        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await backend.load()
        _ = try await backend.generate(
            prompt: "Hi.",
            systemPrompt: "Reply in one word."
        )
        let gatePassed = await backend.lastActivationGatePassed
        XCTAssertFalse(
            gatePassed,
            "Short prompt should keep the activation gate closed (FP16 path)."
        )
        await backend.unload()
    }

    /// Verifies that the system-prompt KV cache snapshot/restore round-trip
    /// continues to work under `.turbo3`. PHASE1_FEASIBILITY.md called this
    /// out as the single most regression-prone path. Two back-to-back
    /// generations with the same system prompt; the second must succeed
    /// and produce non-empty output (the cached state is restored into a
    /// fresh `KVCacheSimple` and the generate path must accept it).
    func testKVCompressionSystemPromptKVRestoreUnderTurbo3() async throws {
        try Self.requireLiveEnv()

        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await backend.load()
        let sysPrompt = "You are a helpful assistant. Reply in one short sentence."

        let out1 = try await backend.generate(
            prompt: "Name a primary color.",
            systemPrompt: sysPrompt
        )
        XCTAssertFalse(out1.isEmpty)

        // Second call hits the cached system-prompt KV state path. Under
        // turbo3, the snapshot is taken via `KVCacheSimple.state` (which
        // decodes polarKeys back to fp16) and restored before the new
        // call's tokens are appended. If the round-trip ever breaks, this
        // call fails or produces empty output.
        let out2 = try await backend.generate(
            prompt: "Name another primary color.",
            systemPrompt: sysPrompt
        )
        XCTAssertFalse(out2.isEmpty, "Second call under turbo3 with cached system prompt must produce output")
        await backend.unload()
    }

    /// FP16-vs-turbo3 equality check on a short prompt. Below the 2048
    /// activation gate, both paths must produce identical output at
    /// `temperature: 0` (currently the engine fixes 0.1; we still expect
    /// stable strings on this trivial prompt). If turbo3 produces a
    /// different string on a sub-gate prompt, the gate is leaking.
    func testKVCompressionFP16VsTurbo3SubGateStable() async throws {
        try Self.requireLiveEnv()

        let prompt = "What is 2+2?"
        let sysPrompt = "Answer with the digit only."

        let fp16Backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .off
        )
        try await fp16Backend.load()
        let fp16Out = try await fp16Backend.generate(prompt: prompt, systemPrompt: sysPrompt)
        await fp16Backend.unload()

        let turboBackend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await turboBackend.load()
        let turboOut = try await turboBackend.generate(prompt: prompt, systemPrompt: sysPrompt)
        let turboGatePassed = await turboBackend.lastActivationGatePassed
        await turboBackend.unload()

        XCTAssertFalse(turboGatePassed, "Trivial prompt must stay below the activation gate.")
        XCTAssertFalse(fp16Out.isEmpty)
        XCTAssertFalse(turboOut.isEmpty)
        // Below the gate the two outputs should be identical. If this
        // ever fires it is a real regression — investigate before
        // merging.
        XCTAssertEqual(
            fp16Out, turboOut,
            "Sub-gate FP16 and turbo3 outputs diverged; the activation gate is leaking."
        )
    }

    /// Long-prompt (>2048 tokens) test that activates the encode path.
    /// Without this, the only "live" coverage is the under-gate smoke
    /// test — code-reviewer #6 explicitly called this out.
    func testKVCompressionLongPromptActivatesEncode() async throws {
        try Self.requireLiveEnv()

        // Build a >2048-token prompt by repeating a paragraph. Conservative
        // upper bound: ~2 chars/token for English; 6000 chars = ~3000
        // tokens. The exact count doesn't matter as long as it's clearly
        // above the 2048 gate.
        let para = "The quick brown fox jumps over the lazy dog near the riverbank in the early morning sunshine. "
        let longPrompt = String(repeating: para, count: 70) + "\nSummarize the above in one sentence."

        let backend = MLXLLMBackend(
            modelType: .yoozQuality,
            kvCompression: .turbo3
        )
        try await backend.load()
        let out = try await backend.generate(
            prompt: longPrompt,
            systemPrompt: "Be concise."
        )
        let gatePassed = await backend.lastActivationGatePassed
        let enabled = await backend.lastTurboLayersEnabled
        await backend.unload()

        XCTAssertTrue(gatePassed, "Long prompt should pass the activation gate.")
        XCTAssertGreaterThan(enabled, 0, "Long-prompt turbo3 must enable on >0 layers.")
        XCTAssertFalse(out.isEmpty)
    }

    // MARK: - Helpers

    /// Returns the repo root URL by walking up from the test bundle.
    /// Falls back to the current working directory when run via swift
    /// test from the repo root.
    private static func repoRootURL() -> URL {
        // The host-app TEST_HOST is `Yooz Engine.app`, so Bundle.main is
        // the app bundle inside DerivedData. Walk up to the repo root by
        // looking for `project.yml` (a stable marker).
        var url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<8 {
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("project.yml").path) {
                return url
            }
            url = url.deletingLastPathComponent()
        }
        // Last resort: cwd.
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private static func requireLiveEnv() throws {
        guard ProcessInfo.processInfo.environment["KVCOMPRESSION_LIVE"] == "1" else {
            throw XCTSkip("Set KVCOMPRESSION_LIVE=1 to run live model tests")
        }
    }
}
