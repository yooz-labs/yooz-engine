// EndToEndTests.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// End-to-end checks against a real engine process via `YoozEngineClient`.
// All tests are gated in `IntegrationTestCase.setUp` on `YOOZ_INTEGRATION=1`
// so the default CI run skips cleanly. Heavy-model paths (STT transcription
// with real audio, LLM `/v1/llm/generate`) are deliberately NOT covered
// here — the harness stays with lightweight endpoints (#32 non-goal).

import Foundation
import XCTest
import YoozEngineClient

final class EndToEndTests: IntegrationTestCase {

    // MARK: - /v1/health

    /// Health reports the engine's build variant + module presence. For the
    /// full variant (the only one this harness launches today) we expect
    /// grammar + vad available.
    func testHealthModulesPresentForFullVariant() async throws {
        let health = try await measureEndpoint("GET /v1/health") {
            try await self.client.health()
        }
        XCTAssertEqual(health.status, "ok")
        XCTAssertFalse(health.version.isEmpty)
        XCTAssertTrue(health.modules.grammar,
                      "grammar module should be available on full variant")
        // VAD's `loaded` flag depends on whether CoreML init succeeded at
        // boot. Exercise the flag shape; don't pin true/false here since
        // CoreML readiness is not deterministic on every dev machine.
        _ = health.modules.vad
    }

    // MARK: - /v1/modules

    /// The manifest reflects registered modules and a well-formed
    /// build-variant string. See A3 (#31).
    func testModulesManifestContainsBundledModules() async throws {
        let manifest = try await measureEndpoint("GET /v1/modules") {
            try await self.client.modules()
        }

        XCTAssertFalse(manifest.engineVersion.isEmpty)
        XCTAssertFalse(manifest.buildVariant.isEmpty,
                       "build variant must be a non-empty string")
        XCTAssertFalse(manifest.modules.isEmpty,
                       "manifest must list at least one module")

        let names = manifest.modules.map(\.name)
        XCTAssertTrue(names.contains("grammar"),
                      "expected grammar module in manifest; got: \(names)")

        // Every module reports the unified engine version.
        for module in manifest.modules {
            XCTAssertEqual(
                module.version, manifest.engineVersion,
                "\(module.name) should report unified engine version"
            )
        }
    }

    // MARK: - /v1/grammar/check

    /// Real Rust rule engine: "I are happy" -> at least one correction.
    /// The exact corrected string varies with rule mode (POS vs simple);
    /// we assert on the count, not the output, to keep the test stable.
    func testGrammarCheckAppliesCorrection() async throws {
        let request = GrammarCheckRequest(text: "I are happy")
        let response = try await measureEndpoint("POST /v1/grammar/check") {
            try await self.client.grammar.check(request)
        }
        XCTAssertGreaterThanOrEqual(
            response.correctionsApplied, 1,
            "grammar engine should flag 'I are happy'"
        )
    }

    // MARK: - /v1/touchup (degraded regex path)

    /// `mode: .off` bypasses the LLM entirely and runs the regex-only path
    /// (`TouchUpEngine.processRegexOnly`). Deterministic, does not require
    /// any loaded model — exactly the "degraded" path the task asks for.
    func testTouchUpRegexOnlyPath() async throws {
        let request = TouchUpRequest(text: "hello  world ", mode: .off)
        let response = try await measureEndpoint("POST /v1/touchup") {
            try await self.client.touchUp.process(request)
        }
        XCTAssertEqual(response.mode, .off)
        XCTAssertFalse(response.result.isEmpty,
                       "regex path must echo some processed text")
    }

    // MARK: - /v1/stt/status

    /// Shape check only: the STT engine may or may not be pre-loaded on a
    /// cold start, but the status payload must always be parseable.
    func testSTTStatusShape() async throws {
        let status = try await measureEndpoint("GET /v1/stt/status") {
            try await self.client.stt.status()
        }
        _ = status.loaded
        _ = status.streaming
        if status.loaded {
            XCTAssertNotNil(status.language,
                            "loaded=true implies language is set")
        }
    }

    // MARK: - /v1/vad/detect

    /// 1 second of silence (16000 zero-valued samples at 16kHz) must
    /// produce no speech segments. 16000 > VADEngine.windowSize (512) so
    /// the server will not reject us with `samples_too_short`.
    func testVADSilenceProducesNoSpeech() async throws {
        // Gate: VAD must be loaded, else we'd be asserting on a 503
        // "vad_not_loaded" error instead of the silence semantics.
        let health = try await client.health()
        try XCTSkipUnless(
            health.modules.vad,
            "VAD CoreML model not loaded on this machine; skipping VAD e2e."
        )

        let samples = [Float](repeating: 0.0, count: 16_000)
        let response = try await measureEndpoint("POST /v1/vad/detect") {
            try await self.client.vad.detect(audioSamples: samples, reset: true)
        }
        XCTAssertTrue(
            response.segments.isEmpty,
            "silence should yield no speech segments; got \(response.segments.count)"
        )
    }
}
