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

    /// Safe accessor for the shared client. `IntegrationTestCase.client` is
    /// an implicitly-unwrapped optional; force-unwrapping it inside a test
    /// body turns any setup failure into a runtime crash that kills the
    /// whole xctest host mid-run (#37). `XCTUnwrap` converts a nil client
    /// into a normal test failure instead.
    private func requireClient() throws -> YoozEngineClient {
        try XCTUnwrap(client, "IntegrationTestCase.client was not initialised")
    }

    // MARK: - /v1/health

    /// Health reports the engine's build variant + module presence. For the
    /// full variant (the only one this harness launches today) we expect
    /// grammar + vad available.
    func testHealthModulesPresentForFullVariant() async throws {
        let client = try requireClient()
        let health = try await measureEndpoint("GET /v1/health") {
            try await client.health()
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
        let client = try requireClient()
        let manifest = try await measureEndpoint("GET /v1/modules") {
            try await client.modules()
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
        let client = try requireClient()
        let request = GrammarCheckRequest(text: "I are happy")
        let response = try await measureEndpoint("POST /v1/grammar/check") {
            try await client.grammar.check(request)
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
        let client = try requireClient()
        let request = TouchUpRequest(text: "hello  world ", mode: .off)
        let response = try await measureEndpoint("POST /v1/touchup") {
            try await client.touchUp.process(request)
        }
        XCTAssertEqual(response.mode, .off)
        XCTAssertFalse(response.result.isEmpty,
                       "regex path must echo some processed text")
    }

    // MARK: - /v1/infinite/* (engine-hosted module)

    /// Consumer-style proof for Infinite as an engine-hosted module. This
    /// goes through `YoozEngineClient` against the served app, not module
    /// internals, so it catches SDK/route/schema drift at the boundary that
    /// downstream apps will use.
    func testInfiniteSDKSessionLifecycleThroughEngine() async throws {
        let client = try requireClient()

        let manifest = try await measureEndpoint("GET /v1/modules infinite") {
            try await client.modules()
        }
        let infiniteManifest = try XCTUnwrap(
            manifest.modules.first { $0.name == "infinite" },
            "full engine variant must advertise the Infinite module"
        )
        XCTAssertEqual(infiniteManifest.version, manifest.engineVersion)
        XCTAssertEqual(
            infiniteManifest.detail["cleanup_policy"],
            "explicit_delete_or_process_exit;max_active_sessions=16"
        )

        let models = try await measureEndpoint("GET /v1/infinite/models") {
            try await client.infinite.availableModels()
        }
        XCTAssertFalse(models.models.isEmpty, "Infinite picker must expose models")
        let active = try XCTUnwrap(
            models.models.first { $0.id == models.activeId },
            "activeId must identify one picker row"
        )
        try XCTSkipUnless(
            active.loadState != .unavailable,
            "Active Infinite model is not available on this machine"
        )
        XCTAssertEqual(active.adapterKind, "infinite-paged-kv-mlx-v1")
        XCTAssertNotNil(active.evidenceRef)

        let statusBefore = try await measureEndpoint("GET /v1/infinite/status") {
            try await client.infinite.status()
        }
        XCTAssertEqual(statusBefore.modelId, models.activeId)
        XCTAssertEqual(
            statusBefore.cleanupPolicy,
            "explicit_delete_or_process_exit;max_active_sessions=16"
        )
        XCTAssertNotNil(statusBefore.resources)

        let created = try await measureEndpoint("POST /v1/infinite/sessions") {
            try await client.infinite.createSession(label: "sdk-integration")
        }
        XCTAssertEqual(created.modelId, models.activeId)
        XCTAssertEqual(created.label, "sdk-integration")
        XCTAssertEqual(created.state, "open")

        do {
            let appended = try await measureEndpoint("POST /v1/infinite/sessions/:id/append") {
                try await client.infinite.append(
                    sessionId: created.id,
                    text: "real hosted Infinite context"
                )
            }
            XCTAssertEqual(appended.session.id, created.id)
            XCTAssertGreaterThan(appended.appendedCharacters, 0)
            XCTAssertGreaterThan(appended.estimatedAppendedTokens, 0)

            let fetched = try await measureEndpoint("GET /v1/infinite/sessions/:id") {
                try await client.infinite.session(id: created.id)
            }
            XCTAssertEqual(fetched.inputCharacters, appended.session.inputCharacters)
            XCTAssertEqual(fetched.estimatedInputTokens, appended.session.estimatedInputTokens)

            let checkpoint = try await measureEndpoint("POST /v1/infinite/sessions/:id/checkpoint") {
                try await client.infinite.checkpoint(sessionId: created.id, label: "after-append")
            }
            XCTAssertEqual(checkpoint.session.id, created.id)
            XCTAssertEqual(checkpoint.session.checkpointCount, 1)
            XCTAssertEqual(checkpoint.checkpoint.label, "after-append")

            do {
                _ = try await measureEndpoint("POST /v1/infinite/sessions/:id/generate") {
                    try await client.infinite.generate(
                        sessionId: created.id,
                        prompt: "summarize",
                        maxTokens: 16
                    )
                }
                XCTFail("Infinite generate should return 501 until backend inference is wired")
            } catch YoozEngineError.serverError(let statusCode, let code, _) {
                // Current served contract: session state is durable, inference is not wired yet.
                XCTAssertEqual(statusCode, 501)
                XCTAssertEqual(code, "generation_unavailable")
            }

            let deleted = try await measureEndpoint("DELETE /v1/infinite/sessions/:id") {
                try await client.infinite.deleteSession(id: created.id)
            }
            XCTAssertTrue(deleted.deleted)
            XCTAssertEqual(deleted.sessionId, created.id)

            do {
                _ = try await client.infinite.session(id: created.id)
                XCTFail("deleted Infinite session should not be fetchable")
            } catch YoozEngineError.serverError(let statusCode, let code, _) {
                XCTAssertEqual(statusCode, 404)
                XCTAssertEqual(code, "session_not_found")
            }
        } catch {
            _ = try? await client.infinite.deleteSession(id: created.id)
            throw error
        }
    }

    // MARK: - /v1/stt/status

    /// Shape check only: the STT engine may or may not be pre-loaded on a
    /// cold start, but the status payload must always be parseable.
    func testSTTStatusShape() async throws {
        let client = try requireClient()
        let status = try await measureEndpoint("GET /v1/stt/status") {
            try await client.stt.status()
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
        let client = try requireClient()
        // Gate: VAD must be loaded, else we'd be asserting on a 503
        // "vad_not_loaded" error instead of the silence semantics.
        let health = try await client.health()
        try XCTSkipUnless(
            health.modules.vad,
            "VAD CoreML model not loaded on this machine; skipping VAD e2e."
        )

        let samples = [Float](repeating: 0.0, count: 16_000)
        let response = try await measureEndpoint("POST /v1/vad/detect") {
            try await client.vad.detect(audioSamples: samples, reset: true)
        }
        XCTAssertTrue(
            response.segments.isEmpty,
            "silence should yield no speech segments; got \(response.segments.count)"
        )
    }
}
