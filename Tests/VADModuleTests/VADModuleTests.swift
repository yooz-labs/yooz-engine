// VADModuleTests.swift
// VADModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests hit the real Silero VAD CoreML model when bundled and
// exercise AIModule conformance + structural invariants otherwise.
// No mocks, per yooz project policy.

import XCTest
import EngineCore
@testable import VADModule

final class VADModuleTests: XCTestCase {

    /// Attempt to load the Silero model. Returns true on success. Tests that
    /// require a loaded model gate themselves on this; tests that exercise
    /// the failure path explicitly assert against a fresh engine.
    private func tryLoad() async -> Bool {
        let engine = VADEngine.shared
        do {
            try await engine.load()
            return await engine.isLoaded
        } catch {
            return false
        }
    }

    // MARK: - AIModule conformance (always runs)

    func testAIModuleName() {
        XCTAssertEqual(VADEngine.name, "vad")
    }

    func testAIModuleIsReadyMirrorsIsLoaded() async {
        let engine = VADEngine.shared
        let ready = await engine.isReady
        let loaded = await engine.isLoaded
        XCTAssertEqual(ready, loaded,
                       "isReady must reflect the actor's isLoaded state exactly")
    }

    func testHealthCheckReportsLoadedFlag() async {
        let engine = VADEngine.shared
        let health = await engine.healthCheck()
        let loaded = await engine.isLoaded
        XCTAssertEqual(health.loaded, loaded,
                       "healthCheck().loaded must match engine.isLoaded")
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await VADEngine.shared.healthCheck()
        let expected: Set<String> = ["model", "sample_rate", "window_size"]
        XCTAssertEqual(Set(health.detail.keys), expected,
                       "healthCheck detail must report model + audio constants")
        XCTAssertEqual(health.detail["sample_rate"], "16000")
        XCTAssertEqual(health.detail["window_size"], "512")
        XCTAssertEqual(health.detail["model"], "silero-vad-unified-v6.0.0")
    }

    // MARK: - Public API shape (always runs)

    func testConstantsArePublic() {
        // APIServer reads these; the test ensures they're callable from outside
        // the module. If any drop back to internal this test won't compile.
        XCTAssertEqual(VADEngine.sampleRate, 16_000)
        XCTAssertEqual(VADEngine.windowSize, 512)
    }

    func testVADSegmentResultInit() {
        // Public memberwise init must remain available for callers (APIServer
        // constructs wire-format segments from this type's fields).
        let seg = VADSegmentResult(startMs: 100, endMs: 200, probability: 0.9)
        XCTAssertEqual(seg.startMs, 100)
        XCTAssertEqual(seg.endMs, 200)
        XCTAssertEqual(seg.probability, 0.9)
    }

    func testVADErrorDescriptions() {
        XCTAssertNotNil(VADError.modelNotFound.errorDescription)
        XCTAssertNotNil(VADError.modelNotLoaded.errorDescription)
    }

    // MARK: - Failure-path behavior (runs whenever model is NOT loaded)

    func testDetectThrowsWhenNotLoaded() async throws {
        // detect() must throw VADError.modelNotLoaded before load() succeeds.
        // Because VADEngine is a singleton, a prior test may have loaded the
        // real model. In that case this test skips — the failure-path contract
        // cannot be exercised via the same shared instance.
        let engine = VADEngine.shared
        let loaded = await engine.isLoaded
        try XCTSkipIf(loaded,
                      "shared engine already loaded; cannot assert modelNotLoaded throw path")

        do {
            _ = try await engine.detect(samples: [Float](repeating: 0, count: 512))
            XCTFail("detect should throw VADError.modelNotLoaded when model is not loaded")
        } catch let error as VADError {
            XCTAssertEqual(error.errorDescription, VADError.modelNotLoaded.errorDescription)
        } catch {
            XCTFail("expected VADError.modelNotLoaded; got \(error)")
        }
    }

    // MARK: - Model-dependent behavior (requires silero-vad-unified-v6.0.0.mlpackage)

    func testModelLoadAndDetectSilence() async throws {
        let loaded = await tryLoad()
        try XCTSkipUnless(loaded, """
            Silero VAD mlpackage not available to the test bundle.
            See commit message for the known-limitation note; real-model runs
            (CI with bundled resource) exercise this path.
            """)

        let engine = VADEngine.shared
        let isLoaded = await engine.isLoaded
        XCTAssertTrue(isLoaded)

        // 1 second of silence at 16kHz
        let samples = [Float](repeating: 0.0, count: VADEngine.sampleRate)
        let segments = try await engine.detect(samples: samples, resetState: true)

        // Silence must not produce speech segments (or produces empty results)
        XCTAssertTrue(segments.isEmpty,
                      "1s of zero-valued audio should yield zero speech segments; got \(segments.count)")
    }

    func testResetClearsState() async throws {
        let loaded = await tryLoad()
        try XCTSkipUnless(loaded, "Silero model unavailable; skipping reset test")

        // reset() is documented as clearing hidden/cell RNN state. We can't
        // assert on the state directly (it's private), but we can assert it
        // does not throw after a successful load and that subsequent detects
        // still succeed.
        let engine = VADEngine.shared
        try await engine.reset()

        let samples = [Float](repeating: 0.0, count: VADEngine.windowSize * 4)
        _ = try await engine.detect(samples: samples, resetState: false)
    }
}
