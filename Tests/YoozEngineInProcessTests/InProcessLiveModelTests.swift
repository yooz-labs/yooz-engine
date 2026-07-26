import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// LIVE in-process tests that exercise the FULL wiring end-to-end through
/// `YoozEngineClient.inProcess()`: module registration -> `InProcessTransport`
/// routing -> real engine actor -> real model load (HuggingFace download on
/// first run) -> real inference -> SDK DTO decode.
///
/// These are **local-only**, gated behind the same env vars as the module-level
/// live tests so a default `swift test` / CI run skips them cleanly:
///
///   - STT paths: `YOOZ_STT_LOAD_MODELS=1`  (downloads ~600 MB Parakeet)
///   - LLM paths: `YOOZ_LLM_LOAD_MODELS=1`
///
/// Audio is a deterministic synthetic sine (not real speech), so we assert
/// load + pipeline INVARIANTS — the model loads, inference runs, the seam maps
/// results into the SDK DTOs — not flaky transcript text.
///
/// IMPORTANT — run context: MLX GPU inference needs the `default.metallib`
/// resource, which is NOT loadable under a plain `swift test` CLI run (it errors
/// "Failed to load the default metallib"). This is a documented MLX-Swift /
/// SwiftPM limitation, the same reason the engine's `STTModuleTests` /
/// `LLMModuleTests` model paths run under **xcodebuild** (whose `.xctest` bundle
/// embeds the metallib via xcodegen) rather than `swift test`. Run these the
/// same way: an app-hosted xctest target with `YOOZ_STT_LOAD_MODELS=1` /
/// `YOOZ_LLM_LOAD_MODELS=1` set. The structural in-process suite
/// (`InProcessTransportTests`, incl. the real grammar-FFI round trip) covers the
/// non-GPU wiring under plain `swift test`.
final class InProcessLiveModelTests: XCTestCase {
    private var sttEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_MODELS"] == "1"
    }

    private var llmEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_LLM_LOAD_MODELS"] == "1"
    }

    private func makeClient() -> YoozEngineClient { YoozEngineClient.inProcess() }

    /// 2 seconds of 16 kHz mono sine — enough to drive the Parakeet encoder.
    private func sineAudio(seconds: Float = 2.0, hz: Float = 440) -> [Float] {
        let sampleRate = 16_000
        let count = Int(Float(sampleRate) * seconds)
        var samples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            samples[i] = 0.1 * sinf(2 * .pi * hz * Float(i) / Float(sampleRate))
        }
        return samples
    }

    // MARK: - STT batch (real model load + inference through the facade)

    func testInProcessBatchTranscribeLoadsAndRunsModel() async throws {
        try XCTSkipUnless(sttEnabled,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the in-process Parakeet load path")
        let client = makeClient()
        try await client.connect()

        // Goes SDK -> InProcessTransport.handleBatch -> YoozSTTEngine.start
        // (loads/downloads) -> batchTranscribe -> SDK TranscriptionResult.
        let result = try await client.stt.transcribe(audioSamples: sineAudio(), language: .english)
        XCTAssertEqual(result.language, "en")
        // Synthetic audio yields no meaningful text; assert the result is a
        // well-formed DTO (text == finalized + draft contract) rather than text.
        XCTAssertNotNil(result.text)

        // The load path actually loaded the model — observable via the
        // in-process status endpoint (proves registration + the status handler too).
        let status = try await client.stt.status()
        XCTAssertTrue(status.loaded, "model should be loaded after a batch transcribe")
    }

    /// Explicit pre-load via `/v1/stt/load` (the route Whisper's `preloadModel`
    /// drives at launch). Proves the in-process transport pre-warms the model
    /// and returns a loaded status without first running a transcribe — i.e. the
    /// endpoint is no longer `unsupportedOperation` in-process.
    func testInProcessLoadModelPrewarmsAndReportsLoaded() async throws {
        try XCTSkipUnless(sttEnabled,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the in-process Parakeet load path")
        let client = makeClient()
        try await client.connect()

        // Blocking pre-load (`/v1/stt/load?wait=true`) — must return a fully
        // loaded status, not throw unsupportedOperation.
        let loaded = try await client.stt.loadModel(language: .english)
        XCTAssertTrue(loaded.loaded, "loadModel must return a loaded status in-process")

        // Status endpoint agrees the model is warm before any transcribe ran.
        let status = try await client.stt.status()
        XCTAssertTrue(status.loaded, "model should remain loaded after explicit pre-load")
    }

    func testInProcessAlignedTranscribeKeepsTokensInWindow() async throws {
        try XCTSkipUnless(sttEnabled,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the in-process aligned path")
        let client = makeClient()
        try await client.connect()

        let duration: Float = 2.0
        let result = try await client.stt.batchTranscribeAligned(
            audioSamples: sineAudio(seconds: duration), language: .english
        )
        let tokens = try XCTUnwrap(result.tokens, "aligned path must return a (possibly empty) tokens array")
        for token in tokens {
            XCTAssertGreaterThanOrEqual(token.start, 0)
            XCTAssertLessThanOrEqual(token.end, duration + 1.0,
                                     "token '\(token.text)' end \(token.end) exceeds the audio window")
            XCTAssertGreaterThanOrEqual(token.end, token.start)
        }
    }

    // MARK: - STT streaming (in-process session drives the real transcriber)

    func testInProcessStreamingProducesAFinalResult() async throws {
        try XCTSkipUnless(sttEnabled,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the in-process streaming path")
        let client = makeClient()
        try await client.connect()

        let stream = try await client.stt.startStream(language: .english)

        // Feed the audio in ~100 ms chunks, draining partials as we go.
        let audio = sineAudio()
        let chunk = 1_600
        var index = 0
        while index < audio.count {
            let end = min(index + chunk, audio.count)
            try await stream.sendAudio(Array(audio[index..<end]))
            _ = try await stream.receive()  // partial (may be empty for sine)
            index = end
        }

        // Closing triggers finalize; drain until the `final` frame (or end).
        stream.close()
        var sawFinal = false
        while let result = try await stream.receive() {
            if result.isFinal { sawFinal = true; break }
        }
        XCTAssertTrue(sawFinal, "streaming must deliver a final result after close()")
    }

    // MARK: - LLM generate (real model load + inference through the facade)

    func testInProcessLLMGenerateProducesText() async throws {
        try XCTSkipUnless(llmEnabled,
                          "Set YOOZ_LLM_LOAD_MODELS=1 to exercise the in-process LLM load path")
        let client = makeClient()
        try await client.connect()

        // SDK -> InProcessTransport.handleLLM -> TouchUpEngine.generate -> SDK DTO.
        let output = try await client.llm.generate(prompt: "Correct: i has a apple")
        XCTAssertFalse(output.isEmpty, "LLM generate should return non-empty text")
    }

    // MARK: - LLM cache clearing (engine#299, real weights)

    /// Clearing a resident tier's cache must leave the weights themselves
    /// resident — the entire point of the endpoint versus `unloadModel`.
    /// SDK -> InProcessTransport.handleLLMClearCache ->
    /// TouchUpEngine.clearCache -> MLXLLMBackend.clearSession.
    func testInProcessClearCacheDropsCacheButKeepsWeightsLoaded() async throws {
        try XCTSkipUnless(llmEnabled,
                          "Set YOOZ_LLM_LOAD_MODELS=1 to exercise the in-process LLM load path")
        let client = makeClient()
        try await client.connect()

        // `/v1/llm/preload` (not the picker's `setModel`) so this doesn't
        // depend on / disturb the single-resident active-tier invariant.
        try await client.touchUp.preloadModel("yooz-light-v3")

        let cleared = try await client.touchUp.clearCache("yooz-light-v3")
        XCTAssertEqual(cleared, ["yooz-light-v3"], "the resident tier's cache must be reported cleared")

        // `/v1/llm/status` reports the ACTIVE tier only; assert through the
        // picker instead so this holds regardless of which tier is preferred.
        let picker: LLMModelsResponse = try await client.touchUp.availableModels()
        let light = picker.available.first { $0.id == "yooz-light-v3" }
        XCTAssertEqual(light?.loaded, true, "weights must remain resident after a cache clear")
    }

    /// Omitting `model` with two resident tiers must clear both — the
    /// "idle-policy sweep" shape a consumer actually calls.
    func testInProcessClearCacheOmittedModelClearsEveryLoadedTier() async throws {
        try XCTSkipUnless(llmEnabled,
                          "Set YOOZ_LLM_LOAD_MODELS=1 to exercise the in-process LLM load path")
        let client = makeClient()
        try await client.connect()

        // `/v1/llm/preload` for both tiers loads them side by side (no
        // single-resident eviction — that only applies to the picker's
        // `setModel`), so both are genuinely resident at once.
        try await client.touchUp.preloadModel("yooz-light-v3")
        try await client.touchUp.preloadModel("yooz-quality-v3")

        let cleared = try await client.touchUp.clearCache()
        XCTAssertEqual(
            Set(cleared), ["yooz-light-v3", "yooz-quality-v3"],
            "omitting model must clear every resident tier"
        )

        let picker: LLMModelsResponse = try await client.touchUp.availableModels()
        let light = picker.available.first { $0.id == "yooz-light-v3" }
        let quality = picker.available.first { $0.id == "yooz-quality-v3" }
        XCTAssertEqual(light?.loaded, true, "Light weights must remain resident after clear")
        XCTAssertEqual(quality?.loaded, true, "Quality weights must remain resident after clear")
    }
}
