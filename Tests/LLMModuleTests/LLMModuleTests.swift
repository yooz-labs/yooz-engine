// LLMModuleTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests exercise real LLMModule code: the domain enums, public API
// surface, AIModule conformance, and the TouchUpProcessor regex-only
// fast path. No mocks, per yooz project policy. Tests that would require
// a real 276MB or 1GB MLX model load are skipped with XCTSkipUnless and
// can be un-skipped by setting YOOZ_LLM_LOAD_MODELS=1 on a machine that
// has the model weights available; see commit message for details.

import XCTest
import EngineCore
@testable import LLMModule

final class LLMModuleTests: XCTestCase {

    /// Gate model-dependent tests behind an env var so CI never tries to
    /// download 1GB of weights. Mirrors the Silero mlpackage gating in
    /// VADModuleTests.
    private var shouldLoadRealModels: Bool {
        ProcessInfo.processInfo.environment["YOOZ_LLM_LOAD_MODELS"] == "1"
    }

    // MARK: - AIModule conformance (always runs)

    func testAIModuleName() {
        XCTAssertEqual(TouchUpEngine.name, "llm")
    }

    func testAIModuleIsReadyMirrorsIsPreloaded() async {
        let engine = TouchUpEngine.shared
        let ready = await engine.isReady
        let preloaded = await engine.isPreloaded
        XCTAssertEqual(ready, preloaded,
                       "isReady must reflect the actor's isPreloaded state exactly")
    }

    func testHealthCheckReportsLoadedFlag() async {
        let engine = TouchUpEngine.shared
        let health = await engine.healthCheck()
        let preloaded = await engine.isPreloaded
        XCTAssertEqual(health.loaded, preloaded,
                       "healthCheck().loaded must match engine.isPreloaded")
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await TouchUpEngine.shared.healthCheck()
        let expected: Set<String> = [
            "light_model",
            "light_loaded",
            "quality_model",
            "quality_loaded",
            "foundation_models_loaded"
        ]
        XCTAssertEqual(Set(health.detail.keys), expected,
                       "healthCheck detail must report model names + load flags")
        XCTAssertEqual(health.detail["light_model"], LLMModelType.yoozLight.rawValue)
        XCTAssertEqual(health.detail["quality_model"], LLMModelType.yoozQuality.rawValue)
    }

    func testHealthCheckWhenNotPreloaded() async throws {
        // If a prior test preloaded the real model, skip; shared singleton.
        let engine = TouchUpEngine.shared
        let preloaded = await engine.isPreloaded
        try XCTSkipIf(preloaded,
                      "shared engine already preloaded; cannot assert not-loaded detail")

        let health = await engine.healthCheck()
        XCTAssertFalse(health.loaded)
        XCTAssertNotNil(health.error, "unloaded engine should surface an error message")
        XCTAssertEqual(health.detail["light_loaded"], "false")
        XCTAssertEqual(health.detail["quality_loaded"], "false")
    }

    // MARK: - LLMModelType (always runs)

    func testLLMModelTypeRawValues() {
        XCTAssertEqual(LLMModelType.yoozLight.rawValue, "yooz-light-v2")
        XCTAssertEqual(LLMModelType.yoozQuality.rawValue, "yooz-quality-v2")
    }

    func testLLMModelTypeAllCases() {
        // APIServer enumerates allCases to report available models to /v1/models.
        // Any shrink of the list is a public-API break; lock it with equality.
        XCTAssertEqual(
            LLMModelType.allCases,
            [.yoozLight, .yoozQuality],
            "LLMModelType.allCases drives /v1/models output; order matters"
        )
    }

    func testLLMModelTypeInitFromRawValue() {
        XCTAssertEqual(LLMModelType(rawValue: "yooz-light-v2"), .yoozLight)
        XCTAssertEqual(LLMModelType(rawValue: "yooz-quality-v2"), .yoozQuality)
        XCTAssertNil(LLMModelType(rawValue: "bogus-model"),
                     "init(rawValue:) must return nil for unknown model names")
    }

    func testLLMModelTypeEstimatedSizesPositive() {
        // APIServer reports estimatedSize in /v1/models. Must be non-zero so
        // clients can show a meaningful size bar.
        XCTAssertGreaterThan(LLMModelType.yoozLight.estimatedSize, 0)
        XCTAssertGreaterThan(LLMModelType.yoozQuality.estimatedSize, 0)
        // Quality is the bigger model; sanity check ordering.
        XCTAssertGreaterThan(
            LLMModelType.yoozQuality.estimatedSize,
            LLMModelType.yoozLight.estimatedSize
        )
    }

    func testLLMModelTypeHuggingFaceIDs() {
        // Both tiers download from Hugging Face on first use (issue #77
        // removed the embedded / GHCR fallback). Pin the exact v2 LoRA
        // identifiers so a stray rename to an unfinetuned base reads as
        // a CI failure rather than silent quality regression.
        XCTAssertEqual(
            LLMModelType.yoozLight.huggingFaceID,
            "YoozLabs/Yooz-Light-v2-Qwen2.5-0.5B-LoRA"
        )
        XCTAssertEqual(
            LLMModelType.yoozQuality.huggingFaceID,
            "YoozLabs/Yooz-Quality-v2-Qwen3.5-0.8B-LoRA"
        )
    }

    func testLLMModelTypeDisplayStrings() {
        // These power UI labels in Whisper's about panel and the engine menu.
        XCTAssertFalse(LLMModelType.yoozLight.displayName.isEmpty)
        XCTAssertFalse(LLMModelType.yoozLight.description.isEmpty)
        XCTAssertFalse(LLMModelType.yoozQuality.displayName.isEmpty)
        XCTAssertFalse(LLMModelType.yoozQuality.description.isEmpty)
    }

    // MARK: - TouchUpMode (always runs)

    func testTouchUpModeRawValues() {
        // Raw values are stable across the module ↔ wire boundary; APIServer
        // round-trips them through ServerTouchUpMode.rawValue -> asDomain.
        XCTAssertEqual(TouchUpMode.off.rawValue, "off")
        XCTAssertEqual(TouchUpMode.light.rawValue, "light")
        XCTAssertEqual(TouchUpMode.standard.rawValue, "standard")
        XCTAssertEqual(TouchUpMode.full.rawValue, "full")
    }

    // MARK: - LLMModelInfo (always runs)

    func testLLMModelInfoMemberwiseInit() {
        // APIServer constructs /v1/models entries by reading getModelInfo().
        // Public memberwise init must stay callable from outside the module.
        let info = LLMModelInfo(type: .yoozLight, isLoaded: false, isCached: true)
        XCTAssertEqual(info.type, .yoozLight)
        XCTAssertFalse(info.isLoaded)
        XCTAssertTrue(info.isCached)
    }

    // MARK: - LLMError (always runs)

    func testLLMErrorDescriptionsAreNonEmpty() {
        // Error messages feed into /v1/llm/generate 500 responses; the client
        // surfaces them in UI, so every case must have a human-readable string.
        let errors: [LLMError] = [
            .notLoaded,
            .loadFailed("test"),
            .generationFailed("test"),
            .notAvailable("test"),
            .downloadFailed("test"),
            .parsingFailed("test")
        ]
        for err in errors {
            XCTAssertNotNil(err.errorDescription, "missing description for \(err)")
            XCTAssertFalse(err.errorDescription!.isEmpty, "empty description for \(err)")
        }
    }

    // MARK: - TouchUpEngine shared singleton (always runs)

    func testSharedSingletonExists() async {
        // The .shared accessor must be public and return a usable actor. If
        // the access pattern regresses to internal, this test won't compile.
        let engine = TouchUpEngine.shared
        _ = await engine.isPreloaded
    }

    func testDefaultStateBeforePreload() async throws {
        let engine = TouchUpEngine.shared
        let preloaded = await engine.isPreloaded
        try XCTSkipIf(preloaded,
                      "shared engine already preloaded; cannot assert default state")

        let lightLoaded = await engine.isLightModelLoaded
        let qualityLoaded = await engine.isQualityModelLoaded
        let fmLoaded = await engine.isFoundationModelsLoaded
        XCTAssertFalse(lightLoaded, "light model should not be loaded before preload()")
        XCTAssertFalse(qualityLoaded, "quality model should not be loaded before preload()")
        XCTAssertFalse(fmLoaded, "Foundation Models should not be loaded before preload()")
    }

    /// Before any preload, the backend instances don't exist yet, so
    /// `downloadProgress(for:)` returns nil for both tiers. After a
    /// successful preload the model is loaded and the progress would be
    /// 1.0 (covered by the env-gated `testPreloadLoadsLightModel`
    /// inference assertion). This unit test pins the pre-preload nil
    /// contract that `/v1/llm/status` consumes.
    func testDownloadProgressBeforePreloadIsNil() async throws {
        let engine = TouchUpEngine.shared
        let preloaded = await engine.isPreloaded
        try XCTSkipIf(preloaded,
                      "shared engine already preloaded; cannot assert default state")

        let lightProgress = await engine.downloadProgress(for: .yoozLight)
        let qualityProgress = await engine.downloadProgress(for: .yoozQuality)
        XCTAssertNil(lightProgress,
                     "Light backend not instantiated yet -> /v1/llm/status returns nil progress")
        XCTAssertNil(qualityProgress,
                     "Quality backend not instantiated yet -> /v1/llm/status returns nil progress")
    }

    // MARK: - TouchUpEngine.processRegexOnly (always runs)

    func testProcessRegexOnlyAppliesVoiceCommands() {
        // processRegexOnly is `nonisolated` and must work without any model
        // loaded. This exercises the TouchUpProcessor integration end-to-end
        // through the public engine API.
        let engine = TouchUpEngine.shared
        let result = engine.processRegexOnly(text: "hello world period")
        XCTAssertTrue(result.text.contains("."),
                      "voice command 'period' should materialize as punctuation; got \(result.text)")
        XCTAssertEqual(result.modelUsed.rawValue, "regex-only")
        XCTAssertGreaterThanOrEqual(result.latencyMs, 0)
    }

    func testProcessRegexOnlyHandlesEmptyInput() {
        let result = TouchUpEngine.shared.processRegexOnly(text: "")
        XCTAssertEqual(result.text, "")
    }

    // MARK: - TouchUpEngine.process mode-off path (always runs, no model load)

    func testProcessOffModeIsRegexOnlyAndDoesNotLoad() async {
        // Mode .off requests no LLM cleanup: process() returns regex-only and
        // must NOT trigger a model load. (process() now lazy-loads the light
        // model for cleanup modes, so the previous "no model => regex-only for
        // .light" contract no longer holds — .off is the deterministic,
        // load-free graceful path, and the one APIServer relies on for off.)
        let engine = TouchUpEngine.shared
        let loadedBefore = await engine.isLightModelLoaded

        let result = await engine.process(text: "test period", mode: .off)
        XCTAssertEqual(result.modelUsed.rawValue, "regex-only",
                       "mode off must be regex-only; got \(result.modelUsed)")

        let loadedAfter = await engine.isLightModelLoaded
        XCTAssertEqual(loadedBefore, loadedAfter,
                       "mode off must not load the light model")
    }

    // MARK: - FoundationModelsBackend availability (always runs)

    func testFoundationModelsAvailabilityMatchesPlatform() {
        let backend = FoundationModelsBackend()
        let reported = backend.isAvailable()

        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            XCTAssertTrue(reported,
                          "FoundationModels is linked and macOS 26+; isAvailable() should be true")
        } else {
            XCTAssertFalse(reported,
                           "FoundationModels linked but runtime macOS < 26; should report unavailable")
        }
        #else
        XCTAssertFalse(reported,
                       "FoundationModels framework not linked; should always report false")
        #endif
    }

    // MARK: - Model-dependent tests (gated by YOOZ_LLM_LOAD_MODELS env var)

    func testPreloadLoadsLightModel() async throws {
        try XCTSkipUnless(shouldLoadRealModels,
                          "Set YOOZ_LLM_LOAD_MODELS=1 to exercise the 276MB light-model load path")

        let engine = TouchUpEngine.shared
        try await engine.preload(loadQuality: false)
        let lightLoaded = await engine.isLightModelLoaded
        XCTAssertTrue(lightLoaded, "preload() must load the embedded light model")

        let health = await engine.healthCheck()
        XCTAssertTrue(health.loaded)
        XCTAssertEqual(health.detail["light_loaded"], "true")

        let result = await engine.process(text: "hello world", mode: .light)
        XCTAssertNotEqual(result.modelUsed.rawValue, "regex-only",
                          "Light v2 must be used for light mode once loaded; got \(result.modelUsed)")
        XCTAssertFalse(result.text.isEmpty,
                       "Light v2 must produce non-empty output for a valid input")
    }

    /// Regression guard for engine #92: the Yooz-Quality v2 LoRA must load
    /// cleanly through `MLXLLMBackend`. The historical failure was
    /// `Unhandled keys [lora_a, lora_b] in QuantizedLinear` thrown when
    /// mlx-swift-lm's loader auto-applied an `adapters/` directory on top
    /// of already-fused weights. If that adapter-pollution regresses on
    /// HF, this test fails at load time.
    func testPreloadLoadsQualityModelV2() async throws {
        try XCTSkipUnless(shouldLoadRealModels,
                          "Set YOOZ_LLM_LOAD_MODELS=1 to exercise the Quality v2 load path")

        let engine = TouchUpEngine.shared
        try await engine.preload(loadQuality: true)
        let qualityLoaded = await engine.isQualityModelLoaded
        XCTAssertTrue(qualityLoaded, "preload(loadQuality: true) must load Yooz-Quality v2")

        let health = await engine.healthCheck()
        XCTAssertEqual(health.detail["quality_loaded"], "true")

        let result = await engine.process(text: "hello world", mode: .standard)
        XCTAssertNotEqual(result.modelUsed.rawValue, "regex-only",
                          "Quality v2 must be used for standard mode once loaded; got \(result.modelUsed)")
        XCTAssertFalse(result.text.isEmpty,
                       "Quality v2 must produce non-empty output for a valid input")
    }
}
