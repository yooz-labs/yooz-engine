// STTModuleTests.swift
// STTModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests exercise real STTModule code: the domain enums, public API
// surface, AIModule conformance, and invariants that survive without a
// loaded model. No mocks, per yooz project policy. Tests that would
// require loading a 600MB+ Parakeet or FastConformer checkpoint are
// skipped with XCTSkipUnless and can be un-skipped by setting
// YOOZ_STT_LOAD_MODELS=1 on a machine that has the weights available;
// mirrors the VAD + LLM gating patterns.

import XCTest
import EngineCore
@testable import STTModule

final class STTModuleTests: XCTestCase {

    /// Gate model-dependent tests behind an env var so CI never tries to
    /// load 600MB+ of STT weights. Mirrors `YOOZ_LLM_LOAD_MODELS` and
    /// the Silero mlpackage gate in VADModuleTests.
    private var shouldLoadRealModels: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_MODELS"] == "1"
    }

    // MARK: - AIModule conformance (always runs)

    func testAIModuleName() {
        XCTAssertEqual(YoozSTTEngine.name, "stt")
    }

    func testAIModuleIsReadyMirrorsIsRunning() async {
        // YoozSTTEngine.isReady is a stored @Published flag set true only
        // after a successful start(). Before any load, both must report
        // the default-false state; after load, both flip together.
        let engine = YoozSTTEngine.shared
        let ready = await engine.isReady
        XCTAssertEqual(ready, engine.isRunning,
                       "isReady must reflect the model-loaded state exactly")
    }

    func testHealthCheckReportsLoadedFlag() async {
        let engine = YoozSTTEngine.shared
        let health = await engine.healthCheck()
        XCTAssertEqual(health.loaded, engine.isRunning,
                       "healthCheck().loaded must match engine.isRunning")
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await YoozSTTEngine.shared.healthCheck()
        let expected: Set<String> = [
            "language",
            "display_name",
            "model_identifier",
            "model_family",
            "streaming"
        ]
        XCTAssertEqual(Set(health.detail.keys), expected,
                       "healthCheck detail must report current-language descriptors")
    }

    func testHealthCheckWhenNotLoaded() async throws {
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot assert not-loaded detail")

        let health = await engine.healthCheck()
        XCTAssertFalse(health.loaded)
        XCTAssertNotNil(health.error, "unloaded engine should surface an error message")
        XCTAssertEqual(health.detail["streaming"], "false")
    }

    // MARK: - STTLanguage (always runs)

    func testAllLanguagesHaveDisplayNames() {
        // APIServer's /v1/stt/languages response reads displayName for every
        // case. An empty string would render a broken UI.
        for lang in STTLanguage.allCases {
            XCTAssertFalse(lang.displayName.isEmpty,
                           "\(lang.rawValue) has empty displayName")
        }
    }

    func testAllLanguagesHaveModelIdentifiers() {
        for lang in STTLanguage.allCases {
            XCTAssertFalse(lang.modelIdentifier.isEmpty,
                           "\(lang.rawValue) has empty modelIdentifier")
        }
    }

    func testFromCodeRoundTrip() {
        // APIServer.parseLanguage uses fromCode to turn wire strings into
        // the domain enum. The case-insensitive upper-case path must also
        // resolve; clients occasionally send "EN" or "FA".
        XCTAssertEqual(STTLanguage.fromCode("en"), .english)
        XCTAssertEqual(STTLanguage.fromCode("EN"), .english,
                       "fromCode should be case-insensitive")
        XCTAssertEqual(STTLanguage.fromCode("fa"), .persian)
        XCTAssertEqual(STTLanguage.fromCode("ar"), .arabic)
        XCTAssertNil(STTLanguage.fromCode("zz-unknown"),
                     "unknown codes must return nil, not a default")
    }

    func testRawValueStabilityForKnownCodes() {
        // Wire protocol: raw values are part of the public /v1/stt/* API.
        // Any rename breaks clients.
        XCTAssertEqual(STTLanguage.english.rawValue, "en")
        XCTAssertEqual(STTLanguage.persian.rawValue, "fa")
        XCTAssertEqual(STTLanguage.arabic.rawValue, "ar")
    }

    func testImplementedFlagsMatchA1Contract() {
        // Per the module design doc + STTLanguage.isImplemented, today the
        // implemented set is: every Parakeet TDT language (English/European)
        // plus Arabic + Persian via FastConformer. Hebrew + CJK are not
        // implemented yet.
        XCTAssertTrue(STTLanguage.english.isImplemented)
        XCTAssertTrue(STTLanguage.arabic.isImplemented)
        XCTAssertTrue(STTLanguage.persian.isImplemented)
        XCTAssertFalse(STTLanguage.hebrew.isImplemented,
                       "Hebrew currently unimplemented; flipping this flag requires model weights")
        XCTAssertFalse(STTLanguage.chinese.isImplemented)
        XCTAssertFalse(STTLanguage.japanese.isImplemented)
        XCTAssertFalse(STTLanguage.korean.isImplemented)
    }

    func testImplementedLanguagesSetMatchesAllCasesFilter() {
        // STTLanguage.implemented is surface-level UI convenience; ensure
        // it actually equals allCases.filter(\.isImplemented).
        let computed = STTLanguage.allCases.filter(\.isImplemented)
        XCTAssertEqual(Set(STTLanguage.implemented), Set(computed))
    }

    func testModelFamilyRouting() {
        XCTAssertEqual(STTLanguage.english.modelFamily, .parakeetTDT)
        XCTAssertEqual(STTLanguage.persian.modelFamily, .fastConformer)
        XCTAssertEqual(STTLanguage.arabic.modelFamily, .fastConformer)
        XCTAssertEqual(STTLanguage.chinese.modelFamily, .cjk)
    }

    // MARK: - ParakeetResult (always runs)

    func testParakeetResultEmptyDefault() {
        XCTAssertEqual(ParakeetResult.empty.text, "")
        XCTAssertEqual(ParakeetResult.empty.finalized, "")
        XCTAssertEqual(ParakeetResult.empty.draft, "")
        XCTAssertTrue(ParakeetResult.empty.isEmpty)
    }

    func testParakeetResultPublicInit() {
        // APIServer's WebSocket handler constructs WSSTTResult from a
        // ParakeetResult's public fields; the type must stay publicly
        // constructible so tests and future consumers can shape one.
        let result = ParakeetResult(text: "hello world", finalized: "hello", draft: "world")
        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.finalized, "hello")
        XCTAssertEqual(result.draft, "world")
        XCTAssertFalse(result.isEmpty)
    }

    func testParakeetResultIsEmptyIgnoresWhitespace() {
        // The streaming + batch paths both trim whitespace before publishing
        // `text`; make sure isEmpty treats whitespace-only as empty.
        let result = ParakeetResult(text: "   ", finalized: "", draft: "")
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - YoozSTTError (always runs)

    func testYoozSTTErrorDescriptions() {
        let errors: [YoozSTTError] = [
            .modelNotFound("/tmp/missing"),
            .modelLoadFailed("bad weights"),
            .notReady,
            .streamError("underrun"),
            .languageNotSupported(.hebrew)
        ]
        for err in errors {
            XCTAssertNotNil(err.errorDescription, "missing description for \(err)")
            XCTAssertFalse(err.errorDescription!.isEmpty, "empty description for \(err)")
        }
    }

    // MARK: - YoozSTTEngine public API shape (always runs)

    func testSharedSingletonExists() {
        // The .shared accessor must be public; if access regresses to
        // internal this test won't compile.
        let engine = YoozSTTEngine.shared
        _ = engine.isRunning
    }

    func testDefaultStateBeforeLoad() throws {
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot assert default state")

        XCTAssertFalse(engine.isRunning)
        XCTAssertFalse(engine.isReady)
        XCTAssertFalse(engine.isStreaming)
        XCTAssertEqual(engine.currentLanguage, .english,
                       "default currentLanguage should be .english")
    }

    func testCreateBatchTranscriberReturnsNilBeforeLoad() throws {
        // APIServer's WebSocket config branch calls createBatchTranscriber
        // after a successful start(); before load it must return nil, not
        // trap or throw. This locks that graceful-degrade contract.
        let engine = YoozSTTEngine.shared
        try XCTSkipIf(engine.isRunning,
                      "shared engine already loaded; cannot test nil-before-load path")

        XCTAssertNil(engine.createBatchTranscriber(mode: .normal))
    }

    func testAvailableLanguagesMatchesImplemented() {
        XCTAssertEqual(
            Set(YoozSTTEngine.shared.availableLanguages),
            Set(STTLanguage.implemented),
            "availableLanguages should mirror STTLanguage.implemented"
        )
    }

    // MARK: - AudioMode (always runs)

    func testAudioModeRawValues() {
        // Raw values are on the wire via /v1/stt/batch `mode` and the
        // WebSocket config message; changing them is a breaking change.
        XCTAssertEqual(AudioMode.normal.rawValue, "normal")
        XCTAssertNotNil(AudioMode(rawValue: "normal"))
    }

    // MARK: - Model-dependent tests (gated by YOOZ_STT_LOAD_MODELS)

    func testStartLoadsEnglishModel() async throws {
        try XCTSkipUnless(shouldLoadRealModels,
                          "Set YOOZ_STT_LOAD_MODELS=1 to exercise the 600MB+ Parakeet load path")

        let engine = YoozSTTEngine.shared
        try await engine.start(language: .english)
        XCTAssertTrue(engine.isRunning)
        XCTAssertEqual(engine.currentLanguage, .english)

        let health = await engine.healthCheck()
        XCTAssertTrue(health.loaded)
        XCTAssertEqual(health.detail["language"], "en")
    }
}
