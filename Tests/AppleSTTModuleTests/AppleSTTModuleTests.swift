// AppleSTTModuleTests.swift
// AppleSTTModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests exercise real AppleSTTModule code: AIModule conformance,
// enum surface, default-state invariants, and the OS authorization probe
// (safely). Any test that would require active speech-recognition
// authorization is gated behind YOOZ_STT_LOAD_APPLE=1 to match the
// yooz-engine "no mocks" testing policy — a test that runs without
// authorization would otherwise have to fake a response.

import XCTest
import EngineCore
@testable import AppleSTTModule

final class AppleSTTModuleTests: XCTestCase {

    /// Gate tests that require speech-recognition authorization / real OS
    /// prompts. Mirrors `YOOZ_STT_LOAD_MODELS` (STTModule) and
    /// `YOOZ_LLM_LOAD_MODELS` (LLMModule).
    private var shouldRunAuthedTests: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_APPLE"] == "1"
    }

    // MARK: - AIModule conformance (always runs)

    func testAIModuleName() {
        XCTAssertEqual(AppleSTTEngine.name, "apple_stt")
    }

    func testAIModuleIsReadyDefaultState() async throws {
        // Fresh, unloaded singleton must report not-ready. If a previous test
        // in this process already loaded the engine, skip the assertion since
        // the singleton can't be "un-loaded" without tearing down the shared
        // instance.
        let engine = AppleSTTEngine.shared
        let loaded = await engine.isLoaded
        try XCTSkipIf(loaded,
                      "shared AppleSTTEngine already loaded; cannot assert default state")
        let ready = await engine.isReady
        XCTAssertFalse(ready, "unloaded engine must not be ready")
    }

    func testHealthCheckReportsLoadedMatchesAuthorization() async {
        // healthCheck.loaded must equal (isLoaded && authorization == authorized).
        // With no start() call and no auth prompt, loaded is false regardless.
        let engine = AppleSTTEngine.shared
        let health = await engine.healthCheck()
        let isLoaded = await engine.isLoaded
        let auth = AppleSTTEngine.authorizationStatus
        XCTAssertEqual(
            health.loaded,
            isLoaded && auth == .authorized,
            "healthCheck.loaded must reflect isLoaded AND authorized status"
        )
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await AppleSTTEngine.shared.healthCheck()
        let expected: Set<String> = [
            "os_version",
            "backend_kind",
            "authorization",
            "language",
            "locale",
            "has_built_in_vad",
            "streaming"
        ]
        XCTAssertEqual(Set(health.detail.keys), expected,
                       "healthCheck detail must expose every documented key")
        XCTAssertEqual(health.detail["has_built_in_vad"], "true",
                       "Apple STT always has built-in VAD")
        XCTAssertFalse(health.detail["os_version"]?.isEmpty ?? true)
    }

    // MARK: - STTEngineType / AppleSTTBackendKind enum coverage

    func testBackendKindRawValues() {
        // Raw values appear in /v1/modules detail; breaking them is a wire
        // change. Keep the test dumb so it fails loudly on rename.
        XCTAssertEqual(AppleSTTBackendKind.sfSpeechRecognizer.rawValue, "sf_speech_recognizer")
        XCTAssertEqual(AppleSTTBackendKind.speechAnalyzer.rawValue, "speech_analyzer")
    }

    func testAuthorizationStatusRawValues() {
        XCTAssertEqual(AppleSTTAuthorizationStatus.authorized.rawValue, "authorized")
        XCTAssertEqual(AppleSTTAuthorizationStatus.denied.rawValue, "denied")
        XCTAssertEqual(AppleSTTAuthorizationStatus.restricted.rawValue, "restricted")
        XCTAssertEqual(AppleSTTAuthorizationStatus.notDetermined.rawValue, "not_determined")
    }

    func testAppleSTTLanguageBCP47Mapping() {
        XCTAssertEqual(AppleSTTLanguage.english.bcp47, "en-US")
        XCTAssertEqual(AppleSTTLanguage.persian.bcp47, "fa-IR")
        XCTAssertEqual(AppleSTTLanguage.arabic.bcp47, "ar-SA")
        XCTAssertEqual(AppleSTTLanguage.japanese.bcp47, "ja-JP")
    }

    func testAppleSTTLanguageFromCodeRoundTrip() {
        XCTAssertEqual(AppleSTTLanguage.from(rawCode: "en"), .english)
        XCTAssertEqual(AppleSTTLanguage.from(rawCode: "EN"), .english,
                       "from(rawCode:) should be case-insensitive")
        XCTAssertEqual(AppleSTTLanguage.from(rawCode: "fa"), .persian)
        XCTAssertNil(AppleSTTLanguage.from(rawCode: "zz-unknown"))
    }

    func testAllLanguagesHaveNonEmptyBCP47() {
        for lang in AppleSTTLanguage.allCases {
            XCTAssertFalse(lang.bcp47.isEmpty, "\(lang.rawValue) bcp47 is empty")
            XCTAssertTrue(lang.bcp47.contains("-"),
                          "\(lang.rawValue) bcp47 should be region-qualified")
        }
    }

    // MARK: - Backend availability probe (always runs; read-only)

    func testBackendAvailabilityProbeDoesNotThrow() {
        // Reading the authorization status and probing `isAvailable` must be
        // safe without an auth prompt. If the shared engine isn't authorized,
        // isAvailable returns false; we only assert it doesn't trap.
        _ = AppleSTTBackend.authorizationStatus
        _ = AppleSTTBackend.isAvailable(localeIdentifier: "en-US")
    }

    func testHasBuiltInVADConstant() {
        XCTAssertTrue(AppleSTTEngine.hasBuiltInVAD,
                      "Apple STT always performs its own endpointing")
    }

    // MARK: - Error descriptions (always runs)

    func testAppleSTTErrorDescriptions() {
        let errors: [AppleSTTError] = [
            .recognizerUnavailable(locale: "en-US"),
            .authorizationDenied(status: .denied),
            .recognitionFailed("boom"),
            .pcmBufferCreationFailed,
            .cancelled
        ]
        for error in errors {
            XCTAssertNotNil(error.errorDescription, "missing description for \(error)")
            XCTAssertFalse(error.errorDescription!.isEmpty, "empty description for \(error)")
        }
    }

    // MARK: - Public API shape (always runs)

    func testSharedSingletonExists() {
        // Access regression guard: if `.shared` becomes internal, this won't
        // compile.
        _ = AppleSTTEngine.shared
    }

    func testBackendKindDefault() {
        // Constructing a backend should resolve a kind based on the running
        // OS version — not fail.
        let backend = AppleSTTBackend(localeIdentifier: "en-US")
        // macOS 14-25 uses SFSpeechRecognizer, 26+ uses SpeechAnalyzer.
        XCTAssertTrue(
            backend.kind == .sfSpeechRecognizer || backend.kind == .speechAnalyzer,
            "Backend kind must be one of the two known paths"
        )
    }

    // MARK: - Auth-dependent tests (gated by YOOZ_STT_LOAD_APPLE)

    func testStartThrowsWithoutAuthorization() async throws {
        try XCTSkipUnless(
            shouldRunAuthedTests,
            "Set YOOZ_STT_LOAD_APPLE=1 to run tests that exercise SFSpeechRecognizer authorization"
        )

        // If auth is denied or not determined, start() must throw. If the
        // test host has authorization already, skip — no negative-path
        // assertion is possible without revoking.
        let status = AppleSTTEngine.authorizationStatus
        try XCTSkipIf(status == .authorized,
                      "speech recognition already authorized on this host")

        do {
            try await AppleSTTEngine.shared.start(language: .english)
            XCTFail("expected start() to throw authorizationDenied")
        } catch let error as AppleSTTError {
            if case .authorizationDenied = error {
                // expected
            } else {
                XCTFail("expected authorizationDenied; got \(error)")
            }
        }
    }

    func testBatchTranscribeBeforeStart() async throws {
        // Must throw recognitionFailed with a helpful message before start().
        // Runs in default state only; skip if the singleton was already
        // started in a prior test.
        let engine = AppleSTTEngine.shared
        let loaded = await engine.isLoaded
        try XCTSkipIf(loaded, "shared engine already started; cannot assert not-started throw")

        do {
            _ = try await engine.batchTranscribe(samples: [Float](repeating: 0, count: 1600))
            XCTFail("expected batchTranscribe to throw before start()")
        } catch let error as AppleSTTError {
            if case .recognitionFailed = error {
                // expected
            } else {
                XCTFail("expected recognitionFailed; got \(error)")
            }
        } catch {
            XCTFail("expected AppleSTTError; got \(error)")
        }
    }

    // MARK: - Aligned transcription (engine#34)

    func testAppleAlignedTokenShape() {
        // Same shape as SDK AlignedToken (text + start + end, seconds).
        let token = AppleAlignedToken(text: "hello", start: 0.1, end: 0.5)
        XCTAssertEqual(token.text, "hello")
        XCTAssertEqual(token.start, 0.1, accuracy: 0.001)
        XCTAssertEqual(token.end, 0.5, accuracy: 0.001)
    }

    func testAppleAlignedTranscriptionEmptyState() {
        // Empty state is what the backend returns on the "no speech detected"
        // error path — aligned callers must be able to handle it uniformly
        // with a successful empty recognition.
        let empty = AppleAlignedTranscription.empty
        XCTAssertEqual(empty.transcription, "")
        XCTAssertTrue(empty.tokens.isEmpty)
    }

    func testBatchTranscribeAlignedBeforeStart() async throws {
        // Mirrors the text-only batchTranscribe contract: aligned path must
        // also throw `recognitionFailed` before start().
        let engine = AppleSTTEngine.shared
        let loaded = await engine.isLoaded
        try XCTSkipIf(loaded, "shared engine already started; cannot assert not-started throw")

        do {
            _ = try await engine.batchTranscribeAligned(samples: [Float](repeating: 0, count: 1600))
            XCTFail("expected batchTranscribeAligned to throw before start()")
        } catch let error as AppleSTTError {
            if case .recognitionFailed = error {
                // expected
            } else {
                XCTFail("expected recognitionFailed; got \(error)")
            }
        } catch {
            XCTFail("expected AppleSTTError; got \(error)")
        }
    }

    func testBatchTranscribeAlignedReturnsMonotonicTokens() async throws {
        try XCTSkipUnless(
            shouldRunAuthedTests,
            "Set YOOZ_STT_LOAD_APPLE=1 to exercise the Apple STT alignment path"
        )
        let status = AppleSTTEngine.authorizationStatus
        try XCTSkipUnless(status == .authorized,
                          "Speech recognition not authorized; cannot exercise alignment")

        let engine = AppleSTTEngine.shared
        try await engine.start(language: .english)

        // Two seconds of silence: recognizer returns an empty aligned
        // transcription (no speech detected) — the invariant we care about
        // is that the method returns without throwing, and produces
        // monotonically non-decreasing token starts when any are present.
        let sampleRate = 16_000
        let samples = [Float](repeating: 0, count: sampleRate * 2)
        let aligned = try await engine.batchTranscribeAligned(samples: samples)

        let starts = aligned.tokens.map(\.start)
        XCTAssertEqual(
            starts, starts.sorted(),
            "AppleAlignedToken.start must be monotonically non-decreasing"
        )
        for token in aligned.tokens {
            XCTAssertLessThanOrEqual(token.start, token.end,
                                     "token '\(token.text)' has end before start")
        }
    }
}
