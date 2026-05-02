// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 5 — config-flag wiring for `EngineConfig.sttBackend`.
///
/// The flag is env-var driven (`YOOZ_STT_BACKEND`). Setting the env
/// var before the test reads `EngineConfig.sttBackend` must produce
/// the matching `STTBackendID` value; unknown values must fall back
/// to the default (`.parakeet`).
///
/// `EngineConfig.sttBackend` is a computed property (NOT cached
/// `static let`) so per-test env-var changes are observable.
final class STTBackendConfigTests: XCTestCase {

    private let envKey = "YOOZ_STT_BACKEND"

    override func setUp() {
        super.setUp()
        unsetenv(envKey)
    }

    override func tearDown() {
        unsetenv(envKey)
        super.tearDown()
    }

    func testDefaultBackendIsParakeet() {
        XCTAssertEqual(EngineConfig.sttBackend, .parakeet)
    }

    func testQwen3PreviewSelectionViaEnv() {
        setenv(envKey, "qwen3_asr_preview", 1)
        XCTAssertEqual(EngineConfig.sttBackend, .qwen3ASRPreview)
    }

    func testFastConformerSelectionViaEnv() {
        setenv(envKey, "fast_conformer", 1)
        XCTAssertEqual(EngineConfig.sttBackend, .fastConformer)
    }

    func testAppleSTTSelectionViaEnv() {
        setenv(envKey, "apple_stt", 1)
        XCTAssertEqual(EngineConfig.sttBackend, .appleSTT)
    }

    func testUnknownValueFallsBackToParakeet() {
        setenv(envKey, "no-such-backend", 1)
        XCTAssertEqual(EngineConfig.sttBackend, .parakeet)
    }

    func testAllBackendIDsRoundTripThroughRawValue() {
        for backend in STTBackendID.allCases {
            let restored = STTBackendID(rawValue: backend.rawValue)
            XCTAssertEqual(restored, backend)
        }
    }

    func testQwen3SupportsBatchButNotStreaming() {
        XCTAssertTrue(STTBackendID.qwen3ASRPreview.supportsBatch)
        XCTAssertFalse(STTBackendID.qwen3ASRPreview.supportsStreaming)
    }
}
