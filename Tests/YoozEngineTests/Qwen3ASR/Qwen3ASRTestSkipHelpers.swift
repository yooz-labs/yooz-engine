// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

/// macOS TCC gate shared across the heavy `/Volumes/S1`-backed
/// Qwen3-ASR test suites.
///
/// `XCTSkipUnless(FileManager.default.fileExists(atPath:))` is fine
/// for stat operations, but the heavy tests subsequently call
/// `Data(contentsOf:)` / `MLX.loadArrays(url:)` / swift-transformers'
/// tokenizer loader — all of which hang under macOS TCC when the GUI
/// xctest host can't render the access prompt headlessly. Phase 4 ran
/// these via `swift test` only; the gate honors that with an explicit
/// opt-in (`YOOZ_RUN_TCC_TESTS=1`) for users who've granted Full Disk
/// Access to the test host.
enum Qwen3ASRTestEnvironment {

    /// Throws `XCTSkip` if neither the env var is set nor the test
    /// bundle path looks SwiftPM-shaped.
    static func skipUnlessSafeForTCC(
        file: StaticString = #file,
        line: UInt = #line
    ) throws {
        let envOK = ProcessInfo.processInfo.environment[
            "YOOZ_RUN_TCC_TESTS"
        ] == "1"
        let isSwiftPMHost = Bundle(for: TestSentinel.self)
            .bundleURL.path.contains(".build/")
        try XCTSkipUnless(
            envOK || isSwiftPMHost,
            "Skipping /Volumes/S1-backed Qwen3-ASR test under "
                + "xcodebuild (macOS TCC). Run via `swift test --filter "
                + "Qwen3ASR*` or set YOOZ_RUN_TCC_TESTS=1 after granting "
                + "Full Disk Access to the test host.",
            file: file, line: line
        )
    }

    /// Anchor class so `Bundle(for:)` resolves the test bundle, not
    /// some host framework.
    private final class TestSentinel {}
}
