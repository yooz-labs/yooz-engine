// STTDownloadProgressTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Unit coverage for `YoozSTTEngine.setDownloadProgress(_:)` — the
// external progress setter added in engine#144 so the `/v1/stt/load`
// route can forward `Qwen3ASRModelFetcher.download` progress events
// into the published value that `/v1/stt/status.progress` reads.
//
// End-to-end coverage (route boots a real APIServer, mock fetcher,
// asserts /v1/stt/status surfaces the rolling fraction) requires
// dependency-injection on the fetcher singleton — punted for a
// follow-up. The unit test here pins the clamping contract that the
// route handler depends on.

import Foundation
import XCTest

@testable import STTModule

final class STTDownloadProgressTests: XCTestCase {

    /// External progress setter used by the `/v1/stt/load` route to
    /// forward `Qwen3ASRModelFetcher` events into the published
    /// `downloadProgress`. The value must be clamped to `[0, 1]` so
    /// a malformed or out-of-band event can't push the banner into
    /// an invalid state (e.g. a fraction > 1.0 surfacing as
    /// "Downloading... 137%").
    func testSetDownloadProgressPublishesClampedValue() async {
        let engine = YoozSTTEngine.shared
        await engine.setDownloadProgress(0.42)
        await MainActor.run {
            XCTAssertEqual(engine.downloadProgress, 0.42, accuracy: 1e-9,
                           "In-range fraction must publish verbatim")
        }
        await engine.setDownloadProgress(1.5)
        await MainActor.run {
            XCTAssertEqual(engine.downloadProgress, 1.0, accuracy: 1e-9,
                           "Out-of-range high value must clamp to 1.0")
        }
        await engine.setDownloadProgress(-0.1)
        await MainActor.run {
            XCTAssertEqual(engine.downloadProgress, 0.0, accuracy: 1e-9,
                           "Out-of-range low value must clamp to 0.0")
        }
        // Leave the singleton in a clean state so neighboring tests
        // that read `downloadProgress` aren't perturbed by the local
        // mutations above.
        await engine.setDownloadProgress(0)
    }
}
