// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngineClient

/// Phase 6 — sanity checks for the `PreviewBackendUsageExample`
/// snippet in `Sources/YoozEngineClient/Examples/`.
///
/// The example is a non-throw-away artifact: downstream apps copy
/// from it. This suite makes sure (a) the file actually compiles
/// alongside the SDK, and (b) its public helpers behave as advertised.
final class PreviewBackendUsageTests: XCTestCase {

    // MARK: - Display name + preview labelling

    func testDisplayNameForEachBackend() {
        XCTAssertEqual(
            PreviewBackendUsageExample.displayName(for: .parakeet),
            "Parakeet (Recommended)"
        )
        XCTAssertEqual(
            PreviewBackendUsageExample.displayName(for: .qwen3ASRPreview),
            "Multilingual (Preview)"
        )
    }

    func testIsPreviewIsTrueForQwen3Only() {
        for backend in PreviewBackendUsageExample.STTBackendIdentifier
            .allCases
        {
            let expected = (backend == .qwen3ASRPreview)
            XCTAssertEqual(
                PreviewBackendUsageExample.isPreview(backend), expected
            )
        }
    }

    func testEstimatedDownloadMBOnlyForPreview() {
        XCTAssertEqual(
            PreviewBackendUsageExample.estimatedDownloadMB(for: .qwen3ASRPreview),
            3_500
        )
        XCTAssertNil(
            PreviewBackendUsageExample.estimatedDownloadMB(for: .parakeet)
        )
        XCTAssertNil(
            PreviewBackendUsageExample.estimatedDownloadMB(for: .appleSTT)
        )
    }

    func testPreviewDownloadConfirmationOnlyForPreview() {
        XCTAssertNotNil(
            PreviewBackendUsageExample.previewDownloadConfirmation(
                for: .qwen3ASRPreview
            )
        )
        XCTAssertNil(
            PreviewBackendUsageExample.previewDownloadConfirmation(
                for: .parakeet
            )
        )
    }

    // MARK: - Switch request encoding

    func testSwitchRequestEncodesCanonicalRawValue() throws {
        let request = PreviewBackendUsageExample.STTBackendSwitchRequest(
            backend: .qwen3ASRPreview
        )
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["backend"] as? String, "qwen3_asr_preview")
    }

    // MARK: - Reading metrics off disk

    func testReadRecentMetricsFromMissingFileReturnsEmpty() {
        let missing = URL(
            fileURLWithPath: "/tmp/yooz-does-not-exist-\(UUID().uuidString).jsonl"
        )
        let metrics = PreviewBackendUsageExample.readRecentMetrics(
            from: missing, limit: 10
        )
        XCTAssertEqual(metrics, [])
    }

    func testReadRecentMetricsDecodesJSONLFile() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "yooz-metrics-\(UUID().uuidString)"
            )
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let fileURL = tempDir.appendingPathComponent("stt_metrics.jsonl")

        let isoNow = ISO8601DateFormatter().string(from: Date())
        let line1 = """
        {"backend":"parakeet","model_variant":"parakeet-tdt-v3",\
        "audio_duration_ms":1000,"time_to_first_token_ms":null,\
        "end_to_end_latency_ms":250,"hardware_class":"apple_silicon_m3",\
        "fell_back_from_preview":false,"timestamp_utc":"\(isoNow)"}
        """
        let line2 = """
        {"backend":"parakeet","model_variant":"parakeet-tdt-v3",\
        "audio_duration_ms":2000,"time_to_first_token_ms":null,\
        "end_to_end_latency_ms":500,"hardware_class":"apple_silicon_m3",\
        "fell_back_from_preview":true,"timestamp_utc":"\(isoNow)"}
        """
        try (line1 + "\n" + line2 + "\n").write(
            to: fileURL, atomically: true, encoding: .utf8
        )

        let metrics = PreviewBackendUsageExample.readRecentMetrics(
            from: fileURL, limit: 10
        )
        XCTAssertEqual(metrics.count, 2)
        XCTAssertEqual(metrics[0].audioDurationMs, 1_000)
        XCTAssertFalse(metrics[0].fellBackFromPreview)
        XCTAssertTrue(metrics[1].fellBackFromPreview)
    }

    func testReadRecentMetricsSkipsMalformedLines() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "yooz-metrics-\(UUID().uuidString)"
            )
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let fileURL = tempDir.appendingPathComponent("stt_metrics.jsonl")

        let isoNow = ISO8601DateFormatter().string(from: Date())
        let valid = """
        {"backend":"parakeet","model_variant":"parakeet-tdt-v3",\
        "audio_duration_ms":1000,"time_to_first_token_ms":null,\
        "end_to_end_latency_ms":250,"hardware_class":"apple_silicon_m3",\
        "fell_back_from_preview":false,"timestamp_utc":"\(isoNow)"}
        """
        let payload = "this is not json\n" + valid + "\n{}\n"
        try payload.write(
            to: fileURL, atomically: true, encoding: .utf8
        )

        let metrics = PreviewBackendUsageExample.readRecentMetrics(
            from: fileURL, limit: 10
        )
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].backend, "parakeet")
    }

    // MARK: - Download progress tracker

    func testDownloadProgressTrackerStartsAtZero() {
        let tracker = PreviewBackendUsageExample
            .DownloadProgressTracker()
        XCTAssertEqual(tracker.fraction, 0.0)
        XCTAssertFalse(tracker.done)
    }

    func testDownloadProgressTrackerComputesFraction() {
        var tracker = PreviewBackendUsageExample
            .DownloadProgressTracker()
        tracker.consume(.manifestResolved(totalBytes: 1_000, fileCount: 1))
        tracker.consume(.fileBytes(path: "model.safetensors", completed: 250, total: 1_000))
        XCTAssertEqual(tracker.fraction, 0.25, accuracy: 0.001)
        tracker.consume(.fileBytes(path: "model.safetensors", completed: 750, total: 1_000))
        XCTAssertEqual(tracker.fraction, 0.75, accuracy: 0.001)
        tracker.consume(.done)
        XCTAssertEqual(tracker.fraction, 1.0)
        XCTAssertTrue(tracker.done)
    }

    func testDownloadProgressTrackerClampsAtOne() {
        var tracker = PreviewBackendUsageExample
            .DownloadProgressTracker()
        tracker.consume(.manifestResolved(totalBytes: 100, fileCount: 1))
        tracker.consume(.fileBytes(path: "x", completed: 10_000, total: 100))
        XCTAssertEqual(tracker.fraction, 1.0)
    }
}
