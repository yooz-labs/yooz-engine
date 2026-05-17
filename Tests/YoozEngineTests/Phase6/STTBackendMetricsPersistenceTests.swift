// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
@testable import YoozEngine

/// Phase 6 — opt-in / opt-out behavior for the JSONL metrics sink.
///
/// The sink is the only place metrics records are persisted. The
/// public surface is `STTMetricsSink.record(_:)`. Tests assert:
///
/// - `DiscardingMetricsSink` (telemetry off) writes nothing.
/// - `JSONLMetricsSink` (telemetry on) appends one line per record.
/// - The factory `makeSTTMetricsSink(optedIn:fileURL:)` returns the
///   right kind of sink for each opt-in flag.
final class STTBackendMetricsPersistenceTests: XCTestCase {

    private var tempDir: URL!
    private var fileURL: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yooz-stt-metrics-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
        fileURL = tempDir.appendingPathComponent("stt_metrics.jsonl")
    }

    override func tearDownWithError() throws {
        if let tempDir, FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try super.tearDownWithError()
    }

    private func makeMetric(
        backend: STTBackendID = .parakeet,
        fellBack: Bool = false
    ) -> STTBackendMetrics {
        STTBackendMetrics(
            backend: backend,
            modelVariant: STTBackendMetrics.defaultModelVariant(
                for: backend
            ),
            audioDurationMs: 1_000,
            timeToFirstTokenMs: nil,
            endToEndLatencyMs: 250,
            hardwareClass: .appleSiliconM3,
            fellBackFromPreview: fellBack,
            timestampUTC: Date()
        )
    }

    // MARK: - Discarding sink: zero writes

    func testDiscardingSinkWritesNothing() async {
        let sink = DiscardingMetricsSink()
        await sink.record(makeMetric())
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: fileURL.path),
            "DiscardingMetricsSink must not create any file."
        )
    }

    // MARK: - JSONL sink: one record per line

    func testJSONLSinkAppendsLine() async throws {
        let sink = JSONLMetricsSink(fileURL: fileURL)
        await sink.record(makeMetric())

        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(lines.count, 1)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let lineData = lines[0].data(using: .utf8) else {
            return XCTFail("Could not get line data")
        }
        let decoded = try decoder.decode(
            STTBackendMetrics.self, from: lineData
        )
        XCTAssertEqual(decoded.backend, .parakeet)
        XCTAssertFalse(decoded.fellBackFromPreview)
    }

    func testJSONLSinkAppendsMultipleLines() async throws {
        let sink = JSONLMetricsSink(fileURL: fileURL)
        await sink.record(makeMetric(backend: .parakeet))
        await sink.record(makeMetric(backend: .qwen3ASRPreview, fellBack: true))
        await sink.record(makeMetric(backend: .appleSTT))

        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(lines.count, 3)
    }

    func testJSONLSinkCreatesParentDirectoryIfMissing() async throws {
        let nestedURL = tempDir
            .appendingPathComponent("a/b/c/stt_metrics.jsonl")
        let sink = JSONLMetricsSink(fileURL: nestedURL)
        await sink.record(makeMetric())
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: nestedURL.path),
            "JSONLMetricsSink should create missing parent dirs."
        )
    }

    // MARK: - Factory wires the right sink

    func testFactoryReturnsDiscardingWhenOptedOut() async {
        let sink = makeSTTMetricsSink(optedIn: false, fileURL: fileURL)
        await sink.record(makeMetric())
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: fileURL.path),
            "Opted-out factory must produce a no-op sink."
        )
    }

    func testFactoryReturnsJSONLWhenOptedIn() async throws {
        let sink = makeSTTMetricsSink(optedIn: true, fileURL: fileURL)
        await sink.record(makeMetric())
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: fileURL.path),
            "Opted-in factory must produce a writing sink."
        )
        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertFalse(raw.isEmpty)
    }

    // MARK: - EngineConfig.sttMetricsFileURL respects override

    func testEngineConfigTelemetryDirectoryHonorsEnvOverride() {
        let override = "/tmp/yooz-test-metrics-override-\(UUID().uuidString)"
        setenv("YOOZ_TELEMETRY_DIR", override, 1)
        defer { unsetenv("YOOZ_TELEMETRY_DIR") }
        XCTAssertEqual(
            EngineConfig.telemetryDirectory.path, override
        )
        XCTAssertEqual(
            EngineConfig.sttMetricsFileURL.path,
            "\(override)/stt_metrics.jsonl"
        )
    }

    // MARK: - Telemetry opt-in flip-flop end-to-end

    /// Opt in, record, opt out, record, opt in, record. Assert
    /// exactly the right number of JSONL lines land on disk after
    /// the round-trip. Catches a regression where the factory
    /// caches the opt-in flag or where the file is opened in
    /// truncate mode on second use.
    func testTelemetryOptInFlipFlopProducesExpectedLineCount() async throws {
        // Round 1 — opted in: one record, one line.
        let sink1 = makeSTTMetricsSink(optedIn: true, fileURL: fileURL)
        await sink1.record(makeMetric(backend: .parakeet))

        var raw = try String(contentsOf: fileURL, encoding: .utf8)
        var lines = raw.split(
            separator: "\n", omittingEmptySubsequences: true
        )
        XCTAssertEqual(
            lines.count, 1,
            "Opt-in round 1 should append exactly one line."
        )

        // Round 2 — opted out: discarded, file unchanged.
        let sink2 = makeSTTMetricsSink(optedIn: false, fileURL: fileURL)
        await sink2.record(makeMetric(backend: .qwen3ASRPreview))
        raw = try String(contentsOf: fileURL, encoding: .utf8)
        lines = raw.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(
            lines.count, 1,
            "Opt-out round 2 must NOT append a line."
        )

        // Round 3 — opted back in: one more line, total 2.
        let sink3 = makeSTTMetricsSink(optedIn: true, fileURL: fileURL)
        await sink3.record(makeMetric(backend: .appleSTT))
        raw = try String(contentsOf: fileURL, encoding: .utf8)
        lines = raw.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(
            lines.count, 2,
            "Opt-in round 3 should append a second line, not "
                + "truncate the existing record."
        )
    }

    // MARK: - JSONL sink failure counter

    /// JSONL sink exposes `failureCount()` so an operator-facing
    /// health endpoint can see whether the disk has been failing
    /// silently. This test forces a write failure by pointing the
    /// sink at a parent path that already exists as a regular
    /// file (so `createDirectory(... withIntermediateDirectories:
    /// true)` throws), then asserts the counter reflects the
    /// streak.
    func testJSONLSinkExposesFailureCounter() async throws {
        // Create a regular file at the path that the sink will
        // try to use as the parent directory. The
        // `ensureParentDirectoryExists` call throws because a
        // file already exists where a directory should be.
        let blockerPath = tempDir.appendingPathComponent(
            "blocker-as-parent"
        )
        try Data().write(to: blockerPath)
        let sinkURL = blockerPath.appendingPathComponent(
            "stt_metrics.jsonl"
        )
        let sink = JSONLMetricsSink(fileURL: sinkURL)

        // Healthy state before any record() call: 0.
        let initial = await sink.failureCount()
        XCTAssertEqual(initial, 0)

        // Two failed writes — counter increments per failure.
        await sink.record(makeMetric())
        await sink.record(makeMetric())
        let afterTwo = await sink.failureCount()
        XCTAssertEqual(
            afterTwo, 2,
            "Failure counter must increment on each failed write."
        )
    }
}
