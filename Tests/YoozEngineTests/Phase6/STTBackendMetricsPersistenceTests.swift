// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

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
}
