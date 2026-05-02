// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

/// Sink that consumes `STTBackendMetrics` records. The default
/// implementation is opt-in; the engine wires up either a
/// `JSONLMetricsSink` (when `EngineConfig.telemetryOptedIn` is
/// `true`) or a `DiscardingMetricsSink` otherwise.
///
/// The sink is the only place metrics records are persisted. Other
/// components (HTTP routes, log statements, etc.) MUST NOT serialize
/// the struct independently — privacy invariants are enforced here.
public protocol STTMetricsSink: Sendable {
    /// Persist a metric event. Implementations must complete quickly
    /// (no blocking) so they can be `await`ed from request hot paths
    /// without distorting end-to-end latency.
    func record(_ metric: STTBackendMetrics) async
}

// MARK: - Discarding sink (telemetry off)

/// No-op sink used when telemetry is opted out (the default). Records
/// are dropped on the floor; nothing is written, nothing is logged.
public struct DiscardingMetricsSink: STTMetricsSink {
    public init() {}
    public func record(_ metric: STTBackendMetrics) async {}
}

// MARK: - JSONL sink (telemetry on)

/// Append-only JSONL sink. One record per line at
/// `~/Library/Application Support/YoozEngine/telemetry/stt_metrics.jsonl`
/// by default; override the directory via `EngineConfig.telemetryDirectory`
/// (which honors the `YOOZ_TELEMETRY_DIR` env var for tests).
public actor JSONLMetricsSink: STTMetricsSink {

    private let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "STTMetricsSink"
    )

    private let fileURL: URL
    private let encoder: JSONEncoder
    /// Number of consecutive write failures since the last success.
    /// Useful for an operator-facing health surface (`/v1/health`)
    /// without flooding the log on a stuck disk: we log loudly only
    /// on the first failure of a streak.
    private var consecutiveWriteFailures: Int = 0

    public init(fileURL: URL) {
        self.fileURL = fileURL
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        // Sorted keys keep the JSONL diffable; not required for
        // correctness but makes the privacy test's key-set assertion
        // easier to reason about for human reviewers.
        encoder.outputFormatting = [.sortedKeys]
        self.encoder = encoder
    }

    public func record(_ metric: STTBackendMetrics) async {
        do {
            try ensureParentDirectoryExists()
            let payload = try encoder.encode(metric)
            try appendLine(payload)
            consecutiveWriteFailures = 0
        } catch {
            // The sink runs on every transcribe call. A telemetry
            // write failure must NEVER surface to the caller. We log
            // loudly only on the first failure of a streak so a
            // stuck disk doesn't flood the log; the
            // `consecutiveWriteFailures` counter is the operator
            // signal (queryable via `failureCount()`).
            consecutiveWriteFailures += 1
            if consecutiveWriteFailures == 1 {
                logger.error(
                    "Failed to record STT metric (first of streak): \(String(describing: error), privacy: .public)"
                )
            } else {
                logger.debug(
                    "Failed to record STT metric (streak=\(self.consecutiveWriteFailures)): \(String(describing: error), privacy: .public)"
                )
            }
        }
    }

    /// Number of consecutive write failures since the last success.
    /// Zero means the sink is healthy. Exposed for operator-facing
    /// health endpoints; never read on the hot transcribe path.
    public func failureCount() -> Int { consecutiveWriteFailures }

    /// Used by tests to assert "no file written when opted out". Not
    /// part of the public surface.
    var fileURLForTesting: URL { fileURL }

    private func ensureParentDirectoryExists() throws {
        let parent = fileURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parent.path) {
            try FileManager.default.createDirectory(
                at: parent,
                withIntermediateDirectories: true
            )
        }
    }

    private func appendLine(_ payload: Data) throws {
        var line = payload
        line.append(0x0A)  // '\n'

        if !FileManager.default.fileExists(atPath: fileURL.path) {
            try line.write(to: fileURL, options: .atomic)
            return
        }

        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
    }
}

// MARK: - Factory

extension STTMetricsSink where Self == DiscardingMetricsSink {
    /// Sink to use when telemetry is opted out.
    public static var discarding: DiscardingMetricsSink {
        DiscardingMetricsSink()
    }
}

/// Pick the right sink for a given opt-in flag and target file URL.
/// Used by the engine wiring path; tests construct sinks directly
/// against a temp directory.
public func makeSTTMetricsSink(
    optedIn: Bool,
    fileURL: URL
) -> any STTMetricsSink {
    optedIn
        ? JSONLMetricsSink(fileURL: fileURL)
        : DiscardingMetricsSink()
}
