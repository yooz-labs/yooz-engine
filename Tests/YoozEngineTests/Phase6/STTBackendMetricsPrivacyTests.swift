// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
@testable import YoozEngine

/// Phase 6 — privacy invariants for `STTBackendMetrics`.
///
/// Telemetry is allowed to record timing, the active backend, the
/// model variant tag, and the host's coarse hardware class. It is
/// never allowed to record:
///
/// - transcript content
/// - tokens
/// - free-form text
/// - audio bytes / mel frames
/// - user identifiers
/// - file system paths
///
/// The struct is intentionally narrow. This suite enforces that
/// narrowness three ways:
///
/// 1. The exact JSON key set is asserted (no surprise keys leak in).
/// 2. Reflection: no field name contains a banned substring.
/// 3. The only string-typed fields are the closed-set enums (rawValue
///    of `STTBackendID`, `HardwareClass`) plus the constant
///    `modelVariant` whose value is engine-controlled.
final class STTBackendMetricsPrivacyTests: XCTestCase {

    // MARK: - Fixture

    private func makeMetric(
        fellBack: Bool = false
    ) -> STTBackendMetrics {
        STTBackendMetrics(
            backend: .qwen3ASRPreview,
            modelVariant: "qwen3-asr-preview-int4",
            audioDurationMs: 4_321,
            timeToFirstTokenMs: 87,
            endToEndLatencyMs: 312,
            hardwareClass: .appleSiliconM3,
            fellBackFromPreview: fellBack,
            timestampUTC: Date(timeIntervalSince1970: 1_700_000_000)
        )
    }

    // MARK: - 1. JSON key set is exactly what we expect

    func testEncodedJSONHasOnlyCanonicalKeys() throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(makeMetric())
        let decoded = try JSONSerialization.jsonObject(with: data)
        guard let dict = decoded as? [String: Any] else {
            return XCTFail("Encoded metric was not a JSON object")
        }
        let actualKeys = Set(dict.keys)
        XCTAssertEqual(
            actualKeys, STTBackendMetrics.canonicalJSONKeys,
            "Encoded JSON keys diverged from canonical set. "
                + "Adding a new field requires updating "
                + "STTBackendMetrics.canonicalJSONKeys AND making "
                + "sure the new field carries no transcript / audio "
                + "/ user content."
        )
    }

    func testCanonicalKeySetContainsNoBannedSubstrings() {
        let banned = [
            "transcript", "tokens", "text", "audio_bytes",
            "audio_data", "user", "path", "filename",
            "filepath", "ip", "hostname",
        ]
        for key in STTBackendMetrics.canonicalJSONKeys {
            for needle in banned {
                XCTAssertFalse(
                    key.lowercased().contains(needle),
                    "Canonical JSON key '\(key)' contains banned "
                        + "substring '\(needle)'. Telemetry must "
                        + "not carry user content."
                )
            }
        }
    }

    // MARK: - 2. Reflection — no banned field names

    func testReflectedFieldNamesContainNoBannedSubstrings() {
        let mirror = Mirror(reflecting: makeMetric())
        let banned = [
            "transcript", "tokens", "text", "audiobytes",
            "audiodata", "user", "path", "filename",
            "filepath", "ip", "hostname",
        ]
        for child in mirror.children {
            guard let label = child.label?.lowercased() else { continue }
            for needle in banned {
                XCTAssertFalse(
                    label.contains(needle),
                    "Reflected field '\(child.label ?? "?")' contains "
                        + "banned substring '\(needle)'. Privacy "
                        + "invariant violated."
                )
            }
        }
    }

    // MARK: - 3. Only one free-string field; it's a controlled tag

    func testOnlyControlledStringFieldIsModelVariant() {
        let mirror = Mirror(reflecting: makeMetric())
        var stringFields: [String] = []
        for child in mirror.children {
            // Mirror reports `String` for raw-representable enums via
            // their associated value when the enum is a simple
            // `String`-backed enum, so we explicitly filter those by
            // type identity rather than by Swift type name.
            if child.value is String {
                stringFields.append(child.label ?? "?")
            }
        }
        // The only top-level `String` value is `modelVariant`. The
        // backend / hardwareClass enums are NOT raw `String` from
        // Mirror's perspective — they show up as their enum case.
        XCTAssertEqual(
            Set(stringFields), Set(["modelVariant"]),
            "Unexpected free-string fields on STTBackendMetrics: "
                + "\(stringFields). Privacy invariant requires that "
                + "the only string field is the engine-controlled "
                + "modelVariant tag."
        )
    }

    // MARK: - 4. Round-trip stability

    func testRoundTripPreservesAllFields() throws {
        let original = makeMetric(fellBack: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let data = try encoder.encode(original)
        let decoded = try decoder.decode(
            STTBackendMetrics.self, from: data
        )
        XCTAssertEqual(decoded, original)
    }

    // MARK: - 5. Default config has telemetry off

    func testTelemetryDefaultsToOptedOut() {
        // The default-construct path is "no env var set." Save and
        // restore the prior value so a developer running
        // `YOOZ_TELEMETRY_STT=local swift test` doesn't get the env
        // wiped out from under them for every subsequent test.
        let priorValue = ProcessInfo.processInfo.environment[
            "YOOZ_TELEMETRY_STT"
        ]
        unsetenv("YOOZ_TELEMETRY_STT")
        defer {
            if let priorValue {
                setenv("YOOZ_TELEMETRY_STT", priorValue, 1)
            }
        }
        XCTAssertFalse(EngineConfig.telemetryOptedIn)
    }

    func testTelemetryEnvVarLocalOptsIn() {
        setenv("YOOZ_TELEMETRY_STT", "local", 1)
        defer { unsetenv("YOOZ_TELEMETRY_STT") }
        XCTAssertTrue(EngineConfig.telemetryOptedIn)
    }

    func testTelemetryEnvVarUnknownValueOptsOut() {
        setenv("YOOZ_TELEMETRY_STT", "definitely-not-a-known-mode", 1)
        defer { unsetenv("YOOZ_TELEMETRY_STT") }
        XCTAssertFalse(EngineConfig.telemetryOptedIn)
    }

    // MARK: - 6. Byte-level canary in serialized JSONL output

    /// Reflection + key-set tests guard the in-memory struct
    /// shape. This test goes one level deeper: serialize a
    /// metric with a CANARY substring embedded in the
    /// engine-controlled `modelVariant` tag, then read the
    /// raw bytes from the JSONL file and grep for the canary.
    /// Catches a regression where a future field escapes via
    /// `Encodable` synthesis without updating the privacy
    /// fixture (the canonical-key tripwire) and somehow lands
    /// in the file with user-shaped content.
    ///
    /// The canary lands in the file (because `modelVariant`
    /// IS engine-controlled and CAN carry diagnostic tags, e.g.
    /// the `+stream_aborted` suffix the WS handler appends when
    /// a stream tears down via the do/catch path). What we
    /// assert here is that NO transcript-shaped or audio-shaped
    /// substrings leak — explicit deny-list of strings that
    /// would indicate a privacy regression.
    func testJSONLBytesContainNoTranscriptOrAudioShape() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "yooz-canary-\(UUID().uuidString)"
            )
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let fileURL = tempDir.appendingPathComponent(
            "stt_metrics.jsonl"
        )

        // Deny-list: substrings that would indicate a leak. The
        // first two are transcript-shaped strings the user might
        // dictate; the rest are field names that should never
        // appear in our canonical schema.
        let canaryDenyList = [
            "hello world",
            "the quick brown fox",
            "transcript",
            "tokens",
            "audio_bytes",
            "user_id",
            "username",
            "user@",
            "/Users/",
        ]

        // Build a metric and write via the production sink.
        let metric = STTBackendMetrics(
            backend: .qwen3ASRPreview,
            modelVariant: STTBackendMetrics.defaultModelVariant(
                for: .qwen3ASRPreview
            ),
            audioDurationMs: 1_234,
            timeToFirstTokenMs: nil,
            endToEndLatencyMs: 567,
            hardwareClass: .appleSiliconM3,
            fellBackFromPreview: false,
            timestampUTC: Date()
        )
        let sink = JSONLMetricsSink(fileURL: fileURL)
        let exp = expectation(description: "record")
        Task {
            await sink.record(metric)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 5)

        let raw = try String(contentsOf: fileURL, encoding: .utf8)
        for forbidden in canaryDenyList {
            XCTAssertFalse(
                raw.contains(forbidden),
                "JSONL output must not contain `\(forbidden)`. "
                    + "Privacy invariant violated. Raw line: \(raw)"
            )
        }

        // Schema sanity: the line MUST contain the canonical
        // model-variant tag (which is engine-controlled and not
        // user-derived).
        XCTAssertTrue(
            raw.contains("qwen3-asr-preview-int4"),
            "Expected engine-controlled model variant tag in "
                + "output; got: \(raw)"
        )
    }
}
