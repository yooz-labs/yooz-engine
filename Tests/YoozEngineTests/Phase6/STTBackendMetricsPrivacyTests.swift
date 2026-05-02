// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

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
        // The default-construct path is "no env var set."
        unsetenv("YOOZ_TELEMETRY_STT")
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
}
