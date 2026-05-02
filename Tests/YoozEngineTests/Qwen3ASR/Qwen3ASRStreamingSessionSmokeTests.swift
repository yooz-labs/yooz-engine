// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

#if canImport(YoozEngine)
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Phase 7 streaming session smoke test that runs from `swift test`
/// (i.e. the SwiftPM Qwen3ASRTests target). The xcodebuild
/// counterpart (`Qwen3ASRStreamingSmokeTests` under
/// `Tests/YoozEngineTests/Phase7/`) covers the full WS roundtrip but
/// is skipped under the GUI xctest host because of macOS TCC.
///
/// This test exercises the same code path the WS handler uses — push
/// PCM in 320 ms chunks into a `Qwen3ASRStreamingSession`, run
/// `finalize()`, assert the resulting text equals the Phase 4
/// canonical reference. The only thing it doesn't cover is the WS
/// transport layer.
final class Qwen3ASRStreamingSessionSmokeTests: XCTestCase {

    private static var checkpointDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private static var canonicalAudio: URL {
        URL(fileURLWithPath: "/Volumes/S1/yooz/stt-test-data/english/test_001.wav")
    }

    private static var canonicalReference: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-46/phase4-bridge/"
                + "reference/canonical_transcription.json"
        )
    }

    private struct CanonicalReference: Decodable {
        let transcriptionText: String
        enum CodingKeys: String, CodingKey {
            case transcriptionText = "transcription_text"
        }
    }

    private static func loadPCM(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "wav too short"]
            )
        }
        var idx = 12
        while idx + 8 <= data.count {
            let chunkId = data.subdata(in: idx..<(idx + 4))
            let chunkSize = data.withUnsafeBytes {
                (raw: UnsafeRawBufferPointer) -> UInt32 in
                let base = raw.baseAddress!.advanced(by: idx + 4)
                return base.loadUnaligned(as: UInt32.self)
            }
            if chunkId == "data".data(using: .ascii) {
                let payload = data.subdata(
                    in: (idx + 8)..<(idx + 8 + Int(chunkSize))
                )
                let sampleCount = payload.count / MemoryLayout<Int16>.size
                var samples = [Float](repeating: 0.0, count: sampleCount)
                payload.withUnsafeBytes { raw in
                    let int16Buffer = raw.bindMemory(to: Int16.self)
                    for i in 0..<sampleCount {
                        samples[i] = Float(int16Buffer[i]) / Float(Int16.max)
                    }
                }
                return samples
            }
            idx += 8 + Int(chunkSize)
        }
        throw NSError(domain: "test", code: 1)
    }

    /// 1. End-to-end smoke against the Phase 4 canonical reference.
    /// Chunks the canonical 12.64 s clip into 320 ms slices, pushes
    /// each into the session, runs `finalize()`, asserts the text
    /// matches the offline reference verbatim.
    func testStreamingSessionMatchesCanonicalReference() async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalAudio.path),
            "Canonical audio not at \(Self.canonicalAudio.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalReference.path),
            "Phase 4 reference not at \(Self.canonicalReference.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent("config.json").path
            ),
            "Qwen3-ASR checkpoint not at \(Self.checkpointDir.path)"
        )

        let referenceData = try Data(contentsOf: Self.canonicalReference)
        let reference = try JSONDecoder().decode(
            CanonicalReference.self, from: referenceData
        )
        let pcm = try Self.loadPCM(from: Self.canonicalAudio)

        // Pre-load the backend so the streaming session's finalize
        // call doesn't pay the 1 s pipeline-load cost inside the test
        // body. This mirrors the real WS flow: clients call
        // /v1/stt/load before opening the WS.
        try await Qwen3ASRBackend.shared.ensureLoaded(
            modelDir: Self.checkpointDir
        )

        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 60
        )

        // 320 ms chunks at 16 kHz = 5120 samples.
        let chunkSize = 5_120
        var idx = 0
        var partialCount = 0
        var maxPushLatencyMs: Double = 0
        while idx < pcm.count {
            let end = Swift.min(idx + chunkSize, pcm.count)
            let chunk = Array(pcm[idx..<end])
            let started = Date()
            let count = try await session.push(samples: chunk)
            let elapsed = Date().timeIntervalSince(started) * 1_000
            maxPushLatencyMs = Swift.max(maxPushLatencyMs, elapsed)
            XCTAssertGreaterThanOrEqual(count, end)
            idx = end
            partialCount += 1
        }
        XCTAssertGreaterThan(partialCount, 0)
        // Per-chunk push must be far below the 320 ms chunk
        // duration. The buffer-only push is microsecond-scale, so a
        // P_max above even 50 ms here would be a regression.
        XCTAssertLessThan(
            maxPushLatencyMs, 50.0,
            "Per-chunk push latency \(maxPushLatencyMs) ms exceeded the 50 ms guard rail"
        )

        let finalized = try await session.finalize()
        XCTAssertEqual(
            finalized.text, reference.transcriptionText,
            "Streaming session final text diverged from Phase 4 reference"
        )
        XCTAssertEqual(finalized.language, "English")
    }

    /// 2. Two concurrent sessions on the same backend actor must
    /// each produce the right transcription. Stream A buffers the
    /// full canonical clip; stream B buffers the first half. The
    /// two transcriptions differ in the second clause; cross-
    /// contamination (shared buffer, leaked state across the actor
    /// boundary) would surface as B matching A's full text.
    func testConcurrentStreamingSessionsAreIsolated() async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalAudio.path)
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalReference.path)
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent("config.json").path
            )
        )

        let referenceData = try Data(contentsOf: Self.canonicalReference)
        let reference = try JSONDecoder().decode(
            CanonicalReference.self, from: referenceData
        )
        let pcm = try Self.loadPCM(from: Self.canonicalAudio)
        let pcmA = pcm
        let pcmB = Array(pcm.prefix(pcm.count / 2))

        try await Qwen3ASRBackend.shared.ensureLoaded(
            modelDir: Self.checkpointDir
        )

        async let outcomeA = Self.runSessionEndToEnd(pcm: pcmA)
        async let outcomeB = Self.runSessionEndToEnd(pcm: pcmB)
        let (a, b) = try await (outcomeA, outcomeB)
        XCTAssertEqual(
            a.text, reference.transcriptionText,
            "Stream A (full audio) should match the canonical reference"
        )
        XCTAssertNotEqual(
            b.text, reference.transcriptionText,
            "Stream B (half audio) must NOT match the full-audio reference"
        )
        XCTAssertFalse(
            b.text.isEmpty,
            "Stream B must still produce a non-empty transcription"
        )
    }

    /// 3. Cancellation: a session that finalizes after only half the
    /// PCM has been pushed must still return non-empty text. The
    /// model is non-causal, so we don't assert exact equality with
    /// the half-clip transcription — only that the path completes
    /// without hanging or throwing.
    func testStreamingSessionMidStreamCancellation() async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalAudio.path)
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent("config.json").path
            )
        )

        let pcm = try Self.loadPCM(from: Self.canonicalAudio)
        let half = Array(pcm.prefix(pcm.count / 2))

        try await Qwen3ASRBackend.shared.ensureLoaded(
            modelDir: Self.checkpointDir
        )

        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 60
        )

        let chunkSize = 5_120
        var idx = 0
        while idx < half.count {
            let end = Swift.min(idx + chunkSize, half.count)
            try await session.push(samples: Array(half[idx..<end]))
            idx = end
        }
        let finalized = try await session.finalize()
        XCTAssertFalse(finalized.text.isEmpty)
    }

    // MARK: - Helpers

    private static func runSessionEndToEnd(
        pcm: [Float]
    ) async throws -> Qwen3ASRStreamingSession.FinalResult {
        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 60
        )
        let chunkSize = 5_120
        var idx = 0
        while idx < pcm.count {
            let end = Swift.min(idx + chunkSize, pcm.count)
            try await session.push(samples: Array(pcm[idx..<end]))
            idx = end
        }
        return try await session.finalize()
    }
}
