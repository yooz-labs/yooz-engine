// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 7 — heavy end-to-end streaming tests for the qwen3 backend.
///
/// Mirrors the TCC / artifact gating from
/// `Qwen3ASREngineSmokeTests`: macOS Full Disk Access is required to
/// read `/Volumes/S1` from the GUI xctest host, so these tests skip
/// unless invoked from `swift test` (whose host bundle lives under
/// `.build/`) or with `YOOZ_RUN_TCC_TESTS=1`.
///
/// Tests covered here:
///
/// 1. End-to-end streaming smoke — chunk a 12.64 s clip into 320 ms
///    Float32 segments, drive the WS endpoint, assert the `final`
///    frame's text equals the Phase 4 canonical reference.
/// 2. Streaming-vs-offline WER delta on EN / AR / FA wer_subset.
///    Since our streaming pass IS the offline pass on the buffered
///    PCM, the per-clip text equality holds and the WER delta is 0.
///    The test still runs end-to-end so a future regression in the
///    streaming session (e.g. dropped samples, premature finalize)
///    surfaces immediately.
/// 3. Per-receive-callback latency benchmark — the receive callback
///    only buffers, so per-callback latency is microsecond-scale.
///    The interesting number is the total `final`-emit latency; we
///    record it to S1 for visibility.
/// 4. Concurrent stream isolation — two simultaneous WS connections,
///    different audio in each, assert each gets its own correct
///    transcription.
/// 5. Stream cancellation — client closes mid-stream after sending
///    half the audio; server cleans up without crashing.
final class Qwen3ASRStreamingSmokeTests: XCTestCase {

    // MARK: - Locations

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

    private static var werSubsetDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-46/phase4-bridge/"
                + "wer_subset"
        )
    }

    private static var resultsDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-46/phase7-streaming/results"
        )
    }

    // MARK: - Fixtures

    private struct CanonicalReference: Decodable {
        let transcriptionText: String
        enum CodingKeys: String, CodingKey {
            case transcriptionText = "transcription_text"
        }
    }

    private struct WERSubsetClip: Decodable {
        let clipId: String
        let wav: String
        let durationS: Double
        let reference: String
        let hypothesisPython: String

        enum CodingKeys: String, CodingKey {
            case clipId = "clip_id"
            case wav
            case durationS = "duration_s"
            case reference
            case hypothesisPython = "hypothesis_python"
        }
    }

    private struct WERSubset: Decodable {
        let languageCode: String
        let languageLabel: String
        let clips: [WERSubsetClip]

        enum CodingKeys: String, CodingKey {
            case languageCode = "language_code"
            case languageLabel = "language_label"
            case clips
        }
    }

    // MARK: - Skip gating

    private static func skipUnlessReady() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_RUN_TCC_TESTS"] == "1"
                || Bundle(for: Self.self).bundleURL.path.contains(".build/"),
            "Skipping /Volumes/S1-backed streaming smoke under xcodebuild "
                + "(macOS TCC). Run via `swift test` or set "
                + "YOOZ_RUN_TCC_TESTS=1 with Full Disk Access granted."
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: checkpointDir.path),
            "Qwen3-ASR checkpoint not at \(checkpointDir.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: canonicalAudio.path),
            "Canonical audio not at \(canonicalAudio.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: canonicalReference.path),
            "Phase 4 reference not at \(canonicalReference.path)"
        )
    }

    // MARK: - PCM / WAV helpers

    private static func loadPCM(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "wav too short: \(url.path)"]
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
        throw NSError(
            domain: "test", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "no 'data' chunk in \(url.path)"]
        )
    }

    /// Encode `[Float]` PCM as raw little-endian Float32 bytes ready
    /// for WS `.binary` framing. Float is IEEE 754 binary32; Apple
    /// Silicon is little-endian, so the in-memory layout matches the
    /// wire layout directly.
    private static func encodeFloat32(_ samples: ArraySlice<Float>) -> Data {
        var data = Data(count: samples.count * MemoryLayout<Float>.size)
        let count = samples.count
        data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            let floats = base.bindMemory(to: Float.self, capacity: count)
            var i = 0
            for s in samples {
                floats[i] = s
                i += 1
            }
        }
        return data
    }

    // MARK: - WS driver

    /// Stream a single clip over the WS endpoint and return the text
    /// of the `final` frame.
    private static func runStream(
        pcm: [Float],
        languageCode: String,
        chunkSamples: Int = 5_120,
        chunkDelayMs: UInt64? = nil
    ) async throws -> (text: String, partialCount: Int, totalLatencyMs: Double) {
        guard
            let url = URL(
                string: "ws://\(EngineConfig.host):\(EngineConfig.port)/v1/stt/stream"
            )
        else {
            throw NSError(domain: "test", code: 1)
        }
        let task = URLSession(configuration: .default).webSocketTask(with: url)
        task.resume()

        let started = Date()

        let configJSON = #"{"type":"config","language":"\#(languageCode)"}"#
        try await task.send(.string(configJSON))

        // Block until we see "ready" so the server has finished
        // loading the model. Receive frames in a tight loop because
        // `URLSessionWebSocketTask.receive` doesn't buffer messages
        // for us; callers must drain.
        let ready = try await task.receive()
        switch ready {
        case .string(let s):
            guard s.contains("\"ready\"") else {
                task.cancel(with: .normalClosure, reason: nil)
                throw NSError(
                    domain: "test", code: 2,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Expected ready, got: \(s)"
                    ]
                )
            }
        case .data:
            task.cancel(with: .normalClosure, reason: nil)
            throw NSError(
                domain: "test", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey: "Unexpected binary ready frame"
                ]
            )
        @unknown default:
            break
        }

        // Drain partials concurrently with sends so the server's
        // outbound buffer doesn't back up.
        var partialCount = 0
        var finalText: String? = nil

        let receiver = Task { () -> (Int, String?) in
            var partials = 0
            var final: String? = nil
            while final == nil {
                let msg: URLSessionWebSocketTask.Message
                do {
                    msg = try await task.receive()
                } catch {
                    break
                }
                let txt: String
                switch msg {
                case .string(let s): txt = s
                case .data(let d): txt = String(data: d, encoding: .utf8) ?? ""
                @unknown default: txt = ""
                }
                if txt.contains("\"final\"") {
                    final = Self.extractFinalText(from: txt)
                } else if txt.contains("\"partial\"") {
                    partials += 1
                } else if txt.contains("\"error\"") {
                    final = ""
                    break
                }
            }
            return (partials, final)
        }

        // Send the audio as 320 ms chunks (5120 samples at 16 kHz).
        var idx = 0
        while idx < pcm.count {
            let end = Swift.min(idx + chunkSamples, pcm.count)
            let chunk = Self.encodeFloat32(pcm[idx..<end])
            try await task.send(.data(chunk))
            idx = end
            if let delay = chunkDelayMs {
                try await Task.sleep(nanoseconds: delay * 1_000_000)
            }
        }
        // Closing the WS triggers the server's `final` emission.
        task.cancel(with: .normalClosure, reason: nil)
        let (partials, final) = await receiver.value
        partialCount = partials
        finalText = final

        let elapsedMs = Date().timeIntervalSince(started) * 1_000

        guard let final = finalText else {
            throw NSError(
                domain: "test", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "No final frame received"]
            )
        }
        return (final, partialCount, elapsedMs)
    }

    private static func extractFinalText(from json: String) -> String {
        // Lightweight extraction; full JSON decode would force us to
        // keep the WSSTTResult type stable across compilations of
        // the test target. The frame schema is stable:
        // `{"type":"final","text":"...","finalized":"...","draft":"..."}`.
        // Find the "text":"..." substring — this is the canonical
        // assertion target.
        guard let textRange = json.range(of: #""text":""#) else {
            return ""
        }
        let after = json[textRange.upperBound...]
        // Walk until an unescaped `"`.
        var output = ""
        var iterator = after.makeIterator()
        while let ch = iterator.next() {
            if ch == "\\" {
                if let next = iterator.next() {
                    switch next {
                    case "n": output.append("\n")
                    case "t": output.append("\t")
                    case "\"": output.append("\"")
                    case "\\": output.append("\\")
                    default: output.append(next)
                    }
                }
                continue
            }
            if ch == "\"" {
                break
            }
            output.append(ch)
        }
        return output
    }

    // MARK: - Server lifecycle

    @MainActor
    private static func withQwen3Server(
        _ body: @MainActor () async throws -> Void
    ) async throws {
        setenv("YOOZ_QWEN3_ASR_DIR", checkpointDir.path, 1)
        defer { unsetenv("YOOZ_QWEN3_ASR_DIR") }

        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)

        let server = APIServer()
        try await server.start()

        do {
            try await body()
        } catch {
            await server.stop()
            await YoozSTTEngine.shared.setBackend(.parakeet)
            throw error
        }
        await server.stop()
        await YoozSTTEngine.shared.setBackend(.parakeet)
    }

    // MARK: - Tests

    /// 1. End-to-end smoke: chunk the canonical 12.64 s clip into
    /// 320 ms chunks, stream over WS, assert final == Phase 4
    /// canonical reference.
    @MainActor
    func testStreamingSmokeMatchesCanonicalReference() async throws {
        try Self.skipUnlessReady()

        let referenceData = try Data(contentsOf: Self.canonicalReference)
        let reference = try JSONDecoder().decode(
            CanonicalReference.self, from: referenceData
        )
        let pcm = try Self.loadPCM(from: Self.canonicalAudio)

        try await Self.withQwen3Server {
            let outcome = try await Self.runStream(
                pcm: pcm,
                languageCode: "en",
                chunkSamples: 5_120
            )
            XCTAssertGreaterThan(
                outcome.partialCount, 0,
                "Server must emit at least one partial heartbeat"
            )
            XCTAssertEqual(
                outcome.text, reference.transcriptionText,
                "Streaming `final` text diverged from Phase 4 reference"
            )
            try Self.writeLatencyResult(
                clipId: "canonical_test_001",
                durationS: 12.64,
                totalLatencyMs: outcome.totalLatencyMs,
                partialCount: outcome.partialCount
            )
        }
    }

    /// 2. WER-delta vs offline. Since our streaming pass IS the
    /// offline pass on the buffered PCM, per-clip text equality holds
    /// and per-language WER delta is 0. We log both to S1.
    @MainActor
    func testStreamingMatchesOfflineOnWERSubsets() async throws {
        try Self.skipUnlessReady()

        try await Self.withQwen3Server {
            var perLanguage: [(String, Int, Int)] = []  // (lang, total, mismatch)
            for languageCode in ["en", "ar", "fa"] {
                let url = Self.werSubsetDir.appendingPathComponent(
                    "\(languageCode).json"
                )
                guard FileManager.default.fileExists(atPath: url.path) else {
                    XCTFail("WER subset missing: \(url.path)")
                    return
                }
                let data = try Data(contentsOf: url)
                let subset = try JSONDecoder().decode(
                    WERSubset.self, from: data
                )
                var mismatches = 0
                for clip in subset.clips.prefix(5) {
                    let wavURL = URL(fileURLWithPath: clip.wav)
                    guard FileManager.default.fileExists(
                        atPath: wavURL.path
                    ) else {
                        XCTFail("Clip wav missing: \(clip.wav)")
                        continue
                    }
                    let pcm = try Self.loadPCM(from: wavURL)
                    let outcome = try await Self.runStream(
                        pcm: pcm,
                        languageCode: languageCode,
                        chunkSamples: 5_120
                    )
                    if outcome.text != clip.hypothesisPython {
                        mismatches += 1
                    }
                }
                perLanguage.append(
                    (languageCode, subset.clips.prefix(5).count, mismatches)
                )
            }
            try Self.writeWERResult(perLanguage: perLanguage)

            for (lang, total, mismatch) in perLanguage {
                XCTAssertEqual(
                    mismatch, 0,
                    "Streaming/offline mismatch for \(lang): "
                        + "\(mismatch)/\(total) clips diverged. The streaming "
                        + "pipeline must produce text identical to the offline "
                        + "pipeline on the same PCM."
                )
            }
        }
    }

    /// 3. Concurrent stream isolation: two simultaneous connections
    /// receive different audio, assert each gets only its own
    /// transcription. We use the full canonical clip for stream A
    /// and the first half for stream B. The two transcriptions
    /// differ in the second clause; cross-contamination would show
    /// up as B getting A's full text (or vice versa).
    @MainActor
    func testConcurrentStreamsAreIsolated() async throws {
        try Self.skipUnlessReady()

        let referenceData = try Data(contentsOf: Self.canonicalReference)
        let reference = try JSONDecoder().decode(
            CanonicalReference.self, from: referenceData
        )
        let pcm = try Self.loadPCM(from: Self.canonicalAudio)
        let pcmA = pcm
        let pcmB = Array(pcm.prefix(pcm.count / 2))

        try await Self.withQwen3Server {
            async let outcomeA = Self.runStream(
                pcm: pcmA, languageCode: "en", chunkSamples: 5_120
            )
            async let outcomeB = Self.runStream(
                pcm: pcmB, languageCode: "en", chunkSamples: 5_120
            )
            let a = try await outcomeA
            let b = try await outcomeB
            XCTAssertEqual(
                a.text, reference.transcriptionText,
                "Stream A (full audio) should produce the canonical reference"
            )
            XCTAssertNotEqual(
                b.text, reference.transcriptionText,
                "Stream B (half audio) should NOT match the full-audio "
                    + "canonical reference; matching means sessions are "
                    + "sharing buffers."
            )
            XCTAssertFalse(
                b.text.isEmpty,
                "Stream B must still produce a non-empty transcription"
            )
        }
    }

    /// 4. Cancellation: client closes the connection after sending
    /// only half the audio. Server must still emit a `final` frame
    /// (with text from the partial buffer) and clean up without
    /// crashing.
    @MainActor
    func testStreamCancellationMidStream() async throws {
        try Self.skipUnlessReady()

        let pcm = try Self.loadPCM(from: Self.canonicalAudio)
        let half = Array(pcm.prefix(pcm.count / 2))

        try await Self.withQwen3Server {
            // We don't assert exact text on the partial buffer (the
            // model is non-causal and the second half of the
            // utterance carries information). We assert the server
            // returns a non-throwing `final` frame and doesn't hang.
            let outcome = try await Self.runStream(
                pcm: half, languageCode: "en", chunkSamples: 5_120
            )
            XCTAssertFalse(
                outcome.text.isEmpty,
                "Server must emit a non-empty final after mid-stream close"
            )
        }
    }

    // MARK: - Result persistence

    private static func writeLatencyResult(
        clipId: String,
        durationS: Double,
        totalLatencyMs: Double,
        partialCount: Int
    ) throws {
        let dir = resultsDir
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        let payload: [String: Any] = [
            "clip_id": clipId,
            "audio_duration_s": durationS,
            "total_latency_ms": totalLatencyMs,
            "partial_count": partialCount,
            "chunk_size_ms": 320,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(
            to: dir.appendingPathComponent("PHASE7_LATENCY_\(clipId).json")
        )
    }

    private static func writeWERResult(
        perLanguage: [(String, Int, Int)]
    ) throws {
        let dir = resultsDir
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        var lines: [String] = []
        lines.append("# Phase 7 streaming-vs-offline WER delta")
        lines.append("")
        lines.append("Streaming uses the buffered offline pass; per-clip text")
        lines.append("equality holds by construction. A non-zero mismatch")
        lines.append("indicates a regression in the streaming session (lost")
        lines.append("samples, premature finalize, or backend-actor cross-talk).")
        lines.append("")
        lines.append("| Language | Clips | Mismatches |")
        lines.append("|---|---|---|")
        for (lang, total, mismatch) in perLanguage {
            lines.append("| \(lang) | \(total) | \(mismatch) |")
        }
        let text = lines.joined(separator: "\n") + "\n"
        try text.write(
            to: dir.appendingPathComponent("PHASE7_STREAMING_WER.md"),
            atomically: true,
            encoding: .utf8
        )
    }
}
