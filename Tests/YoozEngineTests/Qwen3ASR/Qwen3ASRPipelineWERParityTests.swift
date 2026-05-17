// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

#if canImport(YoozEngine)
@testable import STTModule
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Phase 4 — WER parity tests against the Python reference, plus an
/// auto-language-identification check on a Persian clip.
final class Qwen3ASRPipelineWERParityTests: XCTestCase {

    private static let phase4Root: URL = {
        if let override = ProcessInfo.processInfo.environment[
            "YOOZ_PHASE4_ARTIFACTS"
        ] {
            return URL(fileURLWithPath: override)
        }
        return URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-46/phase4-bridge"
        )
    }()

    private static var checkpointDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private struct WERSubset: Decodable {
        struct Clip: Decodable {
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
        let languageCode: String
        let languageLabel: String
        let clips: [Clip]
        let pythonWER: Double
        let pythonCER: Double

        enum CodingKeys: String, CodingKey {
            case languageCode = "language_code"
            case languageLabel = "language_label"
            case clips
            case pythonWER = "python_wer"
            case pythonCER = "python_cer"
        }
    }

    private actor PipelineCache {
        private var pipeline: Qwen3ASRPipeline?
        func get(_ build: () async throws -> Qwen3ASRPipeline) async throws
            -> Qwen3ASRPipeline
        {
            if let pipeline { return pipeline }
            let p = try await build()
            pipeline = p
            return p
        }
    }
    private static let cache = PipelineCache()

    private static func loadPipeline() async throws -> Qwen3ASRPipeline {
        try await cache.get {
            try await Qwen3ASRPipeline.load(from: checkpointDir)
        }
    }

    // MARK: - WAV helpers (16-bit mono 16 kHz, no AVFoundation)

    private static func loadPCM(_ url: URL) throws -> [Float] {
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
                        samples[i] =
                            Float(int16Buffer[i]) / Float(Int16.max)
                    }
                }
                return samples
            }
            idx += 8 + Int(chunkSize)
        }
        throw NSError(
            domain: "test", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "no 'data' chunk"]
        )
    }

    // MARK: - WER

    private static func wer(reference: String, hypothesis: String) -> Double {
        let refTokens = referenceTokens(reference)
        let hypTokens = referenceTokens(hypothesis)
        if refTokens.isEmpty {
            return hypTokens.isEmpty ? 0.0 : 1.0
        }
        // Levenshtein over tokens. Phase 4 subsets are short (≤30
        // tokens) so the O(n·m) DP is fine.
        let n = refTokens.count
        let m = hypTokens.count
        var prev = [Int](0...m)
        var curr = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            curr[0] = i
            for j in 1...m {
                if refTokens[i - 1] == hypTokens[j - 1] {
                    curr[j] = prev[j - 1]
                } else {
                    curr[j] = min(prev[j - 1], prev[j], curr[j - 1]) + 1
                }
            }
            swap(&prev, &curr)
        }
        return Double(prev[m]) / Double(n)
    }

    /// Light-weight whitespace tokenization with punctuation stripped
    /// — matches the standard `jiwer.wer` default normalization
    /// closely enough for a ≤0.5 % parity bar. Production-grade WER
    /// would call into a normalization library; we don't add one
    /// here because the parity check just compares two such WERs.
    private static func referenceTokens(_ s: String) -> [String] {
        let normalized = s.lowercased()
        let stripped = normalized.unicodeScalars.map {
            scalar -> Character in
            if CharacterSet.punctuationCharacters.contains(scalar) {
                return " "
            }
            return Character(scalar)
        }
        return String(stripped)
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }

    // MARK: - Test driver

    private func runWERSubset(_ code: String, label: String) async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        let path = Self.phase4Root.appendingPathComponent(
            "wer_subset/\(code).json"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: path.path),
            "WER subset reference missing for \(code) — re-run "
                + "dump_wer_references.py"
        )
        let data = try Data(contentsOf: path)
        let subset = try JSONDecoder().decode(WERSubset.self, from: data)
        try XCTSkipUnless(
            !subset.clips.isEmpty,
            "WER subset \(code) is empty"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: subset.clips[0].wav),
            "Audio data not mounted at \(subset.clips[0].wav)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent(
                    "config.json"
                ).path
            ),
            "Checkpoint not available"
        )

        let pipeline = try await Self.loadPipeline()

        var refs: [String] = []
        var swiftHyps: [String] = []
        var pythonHyps: [String] = []
        for clip in subset.clips {
            let pcm = try Self.loadPCM(URL(fileURLWithPath: clip.wav))
            let result = try pipeline.transcribe(
                pcm: pcm, language: label, maxNewTokens: 256
            )
            refs.append(clip.reference)
            swiftHyps.append(result.text)
            pythonHyps.append(clip.hypothesisPython)
        }

        // Per-utterance WER + aggregate.
        var swiftAggNum = 0
        var swiftAggDen = 0
        var pyAggNum = 0
        var pyAggDen = 0
        for i in 0..<refs.count {
            let refTokens = Self.referenceTokens(refs[i])
            let swiftTokens = Self.referenceTokens(swiftHyps[i])
            let pyTokens = Self.referenceTokens(pythonHyps[i])
            // Use simple insert/sub/del Levenshtein.
            swiftAggNum += Self.editDistance(refTokens, swiftTokens)
            swiftAggDen += refTokens.count
            pyAggNum += Self.editDistance(refTokens, pyTokens)
            pyAggDen += refTokens.count
        }
        let swiftWER =
            swiftAggDen > 0 ? Double(swiftAggNum) / Double(swiftAggDen) : 0.0
        let pyWER =
            pyAggDen > 0 ? Double(pyAggNum) / Double(pyAggDen) : 0.0

        // Bar: Swift WER must be within 0.5 % absolute of the Python
        // reference. The Python reference WER itself is recorded in
        // the dump (`pythonWER`); we recompute here with our own
        // tokenizer to keep both sides on the same normalization.
        XCTAssertLessThanOrEqual(
            swiftWER, pyWER + 0.005,
            "[\(code)] Swift WER \(swiftWER) exceeded "
                + "Python WER \(pyWER) by more than 0.5 % absolute"
        )

        // Persist a per-language report so we have a record of
        // whether Swift hit the parity bar.
        try Self.writeWERReport(
            code: code, swiftWER: swiftWER, pythonWER: pyWER,
            refs: refs, swiftHyps: swiftHyps, pythonHyps: pythonHyps
        )
    }

    private static func editDistance(_ a: [String], _ b: [String]) -> Int {
        let n = a.count
        let m = b.count
        if n == 0 { return m }
        if m == 0 { return n }
        var prev = [Int](0...m)
        var curr = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            curr[0] = i
            for j in 1...m {
                if a[i - 1] == b[j - 1] {
                    curr[j] = prev[j - 1]
                } else {
                    curr[j] = min(prev[j - 1], prev[j], curr[j - 1]) + 1
                }
            }
            swap(&prev, &curr)
        }
        return prev[m]
    }

    private static func writeWERReport(
        code: String,
        swiftWER: Double,
        pythonWER: Double,
        refs: [String],
        swiftHyps: [String],
        pythonHyps: [String]
    ) throws {
        let dir = phase4Root.appendingPathComponent("results")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        let report: [String: Any] = [
            "language_code": code,
            "swift_wer": swiftWER,
            "python_wer": pythonWER,
            "delta_wer": swiftWER - pythonWER,
            "samples": (0..<refs.count).map { i -> [String: String] in
                [
                    "ref": refs[i],
                    "swift": swiftHyps[i],
                    "python": pythonHyps[i],
                ]
            },
        ]
        let data = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted]
        )
        try data.write(
            to: dir.appendingPathComponent("wer_\(code).json")
        )
    }

    // MARK: - Tests

    func testWERParityEN() async throws {
        try await runWERSubset("en", label: "English")
    }

    func testWERParityFA() async throws {
        try await runWERSubset("fa", label: "Persian")
    }

    func testWERParityAR() async throws {
        try await runWERSubset("ar", label: "Arabic")
    }

    /// Auto language identification: feed a Persian clip with no
    /// language hint. The pipeline's `extractLanguageAndContent`
    /// must surface `Persian` (canonical label) from the model's
    /// `language X<asr_text>` preamble.
    func testAutoLanguageIDPersianClip() async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: "/Volumes/S1/yooz/stt-test-data/persian/test_001.wav"
            ),
            "Persian test clip missing"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent(
                    "config.json"
                ).path
            ),
            "Checkpoint missing"
        )
        let pipeline = try await Self.loadPipeline()
        let pcm = try Self.loadPCM(
            URL(fileURLWithPath: "/Volumes/S1/yooz/stt-test-data/persian/test_001.wav")
        )
        let result = try pipeline.transcribe(
            pcm: pcm, language: nil, maxNewTokens: 256
        )
        XCTAssertEqual(result.language, "Persian")
        XCTAssertFalse(
            result.text.isEmpty,
            "Persian transcription must be non-empty"
        )
    }
}
