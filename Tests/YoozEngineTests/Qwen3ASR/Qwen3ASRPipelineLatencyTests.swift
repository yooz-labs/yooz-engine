// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

#if canImport(YoozEngine)
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Phase 4 -- latency micro-benchmark. Records cold-start, warm 2 s /
/// 5 s / 15 s clip latency and peak resident memory to a markdown
/// report. The test always passes; it surfaces numbers, never gates.
/// Run with `swift test --filter Qwen3ASRPipelineLatencyTests`.
final class Qwen3ASRPipelineLatencyTests: XCTestCase {

    private static var checkpointDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private static var phase4Root: URL {
        if let override = ProcessInfo.processInfo.environment[
            "YOOZ_PHASE4_ARTIFACTS"
        ] {
            return URL(fileURLWithPath: override)
        }
        return URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-46/phase4-bridge"
        )
    }

    private static var canonicalAudio: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/stt-test-data/english/test_001.wav"
        )
    }

    private static func loadPCM(_ url: URL) throws -> [Float] {
        // Same minimal WAV reader used by the WER tests; duplicated
        // here to keep test files isolated from each other.
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

    /// Capture the current process resident set in MB. macOS only
    /// (the Phase 4 deliverable is a macOS engine).
    private static func residentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size
                / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0, &count
                )
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(info.resident_size) / 1_048_576.0
    }

    func testLatencyMicroBenchmark() async throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent(
                    "config.json"
                ).path
            ),
            "Checkpoint not available"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalAudio.path),
            "Canonical audio not available"
        )

        let basePCM = try Self.loadPCM(Self.canonicalAudio)
        // Source clip is 12.6 s; truncate / repeat to get 2 s, 5 s,
        // 15 s synthetic durations. Real-world latency depends on
        // duration, not specific content, so synthetic slicing is
        // appropriate for this micro-benchmark.
        func slice(toSeconds seconds: Double) -> [Float] {
            let target = Int(seconds * 16_000)
            if target <= basePCM.count {
                return Array(basePCM[..<target])
            }
            var buf = basePCM
            buf.reserveCapacity(target)
            while buf.count < target {
                buf.append(contentsOf: basePCM)
            }
            return Array(buf[..<target])
        }

        let memBeforeLoad = Self.residentMemoryMB()
        let coldStart = Date()
        let pipeline = try await Qwen3ASRPipeline.load(
            from: Self.checkpointDir
        )
        let coldStartElapsed = Date().timeIntervalSince(coldStart)
        let memAfterLoad = Self.residentMemoryMB()

        // Warm-up: tiny clip so MLX can JIT every kernel before we
        // start timing.
        _ = try pipeline.transcribe(
            pcm: slice(toSeconds: 1.0), language: "English",
            maxNewTokens: 32
        )

        // Three durations × 2 trials each.
        let durations = [2.0, 5.0, 15.0]
        var rows: [(Double, Double, Int)] = []
        for d in durations {
            let pcm = slice(toSeconds: d)
            // Trial 1
            let t1Start = Date()
            let r1 = try pipeline.transcribe(
                pcm: pcm, language: "English", maxNewTokens: 256
            )
            let t1 = Date().timeIntervalSince(t1Start)
            // Trial 2
            let t2Start = Date()
            let r2 = try pipeline.transcribe(
                pcm: pcm, language: "English", maxNewTokens: 256
            )
            let t2 = Date().timeIntervalSince(t2Start)
            let warmAvg = (t1 + t2) / 2.0
            rows.append((d, warmAvg, max(r1.numAudioTokens, r2.numAudioTokens)))
        }
        let memPeak = Self.residentMemoryMB()

        // Persist to results dir.
        let outDir = Self.phase4Root.appendingPathComponent("results")
        try FileManager.default.createDirectory(
            at: outDir, withIntermediateDirectories: true
        )
        var md = "# Phase 4 — Qwen3-ASR Pipeline Latency\n\n"
        md += "Run via `swift test --filter "
        md += "Qwen3ASRPipelineLatencyTests` on the canonical 1.7B-8bit "
        md += "checkpoint.\n\n"
        md += "## Cold start\n\n"
        md += "- Pipeline load: \(String(format: "%.2f", coldStartElapsed)) s\n"
        md += "- Resident memory before load: "
        md += "\(String(format: "%.0f", memBeforeLoad)) MB\n"
        md += "- Resident memory after load: "
        md += "\(String(format: "%.0f", memAfterLoad)) MB\n"
        md += "- Resident memory peak after warm runs: "
        md += "\(String(format: "%.0f", memPeak)) MB\n\n"
        md += "## Warm transcription latency\n\n"
        md += "| Clip duration | Avg latency | Audio tokens |\n"
        md += "| --- | --- | --- |\n"
        for (d, t, n) in rows {
            md += "| \(String(format: "%.0f", d)) s "
            md += "| \(String(format: "%.3f", t)) s "
            md += "| \(n) |\n"
        }
        md += "\nWarm latency is the average of two back-to-back runs "
        md += "after a 1 s warm-up; greedy decode, max 256 tokens.\n"
        try md.write(
            to: outDir.appendingPathComponent("PHASE4_LATENCY.md"),
            atomically: true, encoding: .utf8
        )
    }
}
