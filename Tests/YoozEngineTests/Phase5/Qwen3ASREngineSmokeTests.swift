// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
@testable import YoozEngine

/// Phase 5 — end-to-end smoke test for `/v1/stt/batch` with the
/// qwen3 backend selected. Loads the canonical Qwen3-ASR checkpoint
/// from `/Volumes/S1` (skips when not mounted) and asserts the HTTP
/// route returns the same transcription text as the Phase 4
/// canonical reference.
///
/// Boots a real `APIServer` on its localhost port and dials it via
/// `URLSession`, mirroring `Qwen3ASREngineRouteTests`. We avoid
/// `HummingbirdTesting` for the same Logging-resolution linker
/// reason documented there.
final class Qwen3ASREngineSmokeTests: XCTestCase {

    private static var checkpointDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private static var canonicalAudio: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/stt-test-data/english/test_001.wav"
        )
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

    private static func loadCanonicalPCM() throws -> [Float] {
        let data = try Data(contentsOf: canonicalAudio)
        guard data.count > 44 else {
            throw NSError(domain: "test", code: 1)
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

    @MainActor
    func testBatchRouteWithQwen3MatchesPhase4Reference() async throws {
        // macOS TCC blocks the GUI xctest host from reading /Volumes/S1
        // without an interactive prompt that can't render under
        // xcodebuild — same trap as Phase 4's heavy parity tests.
        // Require explicit opt-in via env var or a SwiftPM-style host.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_RUN_TCC_TESTS"] == "1"
                || Bundle(for: Self.self).bundleURL.path.contains(".build/"),
            "Skipping /Volumes/S1-backed smoke test under xcodebuild "
                + "(macOS TCC). Run `swift test --filter "
                + "Qwen3ASREngineSmokeTests` or set YOOZ_RUN_TCC_TESTS=1 "
                + "after granting Full Disk Access to the test host."
        )

        let configURL = Self.checkpointDir.appendingPathComponent("config.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: configURL.path),
            "Qwen3-ASR checkpoint not mounted at \(Self.checkpointDir.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalAudio.path),
            "Canonical audio not at \(Self.canonicalAudio.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.canonicalReference.path),
            "Phase 4 reference not at \(Self.canonicalReference.path)"
        )

        let referenceData = try Data(contentsOf: Self.canonicalReference)
        let reference = try JSONDecoder().decode(
            CanonicalReference.self, from: referenceData
        )
        let pcm = try Self.loadCanonicalPCM()

        // Point the engine at the canonical checkpoint via env-var
        // override so we don't need to copy 3.5 GB of weights for the
        // test.
        setenv("YOOZ_QWEN3_ASR_DIR", Self.checkpointDir.path, 1)
        defer { unsetenv("YOOZ_QWEN3_ASR_DIR") }

        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(.parakeet)
            }
        }

        // Reserve a fresh port for this boot — see yooz-engine#122.
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        defer {
            Task { @MainActor in await server.stop() }
        }

        let baseURL = URL(
            string: "http://\(EngineConfig.host):\(EngineConfig.port)"
        )!

        // 1) Load model (qwen3 backend, allow_fetch=false because we
        // pre-populated YOOZ_QWEN3_ASR_DIR).
        struct LoadBody: Encodable {
            let language: String
            let allowFetch: Bool
        }
        var loadReq = URLRequest(
            url: baseURL.appendingPathComponent("/v1/stt/load")
        )
        loadReq.httpMethod = "POST"
        loadReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        loadReq.httpBody = try JSONEncoder().encode(
            LoadBody(language: "en", allowFetch: false)
        )
        // Long timeout: the model takes ~1 s to load on M-series.
        loadReq.timeoutInterval = 120
        let (loadBody, loadResp) = try await URLSession.shared.data(
            for: loadReq
        )
        XCTAssertEqual(
            (loadResp as? HTTPURLResponse)?.statusCode, 200,
            "Load failed: \(String(data: loadBody, encoding: .utf8) ?? "")"
        )

        // 2) Batch transcription.
        struct BatchBody: Encodable {
            let samples: [Float]
            let language: String
        }
        var batchReq = URLRequest(
            url: baseURL.appendingPathComponent("/v1/stt/batch")
        )
        batchReq.httpMethod = "POST"
        batchReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        batchReq.httpBody = try JSONEncoder().encode(
            BatchBody(samples: pcm, language: "en")
        )
        batchReq.timeoutInterval = 60
        let (batchBody, batchResp) = try await URLSession.shared.data(
            for: batchReq
        )
        XCTAssertEqual((batchResp as? HTTPURLResponse)?.statusCode, 200)
        let decoded = try JSONDecoder().decode(
            BatchSTTResponse.self, from: batchBody
        )
        XCTAssertEqual(
            decoded.text, reference.transcriptionText,
            "Qwen3 batch transcription diverged from Phase 4 reference"
        )
        XCTAssertEqual(decoded.language, "en")
    }
}
