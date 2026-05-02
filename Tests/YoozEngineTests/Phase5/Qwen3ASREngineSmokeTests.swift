// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import Hummingbird
import HummingbirdTesting
import XCTest

@testable import YoozEngine

/// Phase 5 — end-to-end smoke test for `/v1/stt/batch` with the
/// qwen3 backend selected. Loads the canonical Qwen3-ASR checkpoint
/// from `/Volumes/S1` (skips when not mounted) and asserts the HTTP
/// route returns the same transcription text as the Phase 4
/// canonical reference.
///
/// macOS TCC blocks the GUI test host from reading `/Volumes/S1`
/// without an interactive prompt. This suite is therefore built to
/// run from `swift test` directly OR via `xcodebuild test` after the
/// user has accepted the file-access prompt; otherwise it skips.
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
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent("config.json").path
            ),
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

        // Point the engine at the canonical checkpoint via the env-var
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

        let server = APIServer()
        let router = server.makeTestRouter()
        let app = Application(router: router)

        try await app.test(.router) { client in
            // Load the model first (populates the actor with the
            // checkpoint).
            struct LoadBody: Encodable {
                let language: String
                let allowFetch: Bool
            }
            let loadBody = try JSONEncoder().encode(
                LoadBody(language: "en", allowFetch: false)
            )
            try await client.execute(
                uri: "/v1/stt/load",
                method: .post,
                body: ByteBuffer(data: loadBody)
            ) { res in
                XCTAssertEqual(res.status, .ok, "Load failed: \(String(buffer: res.body))")
            }

            // Run batch transcription.
            struct BatchBody: Encodable {
                let samples: [Float]
                let language: String
            }
            let batchBody = try JSONEncoder().encode(
                BatchBody(samples: pcm, language: "en")
            )
            try await client.execute(
                uri: "/v1/stt/batch",
                method: .post,
                body: ByteBuffer(data: batchBody)
            ) { res in
                XCTAssertEqual(res.status, .ok)
                let decoded = try JSONDecoder().decode(
                    BatchSTTResponse.self, from: Data(buffer: res.body)
                )
                XCTAssertEqual(
                    decoded.text,
                    reference.transcriptionText,
                    "Qwen3 batch transcription diverged from Phase 4 reference"
                )
                XCTAssertEqual(decoded.language, "en")
            }
        }
    }
}
