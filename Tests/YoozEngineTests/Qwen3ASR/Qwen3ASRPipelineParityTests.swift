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

/// Phase 4 — end-to-end Qwen3-ASR parity against the Python reference.
///
/// Heavy tests that depend on the published 1.7B-8bit checkpoint plus
/// the canonical reference dump under
/// `/Volumes/S1/yooz/research/issue-46/phase4-bridge`. macOS TCC
/// blocks the xctest-host (GUI app) from reading `/Volumes/S1`
/// without an interactive prompt, so all the heavy tests run via
/// `swift test` from SwiftPM and skip explicitly when the artifacts
/// are absent (CI / Xcode flow). Override the search path with
/// `YOOZ_PHASE4_ARTIFACTS` for an out-of-volume copy.
final class Qwen3ASRPipelineParityTests: XCTestCase {

    // MARK: - Locations

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

    private static var canonicalReferenceURL: URL {
        phase4Root.appendingPathComponent(
            "reference/canonical_transcription.json"
        )
    }

    private static var decoderInputsURL: URL {
        phase4Root.appendingPathComponent(
            "artifacts/decoder_parity_inputs.safetensors"
        )
    }

    private static var decoderOutputsURL: URL {
        phase4Root.appendingPathComponent(
            "artifacts/decoder_parity_outputs.safetensors"
        )
    }

    private static var canonicalAudioURL: URL {
        URL(fileURLWithPath: "/Volumes/S1/yooz/stt-test-data/english/test_001.wav")
    }

    // MARK: - Helpers

    private struct CanonicalReference: Decodable {
        let numAudioTokens: Int
        let promptTokenIds: [Int]
        let audioPadPositions: [Int]
        let generatedTokenIds: [Int]
        let transcriptionText: String

        enum CodingKeys: String, CodingKey {
            case numAudioTokens = "num_audio_tokens"
            case promptTokenIds = "prompt_token_ids"
            case audioPadPositions = "audio_pad_positions"
            case generatedTokenIds = "generated_token_ids"
            case transcriptionText = "transcription_text"
        }
    }

    /// Cache the pipeline across tests so the multi-second model
    /// load runs once. XCTest invokes test methods serially, so a
    /// plain optional with a serializing actor is enough — no lock
    /// required (the explicit serial actor avoids the warning we'd
    /// otherwise get from `nonisolated(unsafe)` mutable state).
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

    private static func loadCanonicalReference() throws -> CanonicalReference {
        let data = try Data(contentsOf: canonicalReferenceURL)
        let decoder = JSONDecoder()
        return try decoder.decode(CanonicalReference.self, from: data)
    }

    private static func loadCanonicalPCM() throws -> [Float] {
        // Light-weight wav decoder for 16-bit mono 16 kHz; we don't
        // pull in AVFoundation here so the parity test stays portable.
        let data = try Data(contentsOf: canonicalAudioURL)
        // RIFF parsing — find "data" chunk.
        guard data.count > 44 else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "wav too short"]
            )
        }
        // Locate the 'data' chunk header (4-byte ASCII).
        var idx = 12
        while idx + 8 <= data.count {
            let chunkId = data.subdata(in: idx..<(idx + 4))
            let chunkSize = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt32 in
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
            userInfo: [
                NSLocalizedDescriptionKey: "no 'data' chunk in wav"
            ]
        )
    }

    private static func skipUnlessArtifactsAvailable() throws {
        try Qwen3ASRTestEnvironment.skipUnlessSafeForTCC()
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: canonicalReferenceURL.path
            ),
            "Phase 4 reference not available at \(canonicalReferenceURL.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: canonicalAudioURL.path),
            "Canonical wav not available at \(canonicalAudioURL.path)"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: checkpointDir.appendingPathComponent("config.json").path
            ),
            "Qwen3-ASR checkpoint not available at \(checkpointDir.path)"
        )
    }

    // MARK: - Tests

    /// 1. Prompt template parity — Swift's tokenizer must produce the
    /// exact same prompt token sequence as the Python reference.
    /// Independent of weights, so it runs even when the safetensors
    /// is absent (only the tokenizer + reference JSON are needed).
    func testPromptTemplateMatchesReference() async throws {
        try Self.skipUnlessArtifactsAvailable()

        let reference = try Self.loadCanonicalReference()
        let pipeline = try await Self.loadPipeline()

        let promptIDs = try pipeline.buildPromptTokenIDs(
            numAudioTokens: reference.numAudioTokens,
            language: nil
        )

        XCTAssertEqual(
            promptIDs.count, reference.promptTokenIds.count,
            "Prompt token count diverged from reference"
        )
        XCTAssertEqual(
            promptIDs, reference.promptTokenIds,
            "Prompt token IDs diverged from reference"
        )

        // Cross-check audio_pad positions against the reference's
        // explicit list — guards against tokenizer revisions that
        // might shift the audio_pad token id.
        let padIndices = promptIDs.enumerated().compactMap {
            (idx, tok) -> Int? in
            tok == pipeline.config.audioTokenId ? idx : nil
        }
        XCTAssertEqual(padIndices, reference.audioPadPositions)
    }

    /// 2. Decoder prefill parity — given the same audio_features and
    /// prompt as the Python reference, the first generated token must
    /// be `language` (id 11528). We assert by transcribing with
    /// `maxNewTokens: 1` and walking the canonical reference's
    /// generated_token_ids stream: token #0 (= 11528) is the
    /// language tag, token #1 (= 6364) is "English", token #2
    /// (= 151704) is `<asr_text>`. The pipeline strips the language
    /// preamble, so a `maxNewTokens: 1` run yields zero "content"
    /// tokens and `result.numAudioTokens` must match the reference
    /// (cross-checks the audio_tower path was actually exercised).
    func testDecoderPrefillArgmaxMatchesReference() async throws {
        try Self.skipUnlessArtifactsAvailable()

        let reference = try Self.loadCanonicalReference()
        let pipeline = try await Self.loadPipeline()
        let pcm = try Self.loadCanonicalPCM()

        let result = try pipeline.transcribe(
            pcm: pcm, language: nil, maxNewTokens: 1
        )

        XCTAssertEqual(
            result.numAudioTokens, reference.numAudioTokens,
            "audio_tower output token count diverged from reference"
        )
        // With maxNewTokens=1 the model emits exactly one token
        // before the loop exits. The reference's first generated
        // token is 11528 ("language"); the language preamble parser
        // doesn't trigger because `<asr_text>` was never sampled,
        // so generatedTokens contains the raw stream and we can
        // assert directly on the first token.
        XCTAssertEqual(
            result.generatedTokens.first, reference.generatedTokenIds.first,
            "First sampled token diverged from reference (expected the "
                + "'language' tag id 11528)"
        )
    }

    /// 3. End-to-end transcription parity — Swift output must match
    /// the Python reference exactly (greedy decode is deterministic).
    func testEndToEndTranscriptionMatchesReference() async throws {
        try Self.skipUnlessArtifactsAvailable()

        let reference = try Self.loadCanonicalReference()
        let pipeline = try await Self.loadPipeline()
        let pcm = try Self.loadCanonicalPCM()

        let result = try pipeline.transcribe(
            pcm: pcm, language: nil, maxNewTokens: 256
        )

        XCTAssertEqual(
            result.text, reference.transcriptionText,
            "Swift transcription diverged from Python reference"
        )
        XCTAssertEqual(result.language, "English")
    }
}
