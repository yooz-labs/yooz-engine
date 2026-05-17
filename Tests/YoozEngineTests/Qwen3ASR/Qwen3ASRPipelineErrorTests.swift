// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

#if canImport(Qwen3ASRMelFrontend)
import Qwen3ASRMelFrontend
#endif

#if canImport(YoozEngine)
@testable import STTModule
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Phase 4 — typed-error tests for the decoder bridge.
///
/// Mirrors the pattern Phase 3 established for `Qwen3SafetensorsLoader`:
/// every error path the engine HTTP layer might encounter funnels
/// through `Qwen3ASRError` so the API can map it to a typed response
/// without parsing string messages.
///
/// These tests need a real `Tokenizer` instance because the
/// `Qwen3ASRPipeline` initializer takes one — the typed error paths
/// trip before tokenization runs but we still must satisfy the type
/// system. We borrow the tokenizer from the published checkpoint via
/// `AutoTokenizer.from(modelFolder:)`. When the checkpoint isn't
/// mounted (CI), the suite explicitly skips.
final class Qwen3ASRPipelineErrorTests: XCTestCase {

    private static var checkpointDir: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private actor TokCache {
        var tokenizer: (any Tokenizers.Tokenizer)?
        func get(
            _ build: () async throws -> any Tokenizers.Tokenizer
        ) async throws -> any Tokenizers.Tokenizer {
            if let t = tokenizer { return t }
            let t = try await build()
            tokenizer = t
            return t
        }
    }
    private static let tokCache = TokCache()

    private static func loadTokenizer() async throws -> any Tokenizers.Tokenizer {
        try await tokCache.get {
            try await AutoTokenizer.from(modelFolder: checkpointDir)
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        // macOS TCC blocks the GUI xctest host from reading
        // /Volumes/S1 without an interactive prompt that doesn't
        // render under xcodebuild. Phase 4 historically ran these
        // tests via swift test only; gate behind an env-var or a
        // SwiftPM-style host bundle path.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_RUN_TCC_TESTS"] == "1"
                || Bundle(for: Self.self).bundleURL.path.contains(".build/"),
            "Skipping /Volumes/S1-backed Qwen3 error test under "
                + "xcodebuild (macOS TCC). Run via swift test or set "
                + "YOOZ_RUN_TCC_TESTS=1."
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.checkpointDir.appendingPathComponent(
                    "tokenizer.json"
                ).path
            ),
            "Tokenizer not available; error tests need the published "
                + "checkpoint to satisfy the Qwen3ASRPipeline initializer"
        )
    }

    private func smallTextConfig() -> Qwen3ASRTextConfig {
        Qwen3ASRTextConfig(
            vocabSize: 128, hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            maxPositionEmbeddings: 256,
            rmsNormEps: 1e-6,
            ropeTheta: 10_000.0,
            tieWordEmbeddings: true,
            attentionBias: false
        )
    }

    private func smallAudioConfig() -> Qwen3ASRConfig {
        Qwen3ASRConfig(
            numMelBins: 16,
            encoderLayers: 1,
            encoderAttentionHeads: 4,
            encoderFFNDim: 64,
            dModel: 32,
            downsampleHiddenSize: 16,
            outputDim: 32,
            maxSourcePositions: 32,
            nWindow: 25,
            nWindowInfer: 200,
            convChunkSize: 50
        )
    }

    /// A loadWeights call against a directory missing both
    /// safetensors and config must surface `fileNotFound`, not a
    /// generic IO error.
    func testLoadWeightsRejectsMissingFile() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cfg = Qwen3ASRFullConfig(
            audio: smallAudioConfig(),
            text: smallTextConfig(),
            audioTokenId: 0,
            audioStartTokenId: 1,
            audioEndTokenId: 2,
            supportLanguages: [],
            quantBits: nil,
            quantGroupSize: nil
        )
        let pipeline = Qwen3ASRPipeline(
            config: cfg,
            modelDirectory: FileManager.default.temporaryDirectory,
            melFrontend: MelFrontend(config: .qwen3ASR),
            tokenizer: tokenizer
        )
        let bogus = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).safetensors")
        XCTAssertThrowsError(
            try pipeline.loadWeights(from: bogus)
        ) { error in
            guard let e = error as? Qwen3ASRError else {
                return XCTFail("expected Qwen3ASRError, got \(error)")
            }
            if case .fileNotFound = e {
                return
            }
            XCTFail("expected fileNotFound, got \(e)")
        }
    }

    /// A safetensors file missing every `audio_tower.*` key (e.g.
    /// the pure text decoder slice) must surface `noAudioTowerWeights`.
    func testLoadWeightsRejectsTextOnlyCheckpoint() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cfg = Qwen3ASRFullConfig(
            audio: smallAudioConfig(),
            text: smallTextConfig(),
            audioTokenId: 0,
            audioStartTokenId: 1,
            audioEndTokenId: 2,
            supportLanguages: [],
            quantBits: nil,
            quantGroupSize: nil
        )
        let pipeline = Qwen3ASRPipeline(
            config: cfg,
            modelDirectory: FileManager.default.temporaryDirectory,
            melFrontend: MelFrontend(config: .qwen3ASR),
            tokenizer: tokenizer
        )

        // Build an in-memory safetensors file with only text-decoder
        // keys; the loader must trip the `noAudioTowerWeights`
        // branch.
        let dummy = MLXArray.zeros([1, 1])
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("text-only-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: ["model.embed_tokens.weight": dummy], url: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(
            try pipeline.loadWeights(from: url)
        ) { error in
            guard let e = error as? Qwen3ASRError else {
                return XCTFail("expected Qwen3ASRError, got \(error)")
            }
            if case .noAudioTowerWeights = e {
                return
            }
            XCTFail("expected noAudioTowerWeights, got \(e)")
        }
    }

    /// Empty PCM input must throw `Qwen3ASRError.invalidInput`
    /// rather than crashing the engine.
    func testTranscribeRejectsEmptyPCM() async throws {
        let tokenizer = try await Self.loadTokenizer()
        let cfg = Qwen3ASRFullConfig(
            audio: smallAudioConfig(),
            text: smallTextConfig(),
            audioTokenId: 0,
            audioStartTokenId: 1,
            audioEndTokenId: 2,
            supportLanguages: [],
            quantBits: nil,
            quantGroupSize: nil
        )
        let pipeline = Qwen3ASRPipeline(
            config: cfg,
            modelDirectory: FileManager.default.temporaryDirectory,
            melFrontend: MelFrontend(config: .qwen3ASR),
            tokenizer: tokenizer
        )

        XCTAssertThrowsError(
            try pipeline.transcribe(pcm: [], language: "English")
        ) { error in
            guard let e = error as? Qwen3ASRError else {
                return XCTFail("expected Qwen3ASRError, got \(error)")
            }
            if case .invalidInput = e {
                return
            }
            XCTFail("expected invalidInput, got \(e)")
        }
    }
}
