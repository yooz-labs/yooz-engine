// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

#if canImport(YoozEngine)
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Phase 4 — pure-Swift structural tests for the Qwen3-ASR text
/// decoder. Run without external artifacts so they are part of
/// every `xcodebuild test` and `swift test` invocation.
final class Qwen3ASRTextDecoderStructuralTests: XCTestCase {

    private func smallConfig() -> Qwen3ASRTextConfig {
        Qwen3ASRTextConfig(
            vocabSize: 128,
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 2,
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

    func testTokenForwardProducesLogitsOfCorrectShape() {
        let cfg = smallConfig()
        let decoder = Qwen3ASRTextDecoder(cfg)
        let inputs = MLXArray(
            Array(repeating: Int32(0), count: 5)
        ).reshaped(1, 5)
        let logits = decoder(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.dim(0), 1)
        XCTAssertEqual(logits.dim(1), 5)
        XCTAssertEqual(logits.dim(2), cfg.vocabSize)
        XCTAssertFalse(MLX.any(MLX.isNaN(logits)).item(Bool.self))
        XCTAssertFalse(MLX.any(MLX.isInf(logits)).item(Bool.self))
    }

    func testEmbeddingsForwardMatchesTokenForwardOnSamePrompt() {
        let cfg = smallConfig()
        let decoder = Qwen3ASRTextDecoder(cfg)
        let inputs = MLXArray(
            (0..<7).map { Int32($0 % cfg.vocabSize) }
        ).reshaped(1, 7)

        // Embedding-only forward must match token forward when the
        // embeddings come straight from the model's own embed table.
        let embeds = decoder.model.embedTokens(inputs)
        let viaTokens = decoder(inputs, cache: nil)
        let viaEmbeds = decoder.callAsEmbeddings(embeds, cache: nil)
        eval(viaTokens, viaEmbeds)

        let drift = MLX.abs(viaTokens - viaEmbeds).max().item(Float.self)
        XCTAssertEqual(
            drift, 0.0,
            "callAsEmbeddings(embedTokens(x)) must equal callAsFunction(x)"
        )
    }

    func testKVCacheAdvancesBetweenSteps() {
        let cfg = smallConfig()
        let decoder = Qwen3ASRTextDecoder(cfg)

        let cache = decoder.newCache(parameters: nil)
        XCTAssertEqual(cache.count, cfg.numHiddenLayers)
        for c in cache {
            XCTAssertEqual(c.offset, 0)
        }

        let prompt = MLXArray(
            (0..<5).map { Int32($0) }
        ).reshaped(1, 5)
        _ = decoder(prompt, cache: cache)
        eval(cache.first!.innerState())
        for c in cache {
            XCTAssertEqual(c.offset, 5)
        }

        // One-token step should advance the cache by exactly one.
        let nextTok = MLXArray([Int32(1)]).reshaped(1, 1)
        _ = decoder(nextTok, cache: cache)
        eval(cache.first!.innerState())
        for c in cache {
            XCTAssertEqual(c.offset, 6)
        }
    }

    func testInvalidConfigIsRejected() {
        var cfg = smallConfig()
        cfg.numAttentionHeads = 5  // not divisible by numKeyValueHeads (2)
        XCTAssertThrowsError(try cfg.validate())

        var cfg2 = smallConfig()
        cfg2.headDim = 0
        XCTAssertThrowsError(try cfg2.validate())

        var cfg3 = smallConfig()
        cfg3.rmsNormEps = 0
        XCTAssertThrowsError(try cfg3.validate())
    }

    func testFullConfigDecodesCanonicalJSON() throws {
        let json: [String: Any] = [
            "thinker_config": [
                "audio_config": [
                    "num_mel_bins": 128,
                    "encoder_layers": 24,
                    "encoder_attention_heads": 16,
                    "encoder_ffn_dim": 4096,
                    "d_model": 1024,
                    "downsample_hidden_size": 480,
                    "output_dim": 2048,
                    "max_source_positions": 1500,
                    "n_window": 50,
                    "n_window_infer": 800,
                    "conv_chunksize": 500,
                    "scale_embedding": false,
                    "activation_function": "gelu",
                ] as [String: Any],
                "text_config": [
                    "vocab_size": 151_936,
                    "hidden_size": 2_048,
                    "intermediate_size": 6_144,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "max_position_embeddings": 65_536,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1_000_000,
                    "tie_word_embeddings": true,
                ] as [String: Any],
                "audio_token_id": 151_676,
                "audio_start_token_id": 151_669,
                "audio_end_token_id": 151_670,
            ] as [String: Any],
            "support_languages": ["English", "Persian", "Arabic"],
            "quantization": ["bits": 8, "group_size": 64] as [String: Any],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "qwen3asr-config-\(UUID().uuidString).json"
            )
        try data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let parsed = try Qwen3ASRFullConfig.load(from: tmp)
        XCTAssertEqual(parsed.text.numHiddenLayers, 28)
        XCTAssertEqual(parsed.text.numKeyValueHeads, 8)
        XCTAssertEqual(parsed.audioTokenId, 151_676)
        XCTAssertEqual(parsed.quantBits, 8)
        XCTAssertEqual(parsed.quantGroupSize, 64)
        XCTAssertTrue(parsed.supportLanguages.contains("Persian"))
    }
}
