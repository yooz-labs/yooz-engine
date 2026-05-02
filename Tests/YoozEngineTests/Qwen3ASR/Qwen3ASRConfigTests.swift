// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

#if canImport(YoozEngine)
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Pure-config unit tests. No MLX, no S1, no weights.
final class Qwen3ASRConfigTests: XCTestCase {

    func testDefaultsMatchHuggingFaceCheckpoint() {
        let cfg = Qwen3ASRConfig()
        XCTAssertEqual(cfg.numMelBins, 128)
        XCTAssertEqual(cfg.encoderLayers, 24)
        XCTAssertEqual(cfg.encoderAttentionHeads, 16)
        XCTAssertEqual(cfg.encoderFFNDim, 4096)
        XCTAssertEqual(cfg.dModel, 1024)
        XCTAssertEqual(cfg.downsampleHiddenSize, 480)
        XCTAssertEqual(cfg.outputDim, 2048)
        XCTAssertEqual(cfg.maxSourcePositions, 1500)
        XCTAssertEqual(cfg.nWindow, 50)
        XCTAssertEqual(cfg.nWindowInfer, 800)
        XCTAssertEqual(cfg.convChunkSize, 500)
        XCTAssertFalse(cfg.scaleEmbedding)
        XCTAssertEqual(cfg.activationFunction, "gelu")
    }

    func testFreqAfterConvMatchesReference() {
        // Ref: ((((128 + 1) // 2) + 1) // 2 + 1) // 2 == 16.
        XCTAssertEqual(Qwen3ASRConfig().freqAfterConv, 16)
        XCTAssertEqual(
            Qwen3ASRConfig(numMelBins: 80).freqAfterConv, 10
        )
    }

    func testConfigCodableRoundTripsHuggingFaceJSON() throws {
        // The 1.7B-8bit checkpoint stores the audio config with
        // snake_case keys. Confirm the Codable mapping matches.
        let json = """
            {
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
              "activation_function": "gelu"
            }
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(
            Qwen3ASRConfig.self, from: json
        )
        XCTAssertEqual(cfg, Qwen3ASRConfig())
    }

    // MARK: - validate()

    func testValidateAcceptsDefaults() throws {
        XCTAssertNoThrow(try Qwen3ASRConfig().validate())
    }

    func testValidateRejectsNonDivisibleHeadDim() {
        // Config struct now uses `let` fields, so the test composes
        // the invalid config via the explicit initializer rather
        // than mutating after construction.
        let cfg = Qwen3ASRConfig(encoderAttentionHeads: 17)
        XCTAssertThrowsError(try cfg.validate()) { error in
            guard case Qwen3ASRError.invalidConfig(let msg) = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("divisible"))
        }
    }

    func testValidateRejectsZeroLayers() {
        let cfg = Qwen3ASRConfig(encoderLayers: 0)
        XCTAssertThrowsError(try cfg.validate()) { error in
            guard case Qwen3ASRError.invalidConfig = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
        }
    }

    func testValidateRejectsNonMultipleNWindowInfer() {
        // Not a multiple of nWindow*2 == 100.
        let cfg = Qwen3ASRConfig(nWindowInfer: 7)
        XCTAssertThrowsError(try cfg.validate()) { error in
            guard case Qwen3ASRError.invalidConfig(let msg) = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("nWindowInfer"))
        }
    }

    func testValidateRejectsZeroMelBins() {
        let cfg = Qwen3ASRConfig(numMelBins: 0)
        XCTAssertThrowsError(try cfg.validate()) { error in
            guard case Qwen3ASRError.invalidConfig = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
        }
    }

    // Confirms the new `init(from:)` runs `validate()` so an
    // unvalidated instance cannot escape decoding.
    func testDecodeRejectsBadConfig() {
        let badJSON = """
        {"d_model": 1024, "encoder_attention_heads": 17}
        """.data(using: .utf8)!
        XCTAssertThrowsError(
            try JSONDecoder().decode(Qwen3ASRConfig.self, from: badJSON)
        ) { error in
            guard case Qwen3ASRError.invalidConfig = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
        }
    }
}
