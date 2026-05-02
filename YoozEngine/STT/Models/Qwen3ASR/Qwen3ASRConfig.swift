// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Configuration for the Qwen3-ASR audio encoder.
///
/// Lifted directly from `thinker_config.audio_config` in the
/// HuggingFace `mlx-community/Qwen3-ASR-1.7B-8bit` config.json. All
/// defaults match the official 1.7B checkpoint.
///
/// `Codable` so a real `config.json` can round-trip through this
/// type. `Sendable` + `Equatable` so it can be passed across actor
/// boundaries inside the engine without copying caveats.
public struct Qwen3ASRConfig: Codable, Sendable, Equatable {
    public var numMelBins: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFFNDim: Int
    public var dModel: Int
    public var downsampleHiddenSize: Int
    public var outputDim: Int
    public var maxSourcePositions: Int
    public var nWindow: Int
    public var nWindowInfer: Int
    public var convChunkSize: Int
    public var scaleEmbedding: Bool
    public var activationFunction: String

    public init(
        numMelBins: Int = 128,
        encoderLayers: Int = 24,
        encoderAttentionHeads: Int = 16,
        encoderFFNDim: Int = 4096,
        dModel: Int = 1024,
        downsampleHiddenSize: Int = 480,
        outputDim: Int = 2048,
        maxSourcePositions: Int = 1500,
        nWindow: Int = 50,
        nWindowInfer: Int = 800,
        convChunkSize: Int = 500,
        scaleEmbedding: Bool = false,
        activationFunction: String = "gelu"
    ) {
        self.numMelBins = numMelBins
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFFNDim = encoderFFNDim
        self.dModel = dModel
        self.downsampleHiddenSize = downsampleHiddenSize
        self.outputDim = outputDim
        self.maxSourcePositions = maxSourcePositions
        self.nWindow = nWindow
        self.nWindowInfer = nWindowInfer
        self.convChunkSize = convChunkSize
        self.scaleEmbedding = scaleEmbedding
        self.activationFunction = activationFunction
    }

    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case dModel = "d_model"
        case downsampleHiddenSize = "downsample_hidden_size"
        case outputDim = "output_dim"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case nWindowInfer = "n_window_infer"
        case convChunkSize = "conv_chunksize"
        case scaleEmbedding = "scale_embedding"
        case activationFunction = "activation_function"
    }

    /// Frequency dimension after the three stride-2, padding-1
    /// `Conv2d` layers. Matches the Python reference's
    /// `freq_after_conv` computation:
    /// `((((numMelBins + 1) // 2) + 1) // 2 + 1) // 2`.
    /// For `numMelBins = 128` this is 16.
    public var freqAfterConv: Int {
        var f = numMelBins
        for _ in 0..<3 {
            f = (f + 1) / 2
        }
        return f
    }

    /// Validate semantic invariants the encoder relies on. Throws a
    /// typed `Qwen3ASRError.invalidConfig` describing the offending
    /// field — never crashes a long-running engine on a config typo.
    public func validate() throws {
        if dModel <= 0 || encoderAttentionHeads <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "dModel (\(dModel)) and encoderAttentionHeads "
                    + "(\(encoderAttentionHeads)) must be positive"
            )
        }
        if dModel % encoderAttentionHeads != 0 {
            throw Qwen3ASRError.invalidConfig(
                "dModel (\(dModel)) must be divisible by "
                    + "encoderAttentionHeads (\(encoderAttentionHeads))"
            )
        }
        if encoderLayers <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "encoderLayers must be > 0, got \(encoderLayers)"
            )
        }
        if encoderFFNDim <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "encoderFFNDim must be > 0, got \(encoderFFNDim)"
            )
        }
        if numMelBins <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "numMelBins must be > 0, got \(numMelBins)"
            )
        }
        if maxSourcePositions <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "maxSourcePositions must be > 0, got "
                    + "\(maxSourcePositions)"
            )
        }
        if nWindow <= 0 || nWindowInfer <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "nWindow (\(nWindow)) and nWindowInfer "
                    + "(\(nWindowInfer)) must be positive"
            )
        }
        if (nWindow * 2) == 0 || nWindowInfer % (nWindow * 2) != 0 {
            throw Qwen3ASRError.invalidConfig(
                "nWindowInfer (\(nWindowInfer)) must be a positive "
                    + "multiple of (nWindow * 2 = \(nWindow * 2))"
            )
        }
        if downsampleHiddenSize <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "downsampleHiddenSize must be > 0, got "
                    + "\(downsampleHiddenSize)"
            )
        }
        if outputDim <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "outputDim must be > 0, got \(outputDim)"
            )
        }
    }
}
