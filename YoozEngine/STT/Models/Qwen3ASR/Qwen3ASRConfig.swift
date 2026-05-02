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
    public let numMelBins: Int
    public let encoderLayers: Int
    public let encoderAttentionHeads: Int
    public let encoderFFNDim: Int
    public let dModel: Int
    public let downsampleHiddenSize: Int
    public let outputDim: Int
    public let maxSourcePositions: Int
    public let nWindow: Int
    public let nWindowInfer: Int
    public let convChunkSize: Int
    public let scaleEmbedding: Bool
    public let activationFunction: String

    /// Memberwise initializer. Non-throwing for source-compat with
    /// the wide call-site set (encoder constructors, parity tests,
    /// derived configs). Callers that need the validated invariant
    /// should call `validated(...)` instead — that throws on the
    /// same bad fields `validate()` rejects.
    ///
    /// This memberwise init is deliberately permissive so tests can
    /// construct invalid configs to exercise `validate()` itself
    /// (`testValidateRejectsNonDivisibleHeadDim` and friends). The
    /// `init(from:)` decode path always runs `validate()`, so
    /// configs that crossed a JSON boundary are checked.
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

    /// Throwing factory: same fields as the memberwise init, but
    /// runs `validate()` before returning. Use this from production
    /// call sites where receiving a malformed config should produce
    /// a typed error instead of an instance that crashes the
    /// encoder later. The decode path (`init(from:)`) already calls
    /// `validate()`, so this factory is only needed for direct
    /// construction (tests, programmatic config building).
    public static func validated(
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
    ) throws -> Qwen3ASRConfig {
        let cfg = Qwen3ASRConfig(
            numMelBins: numMelBins,
            encoderLayers: encoderLayers,
            encoderAttentionHeads: encoderAttentionHeads,
            encoderFFNDim: encoderFFNDim,
            dModel: dModel,
            downsampleHiddenSize: downsampleHiddenSize,
            outputDim: outputDim,
            maxSourcePositions: maxSourcePositions,
            nWindow: nWindow,
            nWindowInfer: nWindowInfer,
            convChunkSize: convChunkSize,
            scaleEmbedding: scaleEmbedding,
            activationFunction: activationFunction
        )
        try cfg.validate()
        return cfg
    }

    /// Codable entry point — calls `validate()` so an unvalidated
    /// instance cannot exist after a successful decode.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.numMelBins =
            try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        self.encoderLayers =
            try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 24
        self.encoderAttentionHeads =
            try c.decodeIfPresent(
                Int.self, forKey: .encoderAttentionHeads
            ) ?? 16
        self.encoderFFNDim =
            try c.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? 4096
        self.dModel =
            try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        self.downsampleHiddenSize =
            try c.decodeIfPresent(
                Int.self, forKey: .downsampleHiddenSize
            ) ?? 480
        self.outputDim =
            try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 2048
        self.maxSourcePositions =
            try c.decodeIfPresent(
                Int.self, forKey: .maxSourcePositions
            ) ?? 1500
        self.nWindow =
            try c.decodeIfPresent(Int.self, forKey: .nWindow) ?? 50
        self.nWindowInfer =
            try c.decodeIfPresent(Int.self, forKey: .nWindowInfer) ?? 800
        self.convChunkSize =
            try c.decodeIfPresent(Int.self, forKey: .convChunkSize) ?? 500
        self.scaleEmbedding =
            try c.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
        self.activationFunction =
            try c.decodeIfPresent(
                String.self, forKey: .activationFunction
            ) ?? "gelu"
        try self.validate()
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
