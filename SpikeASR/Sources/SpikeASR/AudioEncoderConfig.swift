import Foundation

/// Configuration for the Qwen3-ASR audio encoder, lifted directly
/// from `thinker_config.audio_config` in the HuggingFace
/// `mlx-community/Qwen3-ASR-1.7B-8bit` config.json.
///
/// All defaults match the official 1.7B checkpoint. Exposed as a
/// `Codable` struct so tests can round-trip a real config.json.
public struct AudioEncoderConfig: Codable, Sendable, Equatable {
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

    /// Frequency dimension after the three stride-2 convolutions, i.e.
    /// `((((numMelBins + 1) // 2) + 1) // 2 + 1) // 2`. Matches the
    /// Python `freq_after_conv` computation exactly. For the default
    /// `numMelBins=128` this is 16.
    public var freqAfterConv: Int {
        var f = numMelBins
        for _ in 0..<3 {
            f = (f + 1) / 2
        }
        return f
    }
}
