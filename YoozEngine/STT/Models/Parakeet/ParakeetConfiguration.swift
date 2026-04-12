// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - Preprocessing Configuration

/// Audio preprocessing configuration matching Parakeet's PreprocessArgs
// MARK: - Audio Mode

/// Audio mode for speech recognition
public enum AudioMode: String, Codable, Sendable {
    case normal
    case whispered
}

// MARK: - Preprocessing Configuration

public struct PreprocessConfig: Codable, Sendable {
    public let sampleRate: Int
    public let normalize: String
    public let windowSize: Float
    public let windowStride: Float
    public let window: String
    public let features: Int
    public let nFft: Int
    public let dither: Float
    public let padTo: Int
    public let padValue: Float
    public let preemph: Float
    
    // Whisper mode support
    public var audioMode: AudioMode
    public var whisperPreemph: Float
    public var whisperSpectralTilt: Float

    public var winLength: Int {
        Int(windowSize * Float(sampleRate))
    }

    public var hopLength: Int {
        Int(windowStride * Float(sampleRate))
    }

    /// Active preemphasis coefficient based on audio mode
    public var activePreemph: Float {
        audioMode == .whispered ? whisperPreemph : preemph
    }

    /// Active spectral tilt compensation based on audio mode
    /// Applies linear ramp to log mel spectrogram to compensate for flatter spectral envelope
    public var activeSpectralTilt: Float {
        audioMode == .whispered ? whisperSpectralTilt : 0.0
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFft = "n_fft"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
        case audioMode = "audio_mode"
        case whisperPreemph = "whisper_preemph"
        case whisperSpectralTilt = "whisper_spectral_tilt"
    }

    public init(
        sampleRate: Int = 16000,
        normalize: String = "per_feature",
        windowSize: Float = 0.025,
        windowStride: Float = 0.01,
        window: String = "hann",
        features: Int = 80,
        nFft: Int = 512,
        dither: Float = 0.0,
        padTo: Int = 0,
        padValue: Float = 0.0,
        preemph: Float = 0.97,
        audioMode: AudioMode = .normal,
        whisperPreemph: Float = 0.99,
        whisperSpectralTilt: Float = 0.4
    ) {
        self.sampleRate = sampleRate
        self.normalize = normalize
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.window = window
        self.features = features
        self.nFft = nFft
        self.dither = dither
        self.padTo = padTo
        self.padValue = padValue
        self.preemph = preemph
        self.audioMode = audioMode
        self.whisperPreemph = whisperPreemph
        self.whisperSpectralTilt = whisperSpectralTilt
    }
    
    /// Custom decoder to maintain backward compatibility
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        sampleRate = try container.decode(Int.self, forKey: .sampleRate)
        normalize = try container.decode(String.self, forKey: .normalize)
        windowSize = try container.decode(Float.self, forKey: .windowSize)
        windowStride = try container.decode(Float.self, forKey: .windowStride)
        window = try container.decode(String.self, forKey: .window)
        features = try container.decode(Int.self, forKey: .features)
        nFft = try container.decode(Int.self, forKey: .nFft)
        dither = try container.decode(Float.self, forKey: .dither)
        padTo = try container.decode(Int.self, forKey: .padTo)
        padValue = try container.decode(Float.self, forKey: .padValue)
        preemph = try container.decode(Float.self, forKey: .preemph)
        
        // Optional fields with defaults for backward compatibility
        audioMode = try container.decodeIfPresent(AudioMode.self, forKey: .audioMode) ?? .normal
        whisperPreemph = try container.decodeIfPresent(Float.self, forKey: .whisperPreemph) ?? 0.99
        whisperSpectralTilt = try container.decodeIfPresent(Float.self, forKey: .whisperSpectralTilt) ?? 0.4
    }
}

// MARK: - Conformer Encoder Configuration

/// Conformer encoder configuration
public struct ConformerConfig: Codable, Sendable {
    public let featIn: Int
    public let nLayers: Int
    public let dModel: Int
    public let nHeads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let selfAttentionModel: String
    public let subsampling: String
    public let convKernelSize: Int
    public let subsamplingConvChannels: Int
    public let posEmbMaxLen: Int
    public let causalDownsampling: Bool
    public let useBias: Bool
    public let xscaling: Bool
    public let subsamplingConvChunkingFactor: Int

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case selfAttentionModel = "self_attention_model"
        case subsampling
        case convKernelSize = "conv_kernel_size"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case posEmbMaxLen = "pos_emb_max_len"
        case causalDownsampling = "causal_downsampling"
        case useBias = "use_bias"
        case xscaling
        case subsamplingConvChunkingFactor = "subsampling_conv_chunking_factor"
    }

    public init(
        featIn: Int = 80,
        nLayers: Int = 17,
        dModel: Int = 512,
        nHeads: Int = 8,
        ffExpansionFactor: Int = 4,
        subsamplingFactor: Int = 4,
        selfAttentionModel: String = "rel_pos",
        subsampling: String = "dw_striding",
        convKernelSize: Int = 31,
        subsamplingConvChannels: Int = 512,
        posEmbMaxLen: Int = 5000,
        causalDownsampling: Bool = false,
        useBias: Bool = true,
        xscaling: Bool = false,
        subsamplingConvChunkingFactor: Int = 1
    ) {
        self.featIn = featIn
        self.nLayers = nLayers
        self.dModel = dModel
        self.nHeads = nHeads
        self.ffExpansionFactor = ffExpansionFactor
        self.subsamplingFactor = subsamplingFactor
        self.selfAttentionModel = selfAttentionModel
        self.subsampling = subsampling
        self.convKernelSize = convKernelSize
        self.subsamplingConvChannels = subsamplingConvChannels
        self.posEmbMaxLen = posEmbMaxLen
        self.causalDownsampling = causalDownsampling
        self.useBias = useBias
        self.xscaling = xscaling
        self.subsamplingConvChunkingFactor = subsamplingConvChunkingFactor
    }
}

// MARK: - Prediction Network Configuration

/// Prediction network (decoder) configuration
public struct PredictConfig: Codable, Sendable {
    public let vocabSize: Int
    public let predHidden: Int
    public let rnnHiddenSize: Int
    public let predRnnLayers: Int
    public let blankAsLastToken: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case predHidden = "pred_hidden"
        case rnnHiddenSize = "rnn_hidden_size"
        case predRnnLayers = "pred_rnn_layers"
        case blankAsLastToken = "blank_as_last_token"
    }

    public init(
        vocabSize: Int = 128,
        predHidden: Int = 640,
        rnnHiddenSize: Int = 640,
        predRnnLayers: Int = 2,
        blankAsLastToken: Bool = true
    ) {
        self.vocabSize = vocabSize
        self.predHidden = predHidden
        self.rnnHiddenSize = rnnHiddenSize
        self.predRnnLayers = predRnnLayers
        self.blankAsLastToken = blankAsLastToken
    }
}

// MARK: - Joint Network Configuration

/// Joint network configuration
public struct JointConfig: Codable, Sendable {
    public let encoderHidden: Int
    public let predHidden: Int
    public let jointHidden: Int
    public let numClasses: Int
    public let numExtraOutputs: Int
    public let activation: String
    public let vocabulary: [String]

    enum CodingKeys: String, CodingKey {
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
        case jointHidden = "joint_hidden"
        case numClasses = "num_classes"
        case numExtraOutputs = "num_extra_outputs"
        case activation
        case vocabulary
    }

    public init(
        encoderHidden: Int = 512,
        predHidden: Int = 640,
        jointHidden: Int = 640,
        numClasses: Int = 128,
        numExtraOutputs: Int = 5,
        activation: String = "relu",
        vocabulary: [String] = []
    ) {
        self.encoderHidden = encoderHidden
        self.predHidden = predHidden
        self.jointHidden = jointHidden
        self.numClasses = numClasses
        self.numExtraOutputs = numExtraOutputs
        self.activation = activation
        self.vocabulary = vocabulary
    }
}

// MARK: - TDT Decoding Configuration

/// TDT (Token-and-Duration Transducer) decoding configuration
public struct TDTDecodingConfig: Codable, Sendable {
    public let modelType: String
    public let durations: [Int]
    public let maxSymbols: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case durations
        case maxSymbols = "max_symbols"
    }

    public init(
        modelType: String = "tdt",
        durations: [Int] = [0, 1, 2, 3, 4],
        maxSymbols: Int? = nil
    ) {
        self.modelType = modelType
        self.durations = durations
        self.maxSymbols = maxSymbols
    }
}

// MARK: - Full Parakeet TDT Configuration

/// Complete Parakeet TDT model configuration
public struct ParakeetTDTConfig: Codable, Sendable {
    public let preprocessor: PreprocessConfig
    public let encoder: ConformerConfig
    public let decoder: PredictConfig
    public let joint: JointConfig
    public let decoding: TDTDecodingConfig

    public init(
        preprocessor: PreprocessConfig = PreprocessConfig(),
        encoder: ConformerConfig = ConformerConfig(),
        decoder: PredictConfig = PredictConfig(),
        joint: JointConfig = JointConfig(),
        decoding: TDTDecodingConfig = TDTDecodingConfig()
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.decoding = decoding
    }
}

// MARK: - HuggingFace Config Wrapper

/// Wrapper for parsing HuggingFace config.json which has nested structure
public struct HuggingFaceConfig: Codable {
    public let preprocessor: PreprocessConfig?
    public let encoder: ConformerConfig?
    public let decoder: PredictConfig?
    public let joint: JointConfig?
    public let decoding: DecodingWrapper?
    public let modelDefaults: ModelDefaults?

    enum CodingKeys: String, CodingKey {
        case preprocessor
        case encoder
        case decoder
        case joint
        case decoding
        case modelDefaults = "model_defaults"
    }

    public struct DecodingWrapper: Codable {
        public let strategy: String?
        public let greedy: GreedyConfig?
    }

    public struct GreedyConfig: Codable {
        public let maxSymbolsPerStep: Int?

        enum CodingKeys: String, CodingKey {
            case maxSymbolsPerStep = "max_symbols_per_step"
        }
    }

    public struct ModelDefaults: Codable {
        public let tdtDurations: [Int]?

        enum CodingKeys: String, CodingKey {
            case tdtDurations = "tdt_durations"
        }
    }

    /// Convert to ParakeetTDTConfig
    public func toParakeetConfig() -> ParakeetTDTConfig {
        let durations = modelDefaults?.tdtDurations ?? [0, 1, 2, 3, 4]

        return ParakeetTDTConfig(
            preprocessor: preprocessor ?? PreprocessConfig(),
            encoder: encoder ?? ConformerConfig(),
            decoder: decoder ?? PredictConfig(),
            joint: joint ?? JointConfig(),
            decoding: TDTDecodingConfig(modelType: "tdt", durations: durations)
        )
    }
}

// MARK: - Aligned Token Result

/// A single aligned token with timing information
public struct AlignedToken: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let start: Float
    public let duration: Float

    public var end: Float {
        start + duration
    }

    public init(id: Int, text: String, start: Float, duration: Float) {
        self.id = id
        self.text = text
        self.start = start
        self.duration = duration
    }
}

/// A sentence with aligned tokens
public struct AlignedSentence: Sendable, Equatable {
    public let text: String
    public let tokens: [AlignedToken]
    public let start: Float
    public let end: Float

    public var duration: Float {
        end - start
    }

    public init(text: String, tokens: [AlignedToken]) {
        self.text = text
        self.tokens = tokens.sorted { $0.start < $1.start }
        self.start = tokens.first?.start ?? 0
        self.end = tokens.last?.end ?? 0
    }
}

/// Complete transcription result with aligned tokens and sentences
public struct AlignedResult: Sendable, Equatable {
    public let text: String
    public let sentences: [AlignedSentence]
    public let tokens: [AlignedToken]

    public init(text: String, sentences: [AlignedSentence], tokens: [AlignedToken]) {
        self.text = text
        self.sentences = sentences
        self.tokens = tokens
    }

    public static let empty = AlignedResult(text: "", sentences: [], tokens: [])
}
