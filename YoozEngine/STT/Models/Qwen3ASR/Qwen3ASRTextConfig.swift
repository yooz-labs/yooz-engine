// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Configuration for the Qwen3-ASR text decoder.
///
/// Mirrors the `thinker_config.text_config` block in the published
/// `mlx-community/Qwen3-ASR-1.7B-8bit` `config.json`. The decoder is
/// the same Qwen3 architecture used elsewhere in the engine, but the
/// pipeline needs a forward path that accepts pre-computed input
/// embeddings (so audio-tower hidden states can be spliced into the
/// embedding stream at the `<|audio_pad|>` token positions). That
/// path is provided by `Qwen3ASRTextDecoder` rather than the stock
/// `MLXLLM.Qwen3` model — hence a Phase 4-local config.
public struct Qwen3ASRTextConfig: Codable, Sendable, Equatable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    public var attentionBias: Bool

    public init(
        modelType: String = "qwen3",
        vocabSize: Int = 151_936,
        hiddenSize: Int = 2_048,
        intermediateSize: Int = 6_144,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 65_536,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
    }

    /// Custom decoder so the config tolerates HuggingFace's habit of
    /// omitting fields from the canonical default (every `head_dim`
    /// in a Qwen3 checkpoint, but no `model_type` in some forks). All
    /// decoded fields default to the published 1.7B-8bit values when
    /// missing — the validator runs immediately afterwards so a real
    /// mismatch still surfaces as `Qwen3ASRError.invalidConfig`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Qwen3ASRTextConfig()
        self.modelType =
            (try container.decodeIfPresent(String.self, forKey: .modelType))
            ?? defaults.modelType
        self.vocabSize =
            (try container.decodeIfPresent(Int.self, forKey: .vocabSize))
            ?? defaults.vocabSize
        self.hiddenSize =
            (try container.decodeIfPresent(Int.self, forKey: .hiddenSize))
            ?? defaults.hiddenSize
        self.intermediateSize =
            (try container.decodeIfPresent(Int.self, forKey: .intermediateSize))
            ?? defaults.intermediateSize
        self.numHiddenLayers =
            (try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers))
            ?? defaults.numHiddenLayers
        self.numAttentionHeads =
            (try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads))
            ?? defaults.numAttentionHeads
        self.numKeyValueHeads =
            (try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads))
            ?? defaults.numKeyValueHeads
        self.headDim =
            (try container.decodeIfPresent(Int.self, forKey: .headDim))
            ?? defaults.headDim
        self.hiddenAct =
            (try container.decodeIfPresent(String.self, forKey: .hiddenAct))
            ?? defaults.hiddenAct
        self.maxPositionEmbeddings =
            (try container.decodeIfPresent(
                Int.self, forKey: .maxPositionEmbeddings
            )) ?? defaults.maxPositionEmbeddings
        self.rmsNormEps =
            (try container.decodeIfPresent(Float.self, forKey: .rmsNormEps))
            ?? defaults.rmsNormEps
        self.ropeTheta =
            (try container.decodeIfPresent(Float.self, forKey: .ropeTheta))
            ?? defaults.ropeTheta
        self.tieWordEmbeddings =
            (try container.decodeIfPresent(
                Bool.self, forKey: .tieWordEmbeddings
            )) ?? defaults.tieWordEmbeddings
        self.attentionBias =
            (try container.decodeIfPresent(Bool.self, forKey: .attentionBias))
            ?? defaults.attentionBias
    }

    /// Validate semantic invariants. Throws `Qwen3ASRError.invalidConfig`
    /// rather than crashing.
    public func validate() throws {
        if hiddenSize <= 0 || numHiddenLayers <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "hiddenSize (\(hiddenSize)) and numHiddenLayers "
                    + "(\(numHiddenLayers)) must be positive"
            )
        }
        if numAttentionHeads <= 0 || numKeyValueHeads <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "numAttentionHeads (\(numAttentionHeads)) and "
                    + "numKeyValueHeads (\(numKeyValueHeads)) must be positive"
            )
        }
        if numAttentionHeads % numKeyValueHeads != 0 {
            throw Qwen3ASRError.invalidConfig(
                "numAttentionHeads (\(numAttentionHeads)) must be divisible "
                    + "by numKeyValueHeads (\(numKeyValueHeads))"
            )
        }
        if headDim <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "headDim must be > 0, got \(headDim)"
            )
        }
        if intermediateSize <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "intermediateSize must be > 0, got \(intermediateSize)"
            )
        }
        if vocabSize <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "vocabSize must be > 0, got \(vocabSize)"
            )
        }
        if rmsNormEps <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "rmsNormEps must be > 0, got \(rmsNormEps)"
            )
        }
        if ropeTheta <= 0 {
            throw Qwen3ASRError.invalidConfig(
                "ropeTheta must be > 0, got \(ropeTheta)"
            )
        }
    }
}

/// Top-level Qwen3-ASR config covering both the audio encoder and the
/// text decoder, plus the cross-modal token IDs.
public struct Qwen3ASRFullConfig: Codable, Sendable, Equatable {
    public var audio: Qwen3ASRConfig
    public var text: Qwen3ASRTextConfig
    public var audioTokenId: Int
    public var audioStartTokenId: Int
    public var audioEndTokenId: Int
    public var supportLanguages: [String]
    public var quantBits: Int?
    public var quantGroupSize: Int?

    public init(
        audio: Qwen3ASRConfig = Qwen3ASRConfig(),
        text: Qwen3ASRTextConfig = Qwen3ASRTextConfig(),
        audioTokenId: Int = 151_676,
        audioStartTokenId: Int = 151_669,
        audioEndTokenId: Int = 151_670,
        supportLanguages: [String] = [],
        quantBits: Int? = 8,
        quantGroupSize: Int? = 64
    ) {
        self.audio = audio
        self.text = text
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.supportLanguages = supportLanguages
        self.quantBits = quantBits
        self.quantGroupSize = quantGroupSize
    }

    /// Decode the canonical Qwen3-ASR `config.json`. The audio + text
    /// configs live under `thinker_config`; the rest of the keys live
    /// at the top level of the file.
    public static func load(from configURL: URL) throws -> Qwen3ASRFullConfig {
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data)
            as? [String: Any]
        else {
            throw Qwen3ASRError.invalidConfig(
                "config.json at \(configURL.path) is not a JSON object"
            )
        }

        guard let thinker = json["thinker_config"] as? [String: Any] else {
            throw Qwen3ASRError.invalidConfig(
                "config.json missing 'thinker_config' block"
            )
        }

        let audio: Qwen3ASRConfig
        if let audioRaw = thinker["audio_config"] as? [String: Any] {
            let audioData = try JSONSerialization.data(
                withJSONObject: audioRaw
            )
            audio = try JSONDecoder().decode(
                Qwen3ASRConfig.self, from: audioData
            )
        } else {
            audio = Qwen3ASRConfig()
        }

        let text: Qwen3ASRTextConfig
        if let textRaw = thinker["text_config"] as? [String: Any] {
            let textData = try JSONSerialization.data(withJSONObject: textRaw)
            text = try JSONDecoder().decode(
                Qwen3ASRTextConfig.self, from: textData
            )
        } else {
            text = Qwen3ASRTextConfig()
        }

        let audioTokenId =
            (thinker["audio_token_id"] as? Int) ?? 151_676
        let audioStartTokenId =
            (thinker["audio_start_token_id"] as? Int) ?? 151_669
        let audioEndTokenId =
            (thinker["audio_end_token_id"] as? Int) ?? 151_670
        let supportLanguages =
            (json["support_languages"] as? [String]) ?? []

        var bits: Int? = nil
        var groupSize: Int? = nil
        if let q = json["quantization"] as? [String: Any] {
            bits = q["bits"] as? Int
            groupSize = q["group_size"] as? Int
        } else if let q = json["quantization_config"] as? [String: Any] {
            bits = q["bits"] as? Int
            groupSize = q["group_size"] as? Int
        }

        try audio.validate()
        try text.validate()

        return Qwen3ASRFullConfig(
            audio: audio,
            text: text,
            audioTokenId: audioTokenId,
            audioStartTokenId: audioStartTokenId,
            audioEndTokenId: audioEndTokenId,
            supportLanguages: supportLanguages,
            quantBits: bits,
            quantGroupSize: groupSize
        )
    }
}
