// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Parser for NeMo-style config.json files
/// Handles the nested structure used by Parakeet models
public struct NeMoConfigParser {

    /// Parse NeMo config.json into ParakeetTDTConfig
    public static func parse(from url: URL) throws -> ParakeetTDTConfig {
        let data = try Data(contentsOf: url)
        return try parse(from: data)
    }

    /// Parse NeMo config.json data into ParakeetTDTConfig
    public static func parse(from data: Data) throws -> ParakeetTDTConfig {
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let preprocessor = parsePreprocessor(json["preprocessor"] as? [String: Any])
        let encoder = parseEncoder(json["encoder"] as? [String: Any])
        let (decoder, vocabulary) = parseDecoderAndJoint(
            decoder: json["decoder"] as? [String: Any],
            joint: json["joint"] as? [String: Any]
        )
        let joint = parseJoint(json["joint"] as? [String: Any], vocabulary: vocabulary)
        let decoding = parseDecoding(
            modelDefaults: json["model_defaults"] as? [String: Any],
            decoding: json["decoding"] as? [String: Any]
        )

        return ParakeetTDTConfig(
            preprocessor: preprocessor,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            decoding: decoding
        )
    }

    private static func parsePreprocessor(_ dict: [String: Any]?) -> PreprocessConfig {
        guard let dict = dict else { return PreprocessConfig() }

        return PreprocessConfig(
            sampleRate: dict["sample_rate"] as? Int ?? 16000,
            normalize: dict["normalize"] as? String ?? "per_feature",
            windowSize: (dict["window_size"] as? Double).map { Float($0) } ?? 0.025,
            windowStride: (dict["window_stride"] as? Double).map { Float($0) } ?? 0.01,
            window: dict["window"] as? String ?? "hann",
            features: dict["features"] as? Int ?? 80,
            nFft: dict["n_fft"] as? Int ?? 512,
            dither: (dict["dither"] as? Double).map { Float($0) } ?? 0.0,
            padTo: dict["pad_to"] as? Int ?? 0,
            padValue: (dict["pad_value"] as? Double).map { Float($0) } ?? 0.0,
            preemph: (dict["preemph"] as? Double).map { Float($0) } ?? 0.97
        )
    }

    private static func parseEncoder(_ dict: [String: Any]?) -> ConformerConfig {
        guard let dict = dict else { return ConformerConfig() }

        return ConformerConfig(
            featIn: dict["feat_in"] as? Int ?? 80,
            nLayers: dict["n_layers"] as? Int ?? 17,
            dModel: dict["d_model"] as? Int ?? 512,
            nHeads: dict["n_heads"] as? Int ?? 8,
            ffExpansionFactor: dict["ff_expansion_factor"] as? Int ?? 4,
            subsamplingFactor: dict["subsampling_factor"] as? Int ?? 4,
            selfAttentionModel: dict["self_attention_model"] as? String ?? "rel_pos",
            subsampling: dict["subsampling"] as? String ?? "dw_striding",
            convKernelSize: dict["conv_kernel_size"] as? Int ?? 31,
            subsamplingConvChannels: dict["subsampling_conv_channels"] as? Int ?? 256,
            posEmbMaxLen: dict["pos_emb_max_len"] as? Int ?? 5000,
            causalDownsampling: dict["causal_downsampling"] as? Bool ?? false,
            useBias: dict["use_bias"] as? Bool ?? true,
            xscaling: dict["xscaling"] as? Bool ?? false,
            subsamplingConvChunkingFactor: dict["subsampling_conv_chunking_factor"] as? Int ?? 1
        )
    }

    private static func parseDecoderAndJoint(
        decoder: [String: Any]?,
        joint: [String: Any]?
    ) -> (PredictConfig, [String]) {
        // Get vocab size and vocabulary
        let vocabSize = decoder?["vocab_size"] as? Int ?? joint?["num_classes"] as? Int ?? 128
        let vocabulary = joint?["vocabulary"] as? [String] ?? []

        // Get prednet config
        let prednet = decoder?["prednet"] as? [String: Any]

        let config = PredictConfig(
            vocabSize: vocabSize,
            predHidden: prednet?["pred_hidden"] as? Int ?? 640,
            rnnHiddenSize: prednet?["rnn_hidden_size"] as? Int ?? prednet?["pred_hidden"] as? Int ?? 640,
            predRnnLayers: prednet?["pred_rnn_layers"] as? Int ?? 2,
            blankAsLastToken: decoder?["blank_as_pad"] as? Bool ?? true
        )

        return (config, vocabulary)
    }

    private static func parseJoint(_ dict: [String: Any]?, vocabulary: [String]) -> JointConfig {
        guard let dict = dict else {
            return JointConfig(vocabulary: vocabulary)
        }

        // Get jointnet config
        let jointnet = dict["jointnet"] as? [String: Any]

        return JointConfig(
            encoderHidden: jointnet?["encoder_hidden"] as? Int ?? 512,
            predHidden: jointnet?["pred_hidden"] as? Int ?? 640,
            jointHidden: jointnet?["joint_hidden"] as? Int ?? 640,
            numClasses: dict["num_classes"] as? Int ?? 128,
            numExtraOutputs: dict["num_extra_outputs"] as? Int ?? 5,
            activation: jointnet?["activation"] as? String ?? "relu",
            vocabulary: vocabulary
        )
    }

    private static func parseDecoding(
        modelDefaults: [String: Any]?,
        decoding: [String: Any]?
    ) -> TDTDecodingConfig {
        let durations = modelDefaults?["tdt_durations"] as? [Int] ?? [0, 1, 2, 3, 4]

        // Get max_symbols from greedy config
        let greedy = decoding?["greedy"] as? [String: Any]
        let maxSymbols = greedy?["max_symbols_per_step"] as? Int

        return TDTDecodingConfig(
            modelType: "tdt",
            durations: durations,
            maxSymbols: maxSymbols
        )
    }
}
