// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - FastConformer Hybrid Configuration

/// Complete FastConformer Hybrid model configuration
/// Similar to ParakeetTDTConfig but for hybrid RNNT+CTC models (no duration prediction)
public struct FastConformerHybridConfig: Codable, Sendable {
    public let preprocessor: PreprocessConfig
    public let encoder: ConformerConfig
    public let decoder: PredictConfig
    public let joint: JointConfig
    public let decoding: RNNTDecodingConfig

    public init(
        preprocessor: PreprocessConfig = PreprocessConfig(),
        encoder: ConformerConfig = ConformerConfig(),
        decoder: PredictConfig = PredictConfig(),
        joint: JointConfig = JointConfig(),
        decoding: RNNTDecodingConfig = RNNTDecodingConfig()
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.decoding = decoding
    }
}

// MARK: - RNNT Decoding Configuration

/// RNNT (Hybrid) decoding configuration
/// Unlike TDT, this has no duration prediction
public struct RNNTDecodingConfig: Codable, Sendable {
    public let modelType: String
    public let maxSymbols: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case maxSymbols = "max_symbols"
    }

    public init(
        modelType: String = "hybrid",
        maxSymbols: Int? = nil
    ) {
        self.modelType = modelType
        self.maxSymbols = maxSymbols
    }
}

// MARK: - FastConformer Config Parser

/// Parser for FastConformer config.json files
public struct FastConformerConfigParser {

    /// Parse FastConformer config from a config.json file
    /// - Parameter url: URL to the config.json file
    /// - Returns: Parsed FastConformerHybridConfig
    public static func parse(from url: URL) throws -> FastConformerHybridConfig {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(FastConformerConfigWrapper.self, from: data)
        return config.toFastConformerConfig()
    }
}

// MARK: - Config Wrapper

/// Wrapper for parsing FastConformer config.json structure
/// The structure is similar to Parakeet but with hybrid-specific fields
struct FastConformerConfigWrapper: Codable {
    let preprocessor: PreprocessConfig?
    let encoder: ConformerConfig?
    let decoder: PredictConfig?
    let joint: JointConfig?
    let decoding: DecodingWrapper?
    let modelDefaults: ModelDefaults?

    enum CodingKeys: String, CodingKey {
        case preprocessor
        case encoder
        case decoder
        case joint
        case decoding
        case modelDefaults = "model_defaults"
    }

    struct DecodingWrapper: Codable {
        let modelType: String?
        let durations: [Int]?
        let maxSymbols: Int?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case durations
            case maxSymbols = "max_symbols"
        }
    }

    struct ModelDefaults: Codable {
        let predHidden: Int?
        let jointHidden: Int?

        enum CodingKeys: String, CodingKey {
            case predHidden = "pred_hidden"
            case jointHidden = "joint_hidden"
        }
    }

    func toFastConformerConfig() -> FastConformerHybridConfig {
        // Log any missing sections that use defaults
        #if DEBUG
        var defaultedSections: [String] = []
        if preprocessor == nil { defaultedSections.append("preprocessor") }
        if encoder == nil { defaultedSections.append("encoder") }
        if decoder == nil { defaultedSections.append("decoder") }
        if joint == nil { defaultedSections.append("joint") }
        if decoding == nil { defaultedSections.append("decoding") }

        if !defaultedSections.isEmpty {
            print("[FastConformerConfig] Warning: Using defaults for missing sections: \(defaultedSections.joined(separator: ", "))")
        }
        #endif

        return FastConformerHybridConfig(
            preprocessor: preprocessor ?? PreprocessConfig(),
            encoder: encoder ?? ConformerConfig(),
            decoder: decoder ?? PredictConfig(),
            joint: joint ?? JointConfig(),
            decoding: RNNTDecodingConfig(
                modelType: decoding?.modelType ?? "hybrid",
                maxSymbols: decoding?.maxSymbols
            )
        )
    }
}
