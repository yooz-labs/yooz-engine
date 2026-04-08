// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// FastConformer Hybrid Speech-to-Text Model
/// Combines Conformer encoder with RNNT decoder (no duration prediction)
/// Supports Arabic, Persian, and other languages with FastConformer architecture
public final class FastConformerModel: Module {
    public let config: FastConformerHybridConfig

    @ModuleInfo(key: "encoder") var encoder: Conformer
    @ModuleInfo(key: "decoder") var decoder: PredictNetwork
    @ModuleInfo(key: "joint") var joint: JointNetwork

    private let preprocessor: AudioPreprocessor
    private let rnntDecoder: RNNTDecoder

    public init(config: FastConformerHybridConfig) {
        self.config = config

        self._encoder.wrappedValue = Conformer(config: config.encoder)
        self._decoder.wrappedValue = PredictNetwork(config: config.decoder, blankAsPad: true)
        self._joint.wrappedValue = JointNetwork(config: config.joint)

        self.preprocessor = AudioPreprocessor(config: config.preprocessor)
        self.rnntDecoder = RNNTDecoder(
            vocabulary: config.joint.vocabulary,
            subsamplingFactor: config.encoder.subsamplingFactor,
            sampleRate: config.preprocessor.sampleRate,
            hopLength: config.preprocessor.hopLength,
            maxSymbols: config.decoding.maxSymbols
        )
    }

    /// Load model from local directory
    /// - Parameters:
    ///   - directory: Path to directory containing config.json and model.safetensors
    ///   - dtype: Data type for model weights (default: bfloat16)
    /// - Returns: Loaded FastConformerModel
    public static func fromDirectory(
        _ directory: URL,
        dtype: DType = .bfloat16
    ) throws -> FastConformerModel {
        // Load config using FastConformer parser
        let configURL = directory.appendingPathComponent("config.json")
        let config = try FastConformerConfigParser.parse(from: configURL)

        // Initialize model
        let model = FastConformerModel(config: config)

        // Load weights
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        var weights = try loadArrays(url: weightsURL)

        // Sanitize weight keys
        weights = model.sanitizeWeights(weights)

        // Convert to target dtype
        weights = weights.mapValues { $0.asType(dtype) }

        // Update model parameters
        let params = ModuleParameters.unflattened(weights)
        try model.update(parameters: params, verify: .noUnusedKeys)

        eval(model)
        return model
    }

    /// Sanitize weight keys from Python/safetensors format to Swift module format
    /// FastConformer uses similar structure to Parakeet with some differences
    /// - Parameter weights: Raw weights dictionary from safetensors
    /// - Returns: Sanitized weights dictionary matching our module structure
    private func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        // First pass: collect LSTM biases for combining
        var lstmBiasIh: [Int: MLXArray] = [:]
        var lstmBiasHh: [Int: MLXArray] = [:]

        for (key, value) in weights {
            // Skip preprocessor weights (we compute features ourselves)
            if key.hasPrefix("preprocessor.") {
                continue
            }

            // Skip CTC decoder weights (we use RNNT path only)
            if key.hasPrefix("ctc_decoder.") {
                continue
            }

            // Skip BatchNorm running statistics (not trainable parameters)
            if key.contains("batch_norm.running_mean") ||
                key.contains("batch_norm.running_var") ||
                key.contains("batch_norm.num_batches_tracked") {
                continue
            }

            var newKey = key

            // === Decoder/Prediction Network ===
            // decoder.prediction.embed.weight -> decoder.embed.weight
            newKey = newKey.replacingOccurrences(of: "decoder.prediction.", with: "decoder.")

            // === LSTM weight conversion ===
            // PyTorch: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
            // MLX: lstm.0.Wx, lstm.0.Wh, lstm.0.bias (combined)
            if newKey.contains("dec_rnn.lstm.") {
                // Extract layer number from _lN suffix
                if let match = newKey.range(of: "_l(\\d+)$", options: .regularExpression) {
                    let layerStr = String(newKey[match]).dropFirst(2) // Remove "_l"
                    if let layerNum = Int(layerStr) {
                        if newKey.contains("weight_ih") {
                            // weight_ih_lN -> lstm.N.Wx
                            newKey = "decoder.dec_rnn.lstm.\(layerNum).Wx"
                            sanitized[newKey] = value
                            continue
                        } else if newKey.contains("weight_hh") {
                            // weight_hh_lN -> lstm.N.Wh
                            newKey = "decoder.dec_rnn.lstm.\(layerNum).Wh"
                            sanitized[newKey] = value
                            continue
                        } else if newKey.contains("bias_ih") {
                            // Collect for combining
                            lstmBiasIh[layerNum] = value
                            continue
                        } else if newKey.contains("bias_hh") {
                            // Collect for combining
                            lstmBiasHh[layerNum] = value
                            continue
                        }
                    }
                }
            }

            // === Joint Network ===
            // joint.joint_net.2.* -> joint.out.*
            newKey = newKey.replacingOccurrences(of: "joint.joint_net.2.", with: "joint.out.")

            // === Subsampling conv layer index remapping ===
            // Python uses indices with gaps (for ReLU): conv.0, conv.2, conv.3, conv.5, conv.6
            // Swift uses consecutive: conv.0, conv.1, conv.2, conv.3, conv.4
            if newKey.contains("pre_encode.conv.") {
                let pythonIndices = [0, 2, 3, 5, 6, 8, 9]
                let swiftIndices =  [0, 1, 2, 3, 4, 5, 6]
                for (pythonIdx, swiftIdx) in zip(pythonIndices, swiftIndices) {
                    let pythonPattern = "pre_encode.conv.\(pythonIdx)."
                    let swiftPattern = "pre_encode.conv.\(swiftIdx)."
                    if newKey.contains(pythonPattern) {
                        newKey = newKey.replacingOccurrences(of: pythonPattern, with: swiftPattern)
                        break
                    }
                }
            }

            // === Convolution weight format conversions ===
            var finalValue = value
            if newKey.hasSuffix(".weight") {
                // Conv2d weight format conversion (subsampling layers)
                // PyTorch Conv2d: OIHW (out_channels, in_channels, kernel_h, kernel_w)
                // MLX-Swift Conv2d: OHWI (out_channels, kernel_h, kernel_w, in_channels)
                // Transpose from (O, I, H, W) to (O, H, W, I) using axes (0, 2, 3, 1)
                if newKey.contains("pre_encode.conv."), value.ndim == 4 {
                    finalValue = value.transposed(0, 2, 3, 1)
                }

                // Conv1d weight format conversion (conformer blocks)
                // PyTorch Conv1d: OIL (out_channels, in_channels, kernel_length)
                // MLX-Swift Conv1d: OLI (out_channels, kernel_length, in_channels)
                // Transpose from (O, I, L) to (O, L, I) using axes (0, 2, 1)
                if value.ndim == 3, newKey.contains("pointwise_conv") || newKey.contains("depthwise_conv") {
                    finalValue = value.transposed(0, 2, 1)
                }
            }

            sanitized[newKey] = finalValue
        }

        // Combine LSTM biases: bias = bias_ih + bias_hh
        for (layerNum, biasIh) in lstmBiasIh {
            if let biasHh = lstmBiasHh[layerNum] {
                let combinedBias = biasIh + biasHh
                sanitized["decoder.dec_rnn.lstm.\(layerNum).bias"] = combinedBias
            }
        }

        return sanitized
    }

    /// Transcribe audio samples to text
    /// - Parameter audio: Audio samples [time] at config.preprocessor.sampleRate Hz
    /// - Returns: Transcription result with aligned tokens
    public func transcribe(_ audio: MLXArray) -> TranscriptionResult {
        // Compute mel spectrogram
        let mel = preprocessor.logMelSpectrogram(audio)

        // Encode
        let (features, lengths) = encoder(mel)
        eval(features, lengths)

        // Decode using RNNT
        let tokens = rnntDecoder.decode(
            features: features,
            lengths: lengths,
            decoder: decoder,
            joint: joint
        )

        // Build result
        return TranscriptionResult(tokens: tokens)
    }

    /// Transcribe audio from Float array
    /// - Parameter audio: Audio samples at config.preprocessor.sampleRate Hz
    /// - Returns: Transcription result
    public func transcribe(_ audio: [Float]) -> TranscriptionResult {
        let x = MLXArray(audio)
        return transcribe(x)
    }

    // MARK: - Streaming Transcription

    /// Create encoder caches for streaming
    public func createEncoderCaches() -> [ConformerCache] {
        encoder.createCaches()
    }

    /// Create rotating encoder caches for long audio streaming
    public func createRotatingEncoderCaches(capacity: Int, dropSize: Int = 0) -> [RotatingConformerCache] {
        encoder.createRotatingCaches(capacity: capacity, dropSize: dropSize)
    }

    /// Streaming transcription with encoder cache
    public func transcribeStreaming(
        _ allAudio: [Float],
        encoderCache: [ConformerCache],
        previousEncoderOutput: MLXArray?
    ) -> (tokens: [AlignedToken], newEncoderOutput: MLXArray) {
        // Compute mel spectrogram for ALL audio
        let allMel = preprocessor.logMelSpectrogram(MLXArray(allAudio))

        // Get cache offset
        let encoderOffset = encoderCache.first?.offset ?? 0
        let subsamplingFactor = config.encoder.subsamplingFactor
        let melOffset = encoderOffset * subsamplingFactor

        // Slice mel to get only NEW frames
        let melToEncode: MLXArray = if melOffset > 0, allMel.dim(1) > melOffset {
            allMel[0..., melOffset..., 0...]
        } else {
            allMel
        }

        // Encode new mel frames with cache
        let (newEncoderOutput, _) = encoder(melToEncode, cache: encoderCache)
        eval(newEncoderOutput)

        // Combine with previous encoder outputs
        let allEncoderOutput: MLXArray = if let prevOutput = previousEncoderOutput {
            concatenated([prevOutput, newEncoderOutput], axis: 1)
        } else {
            newEncoderOutput
        }

        // Get total length
        let totalLen = full([allEncoderOutput.dim(0)], values: Int32(allEncoderOutput.dim(1)))

        // Decode using RNNT
        let tokens = rnntDecoder.decode(
            features: allEncoderOutput,
            lengths: totalLen,
            decoder: decoder,
            joint: joint
        )

        return (tokens, newEncoderOutput)
    }

    /// Decode encoder features to tokens using RNNT greedy search
    public func rnntDecode(features: MLXArray, length: Int) -> [AlignedToken] {
        let lengths = full([features.dim(0)], values: Int32(length))
        return rnntDecoder.decode(
            features: features,
            lengths: lengths,
            decoder: decoder,
            joint: joint
        )
    }
}
