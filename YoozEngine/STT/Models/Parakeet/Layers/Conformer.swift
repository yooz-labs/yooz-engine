// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Conformer Encoder
/// Full encoder stack with positional encoding, subsampling, and conformer blocks
public class Conformer: Module {
    public let config: ConformerConfig

    @ModuleInfo(key: "pos_enc") var posEnc: RelPositionalEncoding?
    @ModuleInfo(key: "pre_encode") var preEncode: Module
    @ModuleInfo(key: "layers") var layers: [ConformerBlock]

    public init(config: ConformerConfig) {
        self.config = config

        // Initialize positional encoding
        if config.selfAttentionModel == "rel_pos" {
            self._posEnc.wrappedValue = RelPositionalEncoding(
                dModel: config.dModel,
                maxLen: config.posEmbMaxLen,
                scaleInput: config.xscaling
            )
        } else {
            self._posEnc.wrappedValue = nil
        }

        // Initialize pre-encoding (subsampling)
        if config.subsamplingFactor > 1 {
            if config.subsampling == "dw_striding" && !config.causalDownsampling {
                self._preEncode.wrappedValue = DwStridingSubsampling(config: config)
            } else {
                fatalError("Only dw_striding non-causal subsampling is implemented")
            }
        } else {
            self._preEncode.wrappedValue = Linear(config.featIn, config.dModel)
        }

        // Initialize conformer layers
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            ConformerBlock(config: config)
        }
    }

    /// Standard forward pass (no streaming)
    public func callAsFunction(
        _ x: MLXArray,
        lengths: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        return callAsFunction(x, lengths: lengths, cache: nil, localContext: nil)
    }

    /// Forward pass with optional cache for streaming
    /// - Parameters:
    ///   - x: Input mel features [batch, time, features]
    ///   - lengths: Sequence lengths per batch
    ///   - cache: Array of ConformerCache for each layer (for streaming)
    ///   - localContext: Optional local attention window (left, right) in encoder frames
    /// - Returns: (encoded features, output lengths)
    public func callAsFunction(
        _ x: MLXArray,
        lengths: MLXArray? = nil,
        cache: [ConformerCache]? = nil,
        localContext: (left: Int, right: Int)? = nil
    ) -> (MLXArray, MLXArray) {
        // Default lengths to full sequence length
        let inputLengths = lengths ?? full([x.dim(0)], values: Int32(x.dim(1)))

        var out: MLXArray
        var outLengths: MLXArray

        // Apply pre-encoding
        if let subsampling = preEncode as? DwStridingSubsampling {
            (out, outLengths) = subsampling(x, lengths: inputLengths)
        } else if let linear = preEncode as? Linear {
            out = linear(x)
            outLengths = inputLengths
        } else {
            fatalError("Unknown pre-encoding type")
        }

        // Apply positional encoding
        // Note: Use cache.offset which is computed from cached keys dimension
        var posEmb: MLXArray?
        if let posEnc = posEnc {
            let offset = cache?.first?.offset ?? 0
            (out, posEmb) = posEnc(out, offset: offset)
        }

        // Apply conformer layers with per-layer cache and optional local attention
        if let cache = cache {
            for (layer, c) in zip(layers, cache) {
                out = layer(out, posEmb: posEmb, cache: c, localContext: localContext)
            }
        } else {
            for layer in layers {
                out = layer(out, posEmb: posEmb, localContext: localContext)
            }
        }

        return (out, outLengths)
    }

    /// Create caches for all layers (for streaming)
    public func createCaches() -> [ConformerCache] {
        return (0..<layers.count).map { _ in ConformerCache() }
    }

    /// Create rotating caches for all layers (for long audio streaming)
    public func createRotatingCaches(capacity: Int, dropSize: Int = 0) -> [RotatingConformerCache] {
        return (0..<layers.count).map { _ in
            RotatingConformerCache(capacity: capacity, dropSize: dropSize)
        }
    }
}

