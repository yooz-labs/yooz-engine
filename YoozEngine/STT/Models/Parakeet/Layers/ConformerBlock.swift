// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Conformer Block
/// Structure: FF1 (0.5x) -> Self-Attention -> Conv -> FF2 (0.5x) -> LayerNorm
public class ConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: FeedForward

    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention

    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvolution

    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: FeedForward

    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    public init(config: ConformerConfig) {
        let ffHiddenDim = config.dModel * config.ffExpansionFactor

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._feedForward1.wrappedValue = FeedForward(
            dModel: config.dModel,
            dFf: ffHiddenDim,
            useBias: config.useBias
        )

        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(
            nHead: config.nHeads,
            nFeat: config.dModel,
            bias: config.useBias
        )

        self._normConv.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._conv.wrappedValue = ConformerConvolution(
            dModel: config.dModel,
            convKernelSize: config.convKernelSize,
            useBias: config.useBias
        )

        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._feedForward2.wrappedValue = FeedForward(
            dModel: config.dModel,
            dFf: ffHiddenDim,
            useBias: config.useBias
        )

        self._normOut.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    /// Standard forward pass (no streaming)
    public func callAsFunction(
        _ x: MLXArray,
        posEmb: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        callAsFunction(x, posEmb: posEmb, mask: mask, cache: nil, localContext: nil)
    }

    /// Forward pass with optional cache for streaming
    public func callAsFunction(
        _ x: MLXArray,
        posEmb: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: ConformerCache? = nil,
        localContext: (left: Int, right: Int)? = nil
    ) -> MLXArray {
        var out = x

        // First feed-forward with 0.5 residual
        out = out + 0.5 * feedForward1(normFeedForward1(out))

        // Self-attention with KV cache and optional local attention
        let xNorm = normSelfAtt(out)
        if let posEmb {
            out = out + selfAttn(
                xNorm, k: xNorm, v: xNorm,
                posEmb: posEmb,
                mask: mask,
                cache: cache,
                localContext: localContext
            )
        }

        // Convolution with conv cache
        out = out + conv(normConv(out), cache: cache)

        // Second feed-forward with 0.5 residual
        out = out + 0.5 * feedForward2(normFeedForward2(out))

        // Final layer norm
        return normOut(out)
    }
}
