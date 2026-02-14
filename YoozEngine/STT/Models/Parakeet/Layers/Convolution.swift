// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Conformer Convolution module
/// Structure: Pointwise Conv (2x channels) -> GLU -> Depthwise Conv -> BatchNorm -> SiLU -> Pointwise Conv
public class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    /// Kernel size for convolution (needed for streaming cache)
    public let kernelSize: Int

    /// Padding amount (needed for streaming cache)
    public let convPadding: Int

    public init(dModel: Int, convKernelSize: Int, useBias: Bool = true) {
        precondition((convKernelSize - 1) % 2 == 0, "convKernelSize must be odd")

        self.kernelSize = convKernelSize
        self.convPadding = (convKernelSize - 1) / 2

        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel * 2,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: useBias
        )

        // Depthwise convolution: groups=dModel means each input channel has its own filter
        // For streaming, we use padding=0 and handle it manually with cache
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: convKernelSize,
            stride: 1,
            padding: convPadding,
            groups: dModel,  // Makes this a depthwise convolution
            bias: useBias
        )

        self._batchNorm.wrappedValue = BatchNorm(featureCount: dModel)

        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: useBias
        )
    }

    /// Standard forward pass (no streaming)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, time, channels]
        var out = pointwiseConv1(x)

        // GLU: split in half along channel dimension, apply sigmoid to one half
        out = glu(out, axis: -1)

        out = depthwiseConv(out)
        out = batchNorm(out)
        out = silu(out)
        out = pointwiseConv2(out)

        return out
    }

    /// Streaming forward pass with cache
    /// - Parameters:
    ///   - x: Input tensor [batch, time, channels]
    ///   - cache: ConformerCache for storing convolution state
    /// - Returns: Output tensor
    public func callAsFunction(_ x: MLXArray, cache: ConformerCache?) -> MLXArray {
        guard let cache = cache else {
            return callAsFunction(x)
        }

        var out = pointwiseConv1(x)
        out = glu(out, axis: -1)

        // For streaming, use cache to handle convolution state
        // The cache prepends previous frames for continuity
        out = cache.updateAndFetchConv(out, padding: convPadding)

        // Apply depthwise conv (cache already added padding)
        out = depthwiseConv(out)

        // Trim to original sequence length
        let seqLen = x.dim(1)
        out = out[0..., 0..<seqLen, 0...]

        out = batchNorm(out)
        out = silu(out)
        out = pointwiseConv2(out)

        return out
    }
}

/// GLU activation: splits input in half and applies sigmoid gate
/// out = a * sigmoid(b) where input = [a, b] along axis
public func glu(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let parts = split(x, parts: 2, axis: axis)
    return parts[0] * sigmoid(parts[1])
}
