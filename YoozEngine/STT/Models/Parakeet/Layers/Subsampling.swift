// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Depthwise Striding Subsampling for Conformer
/// Reduces temporal resolution of input features using strided convolutions
/// Structure: Conv2d -> ReLU -> [DepthwiseConv2d -> PointwiseConv2d -> ReLU] * (sampling_num-1)
public class DwStridingSubsampling: Module {
    private let samplingNum: Int
    private let stride: Int = 2
    private let kernelSize: Int = 3
    private let padding: Int = 1  // (kernelSize - 1) / 2
    private let convChannels: Int

    // Use a Sequential-like approach with explicit layer indexing
    // Layer indices match weight keys: 0, 2, 3, 5, 6, etc.
    @ModuleInfo(key: "conv") var convLayers: [Conv2d]
    @ModuleInfo(key: "out") var outLinear: Linear

    public init(config: ConformerConfig) {
        precondition(
            config.subsamplingFactor > 0 &&
            (config.subsamplingFactor & (config.subsamplingFactor - 1)) == 0,
            "subsamplingFactor must be a power of 2"
        )

        self.samplingNum = Int(log2(Double(config.subsamplingFactor)))
        self.convChannels = config.subsamplingConvChannels

        // Calculate final frequency dimension after subsampling
        var finalFreqDim = config.featIn
        for _ in 0..<samplingNum {
            finalFreqDim = (finalFreqDim + 2 * padding - kernelSize) / stride + 1
            precondition(finalFreqDim >= 1, "Non-positive final frequency dimension!")
        }

        // Build conv layers with indices matching Python weights
        // Weight keys: conv.0, conv.2, conv.3, conv.5, conv.6, ...
        // (ReLU layers at indices 1, 4, 7 have no weights)
        var layers: [Conv2d] = []

        // [0] First conv: regular Conv2d(1 -> convChannels)
        layers.append(Conv2d(
            inputChannels: 1,
            outputChannels: convChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        ))

        // For each additional subsampling step (sampling_num - 1 times)
        for _ in 0..<(samplingNum - 1) {
            // Depthwise conv: Conv2d with groups=in_channels
            layers.append(Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: convChannels  // This makes it depthwise
            ))
            // Pointwise conv: 1x1 convolution
            layers.append(Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                padding: IntOrPair(0)
            ))
        }

        self._convLayers.wrappedValue = layers
        self._outLinear.wrappedValue = Linear(convChannels * finalFreqDim, config.dModel)
    }

    private func convForward(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, height, width] -> transpose for MLXNN Conv2d
        // MLXNN Conv2d expects [batch, height, width, channels]
        var out = x.transposed(0, 2, 3, 1)

        // Layer structure: conv0 -> relu -> [conv_dw -> conv_pw -> relu] * (sampling_num-1)
        for (idx, conv) in convLayers.enumerated() {
            out = conv(out)
            // Apply ReLU after first conv (idx=0) and after each pointwise conv
            // idx 0 -> first conv, relu
            // idx 1 -> depthwise, no relu
            // idx 2 -> pointwise, relu
            // idx 3 -> depthwise, no relu
            // idx 4 -> pointwise, relu
            // etc.
            if idx == 0 || (idx > 0 && idx % 2 == 0) {
                out = relu(out)
            }
        }

        // Transpose back: [batch, height, width, channels] -> [batch, channels, height, width]
        return out.transposed(0, 3, 1, 2)
    }

    public func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        // Update lengths for subsampling
        var outLengths = lengths.asType(.float32)
        for _ in 0..<samplingNum {
            outLengths = floor((outLengths + Float(2 * padding - kernelSize)) / Float(stride)) + 1.0
        }
        outLengths = outLengths.asType(.int32)

        // Add channel dimension: [batch, time, freq] -> [batch, 1, time, freq]
        var out = x.expandedDimensions(axis: 1)

        // Apply conv layers
        out = convForward(out)

        // Reshape: [batch, channels, time, freq] -> [batch, time, channels*freq]
        let (batch, channels, time, freq) = (out.dim(0), out.dim(1), out.dim(2), out.dim(3))
        out = out.transposed(0, 2, 1, 3).reshaped([batch, time, channels * freq])

        // Project to d_model
        out = outLinear(out)

        return (out, outLengths)
    }
}
