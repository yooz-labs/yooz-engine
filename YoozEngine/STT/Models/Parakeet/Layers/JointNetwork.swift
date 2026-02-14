// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Activation type for Joint Network
public enum JointActivation: String, Codable {
    case relu
    case sigmoid
    case tanh
}

/// Joint Network for RNNT/TDT
/// Combines encoder and prediction network outputs
public class JointNetwork: Module {
    public let numClasses: Int

    @ModuleInfo(key: "enc") var encProj: Linear
    @ModuleInfo(key: "pred") var predProj: Linear
    @ModuleInfo(key: "out") var outLinear: Linear

    private let activation: JointActivation

    public init(config: JointConfig) {
        // num_classes + 1 (blank) + num_extra_outputs (durations)
        self.numClasses = config.numClasses + 1 + config.numExtraOutputs

        // Parse activation
        switch config.activation.lowercased() {
        case "relu":
            self.activation = .relu
        case "sigmoid":
            self.activation = .sigmoid
        case "tanh":
            self.activation = .tanh
        default:
            self.activation = .relu  // Default to relu
        }

        self._encProj.wrappedValue = Linear(config.encoderHidden, config.jointHidden)
        self._predProj.wrappedValue = Linear(config.predHidden, config.jointHidden)
        self._outLinear.wrappedValue = Linear(config.jointHidden, numClasses)
    }

    /// Forward pass
    /// - Parameters:
    ///   - enc: Encoder output [batch, enc_time, encoder_hidden]
    ///   - pred: Prediction output [batch, pred_time, pred_hidden]
    /// - Returns: Joint output [batch, enc_time, pred_time, num_classes]
    public func callAsFunction(_ enc: MLXArray, pred: MLXArray) -> MLXArray {
        // Project encoder and prediction outputs
        let encProj = self.encProj(enc)   // [batch, enc_time, joint_hidden]
        let predProj = self.predProj(pred) // [batch, pred_time, joint_hidden]

        // Broadcast and add
        // enc: [batch, enc_time, 1, joint_hidden]
        // pred: [batch, 1, pred_time, joint_hidden]
        let encExpanded = encProj.expandedDimensions(axis: 2)
        let predExpanded = predProj.expandedDimensions(axis: 1)

        var x = encExpanded + predExpanded  // [batch, enc_time, pred_time, joint_hidden]

        // Apply activation
        switch activation {
        case .relu:
            x = relu(x)
        case .sigmoid:
            x = sigmoid(x)
        case .tanh:
            x = tanh(x)
        }

        // Output projection
        return outLinear(x)  // [batch, enc_time, pred_time, num_classes]
    }

    // MARK: - Optimized Methods for TDT Decoding

    /// Pre-compute encoder projection for all time steps
    /// Call once before the decode loop to avoid redundant computation
    /// - Parameter enc: Encoder output [batch, time, encoder_hidden]
    /// - Returns: Projected encoder output [batch, time, joint_hidden]
    public func projectEncoder(_ enc: MLXArray) -> MLXArray {
        return encProj(enc)
    }

    /// Compute joint output using pre-computed encoder projection
    /// - Parameters:
    ///   - encProjected: Pre-computed encoder projection [batch, 1, joint_hidden]
    ///   - pred: Prediction output [batch, 1, pred_hidden]
    /// - Returns: Joint output [batch, 1, 1, num_classes]
    public func forwardWithProjectedEncoder(_ encProjected: MLXArray, pred: MLXArray) -> MLXArray {
        let predProj = self.predProj(pred)  // [batch, 1, joint_hidden]

        // Broadcast and add
        let encExpanded = encProjected.expandedDimensions(axis: 2)
        let predExpanded = predProj.expandedDimensions(axis: 1)

        var x = encExpanded + predExpanded  // [batch, 1, 1, joint_hidden]

        // Apply activation
        switch activation {
        case .relu:
            x = relu(x)
        case .sigmoid:
            x = sigmoid(x)
        case .tanh:
            x = tanh(x)
        }

        return outLinear(x)
    }
}
