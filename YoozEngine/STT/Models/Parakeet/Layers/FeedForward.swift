// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Conformer Feed-Forward module
/// Structure: Linear -> SiLU -> Linear
public class FeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    public init(dModel: Int, dFf: Int, useBias: Bool = true) {
        self._linear1.wrappedValue = Linear(dModel, dFf, bias: useBias)
        self._linear2.wrappedValue = Linear(dFf, dModel, bias: useBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)
        out = silu(out)
        out = linear2(out)
        return out
    }
}
