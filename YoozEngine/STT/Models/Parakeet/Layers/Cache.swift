// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Key-Value cache for transformer attention layers
/// Used for incremental decoding and streaming inference
public class KVCache {
    public var keys: MLXArray?
    public var values: MLXArray?

    public init() {
        self.keys = nil
        self.values = nil
    }

    public init(keys: MLXArray?, values: MLXArray?) {
        self.keys = keys
        self.values = values
    }

    /// Current offset (sequence length) in the cache
    public var offset: Int {
        keys?.dim(2) ?? 0
    }

    /// Update cache with new keys and values, concatenating along sequence dimension
    /// - Returns: Updated (keys, values) tensors
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            // Concatenate along sequence dimension (axis 2)
            self.keys = concatenated([existingKeys, newKeys], axis: 2)
            self.values = concatenated([existingValues, newValues], axis: 2)
        } else {
            self.keys = newKeys
            self.values = newValues
        }

        return (self.keys!, self.values!)
    }

    /// Reset the cache
    public func reset() {
        self.keys = nil
        self.values = nil
    }
}
