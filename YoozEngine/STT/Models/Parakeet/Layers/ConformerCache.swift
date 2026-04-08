// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Extended cache for Conformer layers
/// Supports both KV caching and convolution state caching
public class ConformerCache: KVCache {
    /// Convolution state cache for streaming
    public var conv: MLXArray?

    /// Current offset (number of frames processed)
    public var frameOffset: Int = 0

    override public init() {
        super.init()
    }

    /// Update convolution state and return padded input for streaming convolution
    /// - Parameters:
    ///   - x: Input tensor [batch, time, channels]
    ///   - padding: Required padding (kernel_size - 1) / 2
    /// - Returns: Padded input tensor
    public func updateAndFetchConv(_ x: MLXArray, padding: Int) -> MLXArray {
        guard padding > 0 else { return x }

        let (B, S, D) = (x.dim(0), x.dim(1), x.dim(2))

        // Initialize conv cache if needed
        if conv == nil {
            conv = MLXArray.zeros([B, padding, D]).asType(x.dtype)
        }

        // How many tokens to cache from this input
        let tokensToCache = min(padding, S)
        let cacheUpdate = x[0..., (S - tokensToCache)..<S, 0...]

        // Update cache
        if tokensToCache < padding {
            // Partial update - shift and append
            conv = concatenated([conv![0..., tokensToCache..., 0...], cacheUpdate], axis: 1)
        } else {
            // Full replacement
            conv = cacheUpdate
        }

        // Prepend cached frames to input
        return concatenated([conv!, x], axis: 1)
    }

    /// Reset all cache state
    override public func reset() {
        super.reset()
        conv = nil
        frameOffset = 0
    }
}

/// Rotating cache that limits memory usage for long audio
/// Implements the same logic as Python's RotatingConformerCache
public class RotatingConformerCache: ConformerCache {
    /// Maximum number of frames to keep in KV cache
    public let capacity: Int

    /// Number of frames to drop when processing new audio
    /// This defines the "draft region" that gets reprocessed
    public let dropSize: Int

    public init(capacity: Int, dropSize: Int = 0) {
        self.capacity = capacity
        self.dropSize = dropSize
        super.init()
    }

    /// Update KV cache with rotation to maintain bounded memory
    override public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let (B, H, S, D) = (newKeys.dim(0), newKeys.dim(1), newKeys.dim(2), newKeys.dim(3))

        // Initialize cache if needed
        if self.keys == nil || self.values == nil {
            self.keys = MLXArray.zeros([B, H, capacity, D]).asType(newKeys.dtype)
            self.values = MLXArray.zeros([B, H, capacity, D]).asType(newValues.dtype)
        }

        // Get historical KV from cache
        let histK: MLXArray
        let histV: MLXArray

        if frameOffset < capacity {
            // Cache not full yet - use what we have
            histK = self.keys![0..., 0..., 0..<frameOffset, 0...]
            histV = self.values![0..., 0..., 0..<frameOffset, 0...]
        } else {
            // Cache full - rotate to get chronological order
            let shift = -(frameOffset % capacity)
            histK = roll(self.keys!, shift: shift, axis: 2)
            histV = roll(self.values!, shift: shift, axis: 2)
        }

        // Concatenate history with new KV for attention
        let kOut = concatenated([histK, newKeys], axis: 2)
        let vOut = concatenated([histV, newValues], axis: 2)

        // Update cache with new values (excluding draft region)
        let toCache = min(max(0, S - dropSize), capacity)
        if toCache > 0 {
            let kChunk = newKeys[0..., 0..., (S - dropSize - toCache)..<(S - dropSize), 0...]
            let vChunk = newValues[0..., 0..., (S - dropSize - toCache)..<(S - dropSize), 0...]

            ringAppend(kChunk, to: &self.keys!)
            ringAppend(vChunk, to: &self.values!)
            frameOffset += toCache
        }

        return (kOut, vOut)
    }

    /// Append to cache in ring buffer fashion
    private func ringAppend(_ new: MLXArray, to buffer: inout MLXArray) {
        let C = capacity
        let pos = frameOffset % C
        let T = new.dim(2)

        let first = min(T, C - pos)

        // Write first portion
        if first > 0 {
            // Note: MLX doesn't support in-place slice assignment
            // We need to reconstruct the array
            let before = buffer[0..., 0..., 0..<pos, 0...]
            let after = buffer[0..., 0..., (pos + first)..., 0...]
            let newSlice = new[0..., 0..., 0..<first, 0...]
            buffer = concatenated([before, newSlice, after], axis: 2)
        }

        // Write wraparound portion if needed
        if T > first {
            let wrapSlice = new[0..., 0..., first..., 0...]
            let after = buffer[0..., 0..., (T - first)..., 0...]
            buffer = concatenated([wrapSlice, after], axis: 2)
        }
    }

    /// Update convolution cache with drop size consideration
    override public func updateAndFetchConv(_ x: MLXArray, padding: Int) -> MLXArray {
        guard padding > 0 else { return x }

        let (B, S, D) = (x.dim(0), x.dim(1), x.dim(2))

        // Initialize conv cache if needed
        if conv == nil {
            conv = MLXArray.zeros([B, padding, D]).asType(x.dtype)
        }

        // Only cache from non-draft region
        if S > dropSize {
            let tokensToCache = min(padding, S - dropSize)
            let cacheUpdate = x[0..., (S - dropSize - tokensToCache)..<(S - dropSize), 0...]

            if tokensToCache < padding {
                conv = concatenated([conv![0..., tokensToCache..., 0...], cacheUpdate], axis: 1)
            } else {
                conv = cacheUpdate
            }
        }

        // Prepend cached frames and pad end
        var result = concatenated([conv!, x], axis: 1)
        result = padded(result, widths: [.init((0, 0)), .init((0, padding)), .init((0, 0))])

        return result
    }
}
