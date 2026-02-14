// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX

/// Creates attention masks for local (windowed) attention
/// This bounds memory usage and allows streaming with large context sizes
public enum LocalAttentionMask {

    /// Create a local attention mask that restricts attention to a sliding window
    ///
    /// For each query position q, only key positions k where:
    ///   (q - leftContext) <= k <= (q + rightContext)
    /// are attended to.
    ///
    /// - Parameters:
    ///   - queryLen: Number of query positions
    ///   - keyLen: Number of key positions
    ///   - leftContext: How many positions to the left to attend (inclusive)
    ///   - rightContext: How many positions to the right to attend (inclusive)
    ///   - offset: Offset for streaming (key positions start at this offset relative to query)
    /// - Returns: Mask of shape [1, 1, queryLen, keyLen] where -inf means masked, 0 means attend
    public static func create(
        queryLen: Int,
        keyLen: Int,
        leftContext: Int,
        rightContext: Int,
        offset: Int = 0
    ) -> MLXArray {
        // Create position indices
        // queryPositions: [0, 1, 2, ..., queryLen-1] + offset
        // keyPositions: [0, 1, 2, ..., keyLen-1]
        let queryPositions = MLXArray(0..<queryLen).asType(.int32) + Int32(offset)
        let keyPositions = MLXArray(0..<keyLen).asType(.int32)

        // Expand for broadcasting: [queryLen, 1] and [1, keyLen]
        let qExpanded = queryPositions.expandedDimensions(axis: 1)
        let kExpanded = keyPositions.expandedDimensions(axis: 0)

        // Compute relative positions: key - query
        // Negative means key is to the left of query
        // Positive means key is to the right of query
        let relativePos = kExpanded - qExpanded

        // Create mask: attend if -leftContext <= relativePos <= rightContext
        let leftOk = relativePos .>= -Int32(leftContext)
        let rightOk = relativePos .<= Int32(rightContext)

        // Element-wise AND using logicalAnd
        let attendMask = logicalAnd(leftOk, rightOk)

        // Convert to attention mask format: 0 for attend, -inf for mask
        let mask = which(attendMask, Float(0), -Float.infinity)

        // Expand to [1, 1, queryLen, keyLen] for broadcasting with attention scores
        return mask.expandedDimensions(axes: [0, 1])
    }

    /// Create a causal local attention mask (can only attend to past positions)
    ///
    /// - Parameters:
    ///   - queryLen: Number of query positions
    ///   - keyLen: Number of key positions
    ///   - leftContext: How many positions to the left to attend
    ///   - offset: Offset for streaming
    /// - Returns: Causal local attention mask
    public static func createCausal(
        queryLen: Int,
        keyLen: Int,
        leftContext: Int,
        offset: Int = 0
    ) -> MLXArray {
        return create(
            queryLen: queryLen,
            keyLen: keyLen,
            leftContext: leftContext,
            rightContext: 0,  // Causal: no future positions
            offset: offset
        )
    }

    /// Combine a local attention mask with an existing mask
    ///
    /// - Parameters:
    ///   - localMask: Local attention mask from create()
    ///   - existingMask: Existing mask (e.g., padding mask)
    /// - Returns: Combined mask where either mask blocks attention
    public static func combine(
        localMask: MLXArray,
        existingMask: MLXArray
    ) -> MLXArray {
        // Both masks use -inf for blocked positions
        // Taking element-wise minimum combines them correctly
        return minimum(localMask, existingMask)
    }
}
