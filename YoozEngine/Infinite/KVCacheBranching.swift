// KVCacheBranching.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import MLX
import MLXLMCommon

/// Errors raised while branching (forking) a live `[any KVCache]` array for
/// a durable Infinite session checkpoint.
public enum KVCacheBranchingError: Error, Equatable, CustomStringConvertible {
    /// The cache's concrete class isn't on `branchCaches`'s allowlist.
    /// Carries the runtime type name so a caller can log/triage without a
    /// debugger.
    case unsupportedCacheClass(String)

    /// `cache.copy()` produced a result that fails the structural parity
    /// guard described on `branchCaches`'s doc comment. Carries a
    /// human-readable description of which check failed.
    case parityViolation(String)

    public var description: String {
        switch self {
        case .unsupportedCacheClass(let className):
            return "KVCacheBranchingError.unsupportedCacheClass(\(className))"
        case .parityViolation(let detail):
            return "KVCacheBranchingError.parityViolation(\(detail))"
        }
    }
}

/// Branches (forks) a live `[any KVCache]` array for a durable Infinite
/// session checkpoint.
///
/// Every element must be one of the concrete `KVCache` classes mlx-swift-lm
/// ships as of this writing: `KVCacheSimple`, `QuantizedKVCache`,
/// `RotatingKVCache`, `MambaCache`, `ArraysCache`, `ChunkedKVCache`,
/// `CacheList`. Anything else throws `.unsupportedCacheClass` rather than
/// silently forking through the default `BaseKVCache.copy()` (which
/// `fatalError`s) or aliasing shared `MLXArray` storage between the live
/// session and the persisted branch.
///
/// `copy()` on the current mlx-swift-lm snapshots state via `$0[.ellipsis]`
/// slices (KVCache.swift:502/1075/1234/1404) rather than returning the
/// original arrays. The parity guard below re-verifies the cheap structural
/// half of that contract after every `copy()` call (same dynamic type, same
/// `offset`, same `state` array count and shapes, distinct `MLXArray`
/// object identity per element) so an upstream regression fails loud at
/// fork time. Scope caveat: because `state` getters may mint fresh wrapper
/// objects on every call, the identity check cannot prove buffer-level
/// independence; that stronger property is proven behaviorally, by the
/// mutate-branch-then-compare unit test here and the live bit-exactness
/// gate in the turn-commit phase. This mirrors the Python reference's
/// `_assert_field_parity` (infinite repo, `d1_cache/turns.py`).
public func branchCaches(_ caches: [any KVCache]) throws -> [any KVCache] {
    try caches.map(branchCache)
}

private func branchCache(_ cache: any KVCache) throws -> any KVCache {
    try requireAllowlisted(cache)
    let branched = cache.copy()
    try assertParity(original: cache, branched: branched)
    return branched
}

private func requireAllowlisted(_ cache: any KVCache) throws {
    switch cache {
    case is KVCacheSimple, is QuantizedKVCache, is RotatingKVCache, is MambaCache,
        is ArraysCache, is ChunkedKVCache, is CacheList:
        return
    default:
        throw KVCacheBranchingError.unsupportedCacheClass(String(describing: type(of: cache)))
    }
}

private func assertParity(original: any KVCache, branched: any KVCache) throws {
    guard type(of: branched) == type(of: original) else {
        throw KVCacheBranchingError.parityViolation(
            "type mismatch: expected \(type(of: original)), got \(type(of: branched))"
        )
    }
    guard branched.offset == original.offset else {
        throw KVCacheBranchingError.parityViolation(
            "offset mismatch: expected \(original.offset), got \(branched.offset)"
        )
    }

    let originalState = original.state
    let branchedState = branched.state
    guard originalState.count == branchedState.count else {
        throw KVCacheBranchingError.parityViolation(
            "state array count mismatch: expected \(originalState.count), got \(branchedState.count)"
        )
    }

    for (index, arrays) in zip(originalState, branchedState).enumerated() {
        let (originalArray, branchedArray) = arrays
        guard originalArray.shape == branchedArray.shape else {
            throw KVCacheBranchingError.parityViolation(
                "state[\(index)] shape mismatch: expected \(originalArray.shape), got \(branchedArray.shape)"
            )
        }
        guard originalArray !== branchedArray else {
            throw KVCacheBranchingError.parityViolation(
                "state[\(index)] is the same MLXArray instance as the original; " +
                    "copy() did not produce an independent snapshot"
            )
        }
    }

    // ArraysCache/MambaCache carry batch-generation transients
    // (`leftPadding`, `lengths`) that must not leak into a persisted
    // branch. `leftPadding` has no public accessor on mlx-swift-lm as of
    // this writing; `currentLengths` is the one transient this guard can
    // reach from outside the module.
    if let arraysCacheBranch = branched as? ArraysCache, arraysCacheBranch.currentLengths != nil {
        throw KVCacheBranchingError.parityViolation(
            "ArraysCache/MambaCache branch has a non-nil currentLengths batch transient; " +
                "finalize() should have cleared it before checkpoint"
        )
    }
}
