// MLXResidency.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// A category of MLX-backed model that can be resident at the same time as
/// other categories. One model per category is the invariant each module
/// already enforces (`TouchUpEngine.evictModelsExcept` for `.touchUp`,
/// `YoozSTTEngine` for `.stt`); categories coexist.
public enum MLXModelCategory: Hashable, Sendable {
    /// Speech-to-text (Parakeet / FastConformer / Qwen3-ASR).
    case stt
    /// Text touch-up / LLM (Yooz Light, Yooz Quality).
    case touchUp
    // Extensible: add `.coding`, etc. as future MLX categories ship.
}

/// How a caller should set MLX's process-global buffer-cache knobs after a
/// residency change. Returned by `MLXResidency.register` / `unregister` so the
/// MLX-importing call site — the only place allowed to touch `MLX.Memory` —
/// applies it. Keeping the mutation out of this type is what lets `EngineCore`
/// stay MLX-free and lets the logic be unit-tested under a plain `swift test`
/// run where `default.metallib` is absent.
public struct MLXResidencyDirective: Equatable, Sendable {
    /// Target value for `MLX.Memory.cacheLimit`.
    public let cacheLimitBytes: Int
    /// Whether the caller should also call `MLX.Memory.clearCache()`. True only
    /// when the last resident category was removed — never while another
    /// category is still resident, so a teardown can't flush a coexisting
    /// model's warm buffers.
    public let flush: Bool
}

/// Process-wide tracker of which MLX model categories are resident, and the
/// single source of truth for the global MLX buffer-cache budget.
///
/// Why this exists: the per-model 512 MB cache budget was written to the
/// process-global `MLX.Memory.cacheLimit` unilaterally by whichever model
/// loaded last, and `MLX.Memory.clearCache()` was called on any one category's
/// teardown. With two MLX categories resident (e.g. Parakeet STT + Quality
/// LLM) they shared one 512 MB pool and flushed each other's warm buffers every
/// cycle — the root cause of the ~10x slowdown versus the Apple-STT path (Apple
/// STT is not MLX, so the LLM cache stayed warm). This coordinator makes both
/// global knobs derive from the *set* of resident categories: the cache budget
/// is `residentCategoryCount * perCategoryBudget`, and a flush happens only when
/// the resident set becomes empty.
///
/// Reference-counted per category (not a plain set) on purpose: a TouchUp tier
/// switch loads the new tier before evicting the old one, so two
/// `MLXLLMBackend` instances map to `.touchUp` at once. A set would let the
/// old tier's unload drop `.touchUp` while the new tier is resident, wrongly
/// flushing its buffers; the refcount keeps the category resident until the
/// last model in it unloads.
///
/// Thread-safe via `NSLock` (matching the engine's concurrency style).
/// `register` / `unregister` are synchronous so load/teardown paths in actors
/// and plain classes alike can call them without `await` coloring.
public final class MLXResidency: @unchecked Sendable {
    /// Shared process-wide instance used by the real load/teardown paths.
    public static let shared = MLXResidency()

    private let lock = NSLock()

    /// Live model count per category. Categories with a zero count are pruned,
    /// so `counts.count` is exactly the number of resident categories.
    private var counts: [MLXModelCategory: Int] = [:]

    /// Public so tests can use an isolated instance instead of mutating
    /// `shared`.
    public init() {}

    /// The global cache budget for `count` resident categories. Pure (no Metal)
    /// and exposed for tests. Floors at one category's budget so the limit is
    /// never set below a single model's scratch.
    public static func budgetBytes(forResidentCount count: Int) -> Int {
        max(1, count) * EngineConfig.mlxCacheBudgetPerCategoryBytes
    }

    /// Mark a model in `category` resident. Idempotent at the category level for
    /// the budget (a second model in the same category does not grow it). Never
    /// flushes — a load only grows the budget. Returns the directive to apply.
    @discardableResult
    public func register(_ category: MLXModelCategory) -> MLXResidencyDirective {
        lock.lock()
        defer { lock.unlock() }
        counts[category, default: 0] += 1
        return MLXResidencyDirective(
            cacheLimitBytes: Self.budgetBytes(forResidentCount: counts.count),
            flush: false
        )
    }

    /// Mark a model in `category` no longer resident. The category leaves the
    /// resident set only when its last model unloads. Flushes only when the
    /// resident set is then empty — never while another category is resident.
    /// Over-balanced calls (unregister with no prior register) are ignored.
    @discardableResult
    public func unregister(_ category: MLXModelCategory) -> MLXResidencyDirective {
        lock.lock()
        defer { lock.unlock() }
        if let current = counts[category] {
            if current <= 1 {
                counts[category] = nil
            } else {
                counts[category] = current - 1
            }
        }
        return MLXResidencyDirective(
            cacheLimitBytes: Self.budgetBytes(forResidentCount: counts.count),
            flush: counts.isEmpty
        )
    }

    /// The cache budget for the current resident set. Used by the live
    /// in-process memory test to assert the global `MLX.Memory.cacheLimit`
    /// matches what this coordinator computed.
    public func currentCacheLimitBytes() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return Self.budgetBytes(forResidentCount: counts.count)
    }

    /// Whether any model in `category` is currently resident. Test / diagnostic
    /// helper.
    public func isResident(_ category: MLXModelCategory) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return (counts[category] ?? 0) > 0
    }
}
