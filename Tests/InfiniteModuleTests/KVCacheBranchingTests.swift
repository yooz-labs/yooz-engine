// KVCacheBranchingTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import MLX
import MLXLMCommon
import XCTest
@testable import InfiniteModule

final class KVCacheBranchingTests: XCTestCase {

    // MARK: - KVCacheSimple

    func testBranchKVCacheSimpleAfterOneUpdate() throws {
        let cache = KVCacheSimple()
        let (keys, values) = randomKV()
        _ = cache.update(keys: keys, values: values)

        let branched = try branchCaches([cache])
        XCTAssertEqual(branched.count, 1)
        let branchedCache = branched[0]

        XCTAssertTrue(branchedCache is KVCacheSimple)
        XCTAssertTrue((branchedCache as AnyObject) !== (cache as AnyObject))
        XCTAssertEqual(branchedCache.offset, cache.offset)
        assertDistinctEqualShape(cache.state, branchedCache.state)
    }

    /// CPU-cheap precursor of the live bit-exactness gate: mutating the
    /// branch must never perturb the original's cache bytes. If a future
    /// mlx-swift-lm regression made `copy()` alias the original's storage
    /// instead of slicing it, this would fail (as would the `!==` parity
    /// guard inside `branchCaches` itself).
    func testMutatingBranchLeavesOriginalKVCacheSimpleUnaffected() throws {
        let cache = KVCacheSimple()
        let (keys, values) = randomKV()
        _ = cache.update(keys: keys, values: values)
        let originalOffset = cache.offset
        let originalState = cache.state

        let branched = try branchCaches([cache])[0]
        let (moreKeys, moreValues) = randomKV(length: 1)
        _ = branched.update(keys: moreKeys, values: moreValues)

        XCTAssertEqual(cache.offset, originalOffset)
        XCTAssertNotEqual(branched.offset, cache.offset)
        for (before, after) in zip(originalState, cache.state) {
            XCTAssertTrue(before.arrayEqual(after).item(Bool.self))
        }
    }

    // MARK: - QuantizedKVCache

    func testBranchQuantizedKVCache() throws {
        let simple = KVCacheSimple()
        let (keys, values) = randomKV(heads: 2, length: 8, dim: 64)
        _ = simple.update(keys: keys, values: values)
        let quantized = simple.toQuantized(groupSize: 64, bits: 4)

        let branched = try branchCaches([quantized])[0]

        XCTAssertTrue(branched is QuantizedKVCache)
        XCTAssertEqual(branched.offset, quantized.offset)
        assertDistinctEqualShape(quantized.state, branched.state)
    }

    // MARK: - RotatingKVCache

    func testBranchRotatingKVCache() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0, step: 4)
        let (keys, values) = randomKV(length: 4)
        _ = cache.update(keys: keys, values: values)

        let branched = try branchCaches([cache])[0]

        XCTAssertTrue(branched is RotatingKVCache)
        XCTAssertEqual(branched.offset, cache.offset)
        assertDistinctEqualShape(cache.state, branched.state)
    }

    // MARK: - ChunkedKVCache

    func testBranchChunkedKVCache() throws {
        let cache = ChunkedKVCache(chunkSize: 8)
        let (keys, values) = randomKV(length: 4)
        _ = cache.update(keys: keys, values: values)

        let branched = try branchCaches([cache])[0]

        XCTAssertTrue(branched is ChunkedKVCache)
        XCTAssertEqual(branched.offset, cache.offset)
        assertDistinctEqualShape(cache.state, branched.state)
    }

    // MARK: - ArraysCache

    func testBranchArraysCache() throws {
        let cache = ArraysCache(size: 2)
        cache[0] = MLXRandom.uniform(0 ..< 1, [1, 4])
        cache[1] = MLXRandom.uniform(0 ..< 1, [1, 4])

        let branched = try branchCaches([cache])[0]

        XCTAssertTrue(type(of: branched) == ArraysCache.self)
        assertDistinctEqualShape(cache.state, branched.state)
        XCTAssertNil((branched as? ArraysCache)?.currentLengths)
    }

    // MARK: - MambaCache

    func testBranchMambaCacheWithTwoStateArrays() throws {
        let cache = MambaCache()
        cache[0] = MLXRandom.uniform(0 ..< 1, [1, 16])
        cache[1] = MLXRandom.uniform(0 ..< 1, [1, 16])
        cache.offset = 3

        let branched = try branchCaches([cache])[0]

        XCTAssertTrue(branched is MambaCache)
        XCTAssertEqual(branched.offset, 3)
        assertDistinctEqualShape(cache.state, branched.state)
        XCTAssertNil((branched as? ArraysCache)?.currentLengths)
    }

    // MARK: - CacheList

    func testBranchCacheList() throws {
        let childA = KVCacheSimple()
        let (keysA, valuesA) = randomKV(length: 4)
        _ = childA.update(keys: keysA, values: valuesA)

        let childB = KVCacheSimple()
        let (keysB, valuesB) = randomKV(length: 2)
        _ = childB.update(keys: keysB, values: valuesB)

        let cacheList = CacheList(childA, childB)

        let branched = try branchCaches([cacheList])[0]

        XCTAssertTrue(branched is CacheList)
        XCTAssertEqual(branched.offset, cacheList.offset)
        assertDistinctEqualShape(cacheList.state, branched.state)
    }

    // MARK: - Unsupported cache class

    func testUnsupportedCacheClassThrows() {
        let dummy = DummyUnsupportedCache()

        XCTAssertThrowsError(try branchCaches([dummy])) { error in
            XCTAssertEqual(
                error as? KVCacheBranchingError,
                .unsupportedCacheClass(String(describing: DummyUnsupportedCache.self))
            )
        }
    }

    // MARK: - Helpers

    private func randomKV(
        batch: Int = 1,
        heads: Int = 2,
        length: Int = 4,
        dim: Int = 8
    ) -> (MLXArray, MLXArray) {
        let keys = MLXRandom.uniform(0 ..< 1, [batch, heads, length, dim])
        let values = MLXRandom.uniform(0 ..< 1, [batch, heads, length, dim])
        return (keys, values)
    }

    private func assertDistinctEqualShape(
        _ original: [MLXArray],
        _ branched: [MLXArray],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(original.count, branched.count, file: file, line: line)
        for (originalArray, branchedArray) in zip(original, branched) {
            XCTAssertEqual(originalArray.shape, branchedArray.shape, file: file, line: line)
            XCTAssertTrue(
                originalArray !== branchedArray,
                "branched array must be a distinct MLXArray instance from the original",
                file: file,
                line: line
            )
        }
    }
}

/// Real, minimal `KVCache` conformer that isn't on `branchCaches`'s
/// allowlist — exercises the `.unsupportedCacheClass` throw path with an
/// actual type rather than a mock.
private final class DummyUnsupportedCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }

    func innerState() -> [MLXArray] { state }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    var state: [MLXArray] = []
    var metaState: [String] = [""]
    var isTrimmable: Bool { false }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        DummyUnsupportedCache()
    }

    func prepare(lengths: [Int]?) {}
    func prepare(lengths: MLXArray?) {}
}
