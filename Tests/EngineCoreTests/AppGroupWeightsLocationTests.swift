// AppGroupWeightsLocationTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Coverage for engine#227's app-group weights wiring. The container-resolving
// half (`redirectHuggingFaceCache`) needs a real `application-groups`
// entitlement + provisioning profile to succeed, which a plain `swift test` /
// `xcodebuild test` process never has — so its no-op (never crash) contract
// is what's pinned here. The pure path-construction half
// (`huggingFaceHubCacheURL(inContainer:)`) is fully unit-testable and pins
// the "Application Support, not Caches" contract from the packaging doc.

import XCTest
@testable import EngineCore

final class AppGroupWeightsLocationTests: XCTestCase {

    /// The hub-cache path must land under Application Support, never
    /// Caches — the OS purges container Caches under disk pressure, which
    /// would force a multi-GB re-download of STT/LLM weights.
    func testHuggingFaceHubCacheURLUsesApplicationSupportNotCaches() {
        let container = URL(fileURLWithPath: "/tmp/fake-app-group-container", isDirectory: true)
        let hubCache = AppGroupWeightsLocation.huggingFaceHubCacheURL(inContainer: container)

        XCTAssertTrue(hubCache.path.contains("/Application Support/"))
        XCTAssertFalse(hubCache.path.contains("/Caches/"))
        XCTAssertTrue(hubCache.path.hasSuffix("/YoozEngine/huggingface/hub"))
        XCTAssertTrue(hubCache.path.hasPrefix(container.path))
    }

    /// Path construction is a pure function of the container URL — two
    /// different containers must never collide on the same hub-cache path
    /// (that would defeat the whole point of per-app-group isolation).
    func testHuggingFaceHubCacheURLIsDeterministicPerContainer() {
        let containerA = URL(fileURLWithPath: "/tmp/group-a", isDirectory: true)
        let containerB = URL(fileURLWithPath: "/tmp/group-b", isDirectory: true)

        let cacheA = AppGroupWeightsLocation.huggingFaceHubCacheURL(inContainer: containerA)
        let cacheB = AppGroupWeightsLocation.huggingFaceHubCacheURL(inContainer: containerB)

        XCTAssertNotEqual(cacheA, cacheB)
        XCTAssertEqual(
            AppGroupWeightsLocation.huggingFaceHubCacheURL(inContainer: containerA),
            cacheA,
            "path construction must be deterministic for the same container"
        )
    }

    /// An empty group id is always a no-op — never attempt to resolve a
    /// blank identifier (which would otherwise be a confusing failure mode
    /// distinct from "entitlement missing").
    func testEmptyGroupIdentifierIsANoOp() {
        XCTAssertFalse(AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: ""))
    }

    /// `containerURL(forSecurityApplicationGroupIdentifier:)` doesn't
    /// require the `application-groups` entitlement to resolve SOMETHING
    /// when the calling process is unsandboxed (this test process, like a
    /// consumer app's local unsandboxed dev run, is not sandboxed at all —
    /// macOS only enforces group-entitlement matching for a truly sandboxed
    /// caller). So this asserts the OBSERVABLE CONTRACT rather than a fixed
    /// true/false outcome, keeping it stable across sandboxed CI and
    /// unsandboxed local runs: on success, `HF_HUB_CACHE` must point under
    /// "Application Support" (never "Caches"); on failure, `HF_HUB_CACHE`
    /// must be left exactly as it was (never a crash either way).
    func testRedirectHonorsItsDocumentedEnvVarContract() {
        let priorValue = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
        defer {
            if let priorValue {
                setenv("HF_HUB_CACHE", priorValue, 1)
            } else {
                unsetenv("HF_HUB_CACHE")
            }
        }
        unsetenv("HF_HUB_CACHE")

        let groupID = "engine-core-tests.app-group-contract-probe"
        let succeeded = AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: groupID)

        guard succeeded else {
            XCTAssertNil(ProcessInfo.processInfo.environment["HF_HUB_CACHE"])
            return
        }
        let hubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
        XCTAssertNotNil(hubCache)
        XCTAssertTrue(hubCache?.contains("Application Support") ?? false)
        XCTAssertFalse(hubCache?.contains("/Caches/") ?? true)
        // Clean up the directory this run created on disk.
        if let hubCache {
            try? FileManager.default.removeItem(atPath: hubCache)
        }
    }
}
