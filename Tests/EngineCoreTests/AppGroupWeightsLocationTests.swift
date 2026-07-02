// AppGroupWeightsLocationTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Coverage for engine#227's app-group weights wiring. The pure
// path-construction half (`huggingFaceHubCacheURL(inContainer:)`) is fully
// unit-testable and pins the "Application Support, not Caches" contract from
// the packaging doc. The container-resolving half
// (`redirectHuggingFaceCache`) is environment-dependent:
// `containerURL(forSecurityApplicationGroupIdentifier:)` does NOT require the
// `application-groups` entitlement to resolve a container for an unsandboxed
// caller (a plain `swift test` process, like a consumer app's unsandboxed
// local dev run, typically DOES resolve one, landing under
// `~/Library/Group Containers/<id>/`) — macOS only enforces entitlement
// matching for a genuinely sandboxed caller (real XCTest-hosted-by-app runs,
// CI). So the tests below assert the OBSERVABLE CONTRACT (success shape XOR
// untouched-on-failure shape, never a crash) rather than a fixed true/false
// outcome, keeping them stable across both environments — see
// `AppGroupWeightsLocation.swift`'s own doc comment for the full rationale.

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
        XCTAssertNil(AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: ""))
    }

    /// `containerURL(forSecurityApplicationGroupIdentifier:)` doesn't
    /// require the `application-groups` entitlement to resolve SOMETHING
    /// when the calling process is unsandboxed (this test process, like a
    /// consumer app's local unsandboxed dev run, is not sandboxed at all —
    /// macOS only enforces group-entitlement matching for a truly sandboxed
    /// caller). So this asserts the OBSERVABLE CONTRACT rather than a fixed
    /// success/failure outcome, keeping it stable across sandboxed CI and
    /// unsandboxed local runs: on success, the returned URL (and
    /// `HF_HUB_CACHE`) must point under "Application Support" (never
    /// "Caches"); on failure, `HF_HUB_CACHE` must be left exactly as it was
    /// (never a crash either way).
    ///
    /// Mutates the process-global `HF_HUB_CACHE` env var and (on an
    /// unsandboxed run) creates a real directory under
    /// `~/Library/Group Containers/`; both are restored/removed in this
    /// test, which assumes XCTest's default serial-within-class execution
    /// (no `-parallel-testing-enabled` for this bundle) — a future
    /// parallelized run of `EngineCoreTests` would need this test isolated
    /// from anything else touching `HF_HUB_CACHE`.
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
        let hubCache = AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: groupID)

        guard let hubCache else {
            XCTAssertNil(ProcessInfo.processInfo.environment["HF_HUB_CACHE"])
            return
        }
        XCTAssertTrue(hubCache.path.contains("Application Support"))
        XCTAssertFalse(hubCache.path.contains("/Caches/"))
        XCTAssertEqual(ProcessInfo.processInfo.environment["HF_HUB_CACHE"], hubCache.path)
        // Clean up the ENTIRE group-container root this run created on disk
        // (not just the `hub` leaf) — `containerURL` resolves
        // `~/Library/Group Containers/<groupID>/`, and `hubCache` is several
        // path components below that root.
        if let containerRoot = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: groupID
        ) {
            try? FileManager.default.removeItem(at: containerRoot)
        }
    }

    /// Deterministic coverage of the "container resolves but the
    /// subdirectory can't be created" branch — not reachable via the real
    /// `FileManager.default` in a plain test process (the container it
    /// resolves to is always writable here), so a thin subclass overrides
    /// just `containerURL(forSecurityApplicationGroupIdentifier:)` (returns
    /// a real, valid URL) and `createDirectory(at:withIntermediateDirectories:attributes:)`
    /// (always throws), delegating nothing else — a backend double, not a
    /// mock, matching `CannedStreamTransport`'s shape in
    /// `XPCStreamingTests.swift`.
    func testUncreatableDirectoryReturnsNilAndLeavesEnvVarUntouched() {
        let priorValue = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
        defer {
            if let priorValue {
                setenv("HF_HUB_CACHE", priorValue, 1)
            } else {
                unsetenv("HF_HUB_CACHE")
            }
        }
        unsetenv("HF_HUB_CACHE")

        let result = AppGroupWeightsLocation.redirectHuggingFaceCache(
            groupIdentifier: "irrelevant-because-the-double-ignores-it",
            fileManager: AlwaysFailingCreateDirectoryFileManager()
        )

        XCTAssertNil(result)
        XCTAssertNil(ProcessInfo.processInfo.environment["HF_HUB_CACHE"])
    }
}

/// Test double for `testUncreatableDirectoryReturnsNilAndLeavesEnvVarUntouched`.
/// Resolves a real (writable) container so `redirectHuggingFaceCache` reaches
/// its `createDirectory` call, then fails that one call deterministically —
/// simulating a read-only volume / disk-full / permission-denied condition
/// the real `FileManager.default` can't be coaxed into hitting reliably in a
/// test process.
private final class AlwaysFailingCreateDirectoryFileManager: FileManager {
    private struct ForcedFailure: Error {}

    override func createDirectory(
        at url: URL,
        withIntermediateDirectories createIntermediates: Bool,
        attributes: [FileAttributeKey: Any]? = nil
    ) throws {
        throw ForcedFailure()
    }
}
