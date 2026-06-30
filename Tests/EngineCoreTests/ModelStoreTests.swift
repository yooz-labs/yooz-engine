// ModelStoreTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Real-filesystem coverage for the disk-hygiene engine. Every test builds an
// actual HuggingFace-hub-shaped directory on disk — blobs as real files,
// snapshot entries as real symlinks into `../../blobs/<etag>`, `refs/main`
// holding a commit hash — because the whole point of `ModelStore` is that it
// reasons about that symlink/blob structure. No mocks.

import XCTest
@testable import EngineCore

final class ModelStoreTests: XCTestCase {
    private var root: URL!
    private var hub: URL!
    private var modelsDir: URL!
    private let fm = FileManager.default

    override func setUpWithError() throws {
        root = fm.temporaryDirectory
            .appendingPathComponent("ModelStoreTests-\(UUID().uuidString)")
        hub = root.appendingPathComponent("hub")
        modelsDir = root.appendingPathComponent("models")
        try fm.createDirectory(at: hub, withIntermediateDirectories: true)
        try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? fm.removeItem(at: root)
    }

    private func makeStore() -> ModelStore {
        ModelStore(hubCacheDirectory: hub, modelsDirectory: modelsDir)
    }

    // MARK: - Fixture builder

    /// Writes a blob of `byteCount` bytes into `<repo>/blobs/<name>`.
    @discardableResult
    private func writeBlob(repo: URL, name: String, byteCount: Int) throws -> URL {
        let blobs = repo.appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        let url = blobs.appendingPathComponent(name)
        try Data(count: byteCount).write(to: url)
        return url
    }

    /// Creates `<repo>/snapshots/<commit>/<filename>` as a symlink to
    /// `../../blobs/<blob>`, exactly as swift-huggingface's `HubCache` does.
    private func linkSnapshot(
        repo: URL, commit: String, filename: String, toBlob blob: String
    ) throws {
        let dir = repo.appendingPathComponent("snapshots").appendingPathComponent(commit)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try fm.createSymbolicLink(
            atPath: dir.appendingPathComponent(filename).path,
            withDestinationPath: "../../blobs/\(blob)"
        )
    }

    private func writeRef(repo: URL, ref: String, commit: String) throws {
        let refs = repo.appendingPathComponent("refs")
        try fm.createDirectory(at: refs, withIntermediateDirectories: true)
        try commit.write(
            to: refs.appendingPathComponent(ref), atomically: true, encoding: .utf8
        )
    }

    private func exists(_ url: URL) -> Bool { fm.fileExists(atPath: url.path) }

    // MARK: - Naming helper

    func testHubRepoDirNameMatchesHubCacheConvention() {
        XCTAssertEqual(
            ModelCacheDescriptor.hubRepoDirName(forHuggingFaceID: "YoozLabs/Yooz-Quality-v2"),
            "models--YoozLabs--Yooz-Quality-v2"
        )
        XCTAssertNil(ModelCacheDescriptor.hubRepoDirName(forHuggingFaceID: "  "))
        XCTAssertNil(ModelCacheDescriptor.hubRepoDirName(forHuggingFaceID: "/bad"))
    }

    // MARK: - Size

    func testOnDiskSizeCountsBlobsOnceAndSkipsSymlinks() async throws {
        // Two snapshots share blobA; each symlinks it. A naive recursive sum that
        // followed symlinks would count blobA three times (blob + 2 links).
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "blobA", byteCount: 8_192)
        try writeBlob(repo: repo, name: "blobB", byteCount: 16_384)
        try linkSnapshot(repo: repo, commit: "aaa", filename: "config.json", toBlob: "blobA")
        try linkSnapshot(repo: repo, commit: "bbb", filename: "config.json", toBlob: "blobA")
        try linkSnapshot(repo: repo, commit: "bbb", filename: "model.bin", toBlob: "blobB")
        try writeRef(repo: repo, ref: "main", commit: "bbb")

        let store = makeStore()
        let size = await store.onDiskSize(
            hfRepoDirName: "models--Acme--Demo", modelsDirSubdir: nil
        )

        // Expected = blobA + blobB + refs/main, each counted once. Recomputed
        // independently here to pin "blobs once, symlinks skipped".
        let expected = allocated(repo.appendingPathComponent("blobs/blobA"))
            + allocated(repo.appendingPathComponent("blobs/blobB"))
            + allocated(repo.appendingPathComponent("refs/main"))
        XCTAssertEqual(size, expected)
    }

    func testOnDiskSizeIncludesModelsDirectoryCopy() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "blobA", byteCount: 8_192)

        let copy = modelsDir.appendingPathComponent("demo")
        try fm.createDirectory(at: copy, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: copy.appendingPathComponent("config.json"))

        let store = makeStore()
        let size = await store.onDiskSize(
            hfRepoDirName: "models--Acme--Demo", modelsDirSubdir: "demo"
        )
        let expected = allocated(repo.appendingPathComponent("blobs/blobA"))
            + allocated(copy.appendingPathComponent("config.json"))
        XCTAssertEqual(size, expected)
    }

    func testOnDiskSizeIsZeroForAbsentModel() async throws {
        let store = makeStore()
        let size = await store.onDiskSize(
            hfRepoDirName: "models--None--Here", modelsDirSubdir: "nope"
        )
        XCTAssertEqual(size, 0)
    }

    // MARK: - Collapse

    func testCollapseDeletesSupersededSnapshotAndOrphanBlobs() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "shared", byteCount: 8_192)   // both snapshots
        try writeBlob(repo: repo, name: "old", byteCount: 16_384)     // old only -> orphan
        try writeBlob(repo: repo, name: "new", byteCount: 16_384)     // new only -> kept
        try linkSnapshot(repo: repo, commit: "old111", filename: "config.json", toBlob: "shared")
        try linkSnapshot(repo: repo, commit: "old111", filename: "model.bin", toBlob: "old")
        try linkSnapshot(repo: repo, commit: "new222", filename: "config.json", toBlob: "shared")
        try linkSnapshot(repo: repo, commit: "new222", filename: "model.bin", toBlob: "new")
        try writeRef(repo: repo, ref: "main", commit: "new222")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")

        XCTAssertGreaterThan(reclaimed, 0)
        // Superseded snapshot gone, live snapshot kept.
        XCTAssertFalse(exists(repo.appendingPathComponent("snapshots/old111")))
        XCTAssertTrue(exists(repo.appendingPathComponent("snapshots/new222")))
        // Orphan blob GC'd; shared + live blobs survive.
        XCTAssertFalse(exists(repo.appendingPathComponent("blobs/old")))
        XCTAssertTrue(exists(repo.appendingPathComponent("blobs/shared")))
        XCTAssertTrue(exists(repo.appendingPathComponent("blobs/new")))
        // The live snapshot's symlinks still resolve.
        let liveConfig = repo.appendingPathComponent("snapshots/new222/config.json")
        XCTAssertEqual(try Data(contentsOf: liveConfig).count, 8_192)
    }

    func testCollapseIsIdempotent() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "shared", byteCount: 8_192)
        try writeBlob(repo: repo, name: "old", byteCount: 16_384)
        try writeBlob(repo: repo, name: "new", byteCount: 16_384)
        try linkSnapshot(repo: repo, commit: "old111", filename: "m", toBlob: "old")
        try linkSnapshot(repo: repo, commit: "new222", filename: "m", toBlob: "new")
        try linkSnapshot(repo: repo, commit: "new222", filename: "c", toBlob: "shared")
        try writeRef(repo: repo, ref: "main", commit: "new222")

        let store = makeStore()
        _ = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        let second = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(second, 0)
    }

    func testCollapseWithoutRefIsNoOp() async throws {
        // No refs file -> we can't prove which snapshot is live -> keep everything.
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "a", byteCount: 8_192)
        try writeBlob(repo: repo, name: "b", byteCount: 8_192)
        try linkSnapshot(repo: repo, commit: "c1", filename: "m", toBlob: "a")
        try linkSnapshot(repo: repo, commit: "c2", filename: "m", toBlob: "b")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(reclaimed, 0)
        XCTAssertTrue(exists(repo.appendingPathComponent("snapshots/c1")))
        XCTAssertTrue(exists(repo.appendingPathComponent("snapshots/c2")))
    }

    func testCollapsePreservesIncompleteBlobs() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "live", byteCount: 8_192)
        try writeBlob(repo: repo, name: "partial.incomplete", byteCount: 4_096)
        try linkSnapshot(repo: repo, commit: "c1", filename: "m", toBlob: "live")
        try writeRef(repo: repo, ref: "main", commit: "c1")

        let store = makeStore()
        _ = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        // In-flight partial download is never reaped.
        XCTAssertTrue(exists(repo.appendingPathComponent("blobs/partial.incomplete")))
        XCTAssertTrue(exists(repo.appendingPathComponent("blobs/live")))
    }

    // MARK: - Delete

    func testDeleteModelRemovesHubAndModelsDirCopies() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "blobA", byteCount: 32_768)
        try linkSnapshot(repo: repo, commit: "c1", filename: "m", toBlob: "blobA")
        try writeRef(repo: repo, ref: "main", commit: "c1")
        let copy = modelsDir.appendingPathComponent("demo")
        try fm.createDirectory(at: copy, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: copy.appendingPathComponent("config.json"))

        let store = makeStore()
        let reclaimed = try await store.deleteModel(
            hfRepoDirName: "models--Acme--Demo", modelsDirSubdir: "demo"
        )

        XCTAssertGreaterThan(reclaimed, 0)
        XCTAssertFalse(exists(repo))
        XCTAssertFalse(exists(copy))
    }

    // MARK: - Cleanup migration

    func testCleanupCollapsesAllReposAndDedupesDuplicates() async throws {
        // Repo 1: stacked snapshots -> collapsed.
        let stacked = hub.appendingPathComponent("models--Acme--Stacked")
        try writeBlob(repo: stacked, name: "shared", byteCount: 8_192)
        try writeBlob(repo: stacked, name: "old", byteCount: 16_384)
        try writeBlob(repo: stacked, name: "new", byteCount: 16_384)
        try linkSnapshot(repo: stacked, commit: "o", filename: "m", toBlob: "old")
        try linkSnapshot(repo: stacked, commit: "n", filename: "m", toBlob: "new")
        try linkSnapshot(repo: stacked, commit: "n", filename: "c", toBlob: "shared")
        try writeRef(repo: stacked, ref: "main", commit: "n")

        // Repo 2: bundled duplicate -> whole hub repo removed.
        let bundled = hub.appendingPathComponent("models--Acme--Bundled")
        try writeBlob(repo: bundled, name: "b", byteCount: 32_768)
        try linkSnapshot(repo: bundled, commit: "c1", filename: "m", toBlob: "b")
        try writeRef(repo: bundled, ref: "main", commit: "c1")

        // Repo 3: modelsDirectory copy (complete) -> hub repo removed.
        let mirrored = hub.appendingPathComponent("models--Acme--Mirrored")
        try writeBlob(repo: mirrored, name: "b", byteCount: 32_768)
        try linkSnapshot(repo: mirrored, commit: "c1", filename: "m", toBlob: "b")
        try writeRef(repo: mirrored, ref: "main", commit: "c1")
        let mirroredCopy = modelsDir.appendingPathComponent("mirrored")
        try fm.createDirectory(at: mirroredCopy, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: mirroredCopy.appendingPathComponent("config.json"))

        // Repo 4: no higher-priority copy -> kept.
        let kept = hub.appendingPathComponent("models--Acme--Kept")
        try writeBlob(repo: kept, name: "b", byteCount: 32_768)
        try linkSnapshot(repo: kept, commit: "c1", filename: "m", toBlob: "b")
        try writeRef(repo: kept, ref: "main", commit: "c1")

        let descriptors = [
            ModelCacheDescriptor(
                id: "bundled", module: "llm",
                hfRepoDirName: "models--Acme--Bundled",
                modelsDirSubdir: nil, isBundled: true
            ),
            ModelCacheDescriptor(
                id: "mirrored", module: "stt",
                hfRepoDirName: "models--Acme--Mirrored",
                modelsDirSubdir: "mirrored", isBundled: false
            ),
            ModelCacheDescriptor(
                id: "kept", module: "stt",
                hfRepoDirName: "models--Acme--Kept",
                modelsDirSubdir: "absent", isBundled: false
            ),
        ]

        let store = makeStore()
        let report = try await store.cleanupAll(descriptors: descriptors)

        XCTAssertGreaterThan(report.totalReclaimedBytes, 0)
        // Stacked collapsed: orphan gone, repo retained.
        XCTAssertFalse(exists(stacked.appendingPathComponent("blobs/old")))
        XCTAssertTrue(exists(stacked.appendingPathComponent("blobs/new")))
        // Duplicates removed.
        XCTAssertFalse(exists(bundled))
        XCTAssertFalse(exists(mirrored))
        // No-higher-priority repo kept.
        XCTAssertTrue(exists(kept))
    }

    func testCleanupIsIdempotent() async throws {
        let stacked = hub.appendingPathComponent("models--Acme--Stacked")
        try writeBlob(repo: stacked, name: "old", byteCount: 16_384)
        try writeBlob(repo: stacked, name: "new", byteCount: 16_384)
        try linkSnapshot(repo: stacked, commit: "o", filename: "m", toBlob: "old")
        try linkSnapshot(repo: stacked, commit: "n", filename: "m", toBlob: "new")
        try writeRef(repo: stacked, ref: "main", commit: "n")

        let bundled = hub.appendingPathComponent("models--Acme--Bundled")
        try writeBlob(repo: bundled, name: "b", byteCount: 32_768)
        try linkSnapshot(repo: bundled, commit: "c1", filename: "m", toBlob: "b")
        try writeRef(repo: bundled, ref: "main", commit: "c1")

        let descriptors = [
            ModelCacheDescriptor(
                id: "bundled", module: "llm",
                hfRepoDirName: "models--Acme--Bundled",
                modelsDirSubdir: nil, isBundled: true
            )
        ]

        let store = makeStore()
        let first = try await store.cleanupAll(descriptors: descriptors)
        XCTAssertGreaterThan(first.totalReclaimedBytes, 0)
        let second = try await store.cleanupAll(descriptors: descriptors)
        XCTAssertEqual(second.totalReclaimedBytes, 0)
        XCTAssertTrue(second.perRepo.isEmpty)
    }

    // MARK: - Independent size oracle

    /// Allocated size of one regular file, recomputed independently of
    /// `ModelStore` so size assertions don't just echo the implementation.
    private func allocated(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [
            .totalFileAllocatedSizeKey, .fileSizeKey,
        ])
        return Int64(values?.totalFileAllocatedSize ?? values?.fileSize ?? 0)
    }
}
