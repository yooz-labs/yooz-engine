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

    @discardableResult
    private func writeBlob(repo: URL, name: String, byteCount: Int) throws -> URL {
        let blobs = repo.appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        let url = blobs.appendingPathComponent(name)
        try Data(count: byteCount).write(to: url)
        return url
    }

    /// `<repo>/snapshots/<commit>/<filename>` as a symlink to `../../blobs/<blob>`.
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

    /// A COMPLETE snapshot — `config.json` + `model.safetensors` symlinks — as the
    /// HF downloader leaves a finished snapshot. `ModelStore` only treats such a
    /// snapshot as a materialized survivor.
    private func materialize(
        repo: URL, commit: String, configBlob: String, weightsBlob: String
    ) throws {
        try linkSnapshot(repo: repo, commit: commit, filename: "config.json", toBlob: configBlob)
        try linkSnapshot(repo: repo, commit: commit, filename: "model.safetensors", toBlob: weightsBlob)
    }

    private func writeRef(repo: URL, ref: String, commit: String) throws {
        let refs = repo.appendingPathComponent("refs")
        try fm.createDirectory(at: refs, withIntermediateDirectories: true)
        try commit.write(
            to: refs.appendingPathComponent(ref), atomically: true, encoding: .utf8
        )
    }

    private func exists(_ url: URL) -> Bool { fm.fileExists(atPath: url.path) }
    private func snapshot(_ repo: URL, _ commit: String) -> URL {
        repo.appendingPathComponent("snapshots/\(commit)")
    }
    private func blob(_ repo: URL, _ name: String) -> URL {
        repo.appendingPathComponent("blobs/\(name)")
    }

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
        let expected = allocated(blob(repo, "blobA"))
            + allocated(blob(repo, "blobB"))
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
        let expected = allocated(blob(repo, "blobA"))
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
        try materialize(repo: repo, commit: "old111", configBlob: "shared", weightsBlob: "old")
        try materialize(repo: repo, commit: "new222", configBlob: "shared", weightsBlob: "new")
        try writeRef(repo: repo, ref: "main", commit: "new222")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")

        XCTAssertGreaterThan(reclaimed, 0)
        XCTAssertFalse(exists(snapshot(repo, "old111")))
        XCTAssertTrue(exists(snapshot(repo, "new222")))
        XCTAssertFalse(exists(blob(repo, "old")))
        XCTAssertTrue(exists(blob(repo, "shared")))
        XCTAssertTrue(exists(blob(repo, "new")))
        let liveConfig = snapshot(repo, "new222").appendingPathComponent("config.json")
        XCTAssertEqual(try Data(contentsOf: liveConfig).count, 8_192)
    }

    func testCollapseIsIdempotent() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "shared", byteCount: 8_192)
        try writeBlob(repo: repo, name: "old", byteCount: 16_384)
        try writeBlob(repo: repo, name: "new", byteCount: 16_384)
        try materialize(repo: repo, commit: "old111", configBlob: "shared", weightsBlob: "old")
        try materialize(repo: repo, commit: "new222", configBlob: "shared", weightsBlob: "new")
        try writeRef(repo: repo, ref: "main", commit: "new222")

        let store = makeStore()
        let first = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertGreaterThan(first, 0)
        let second = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(second, 0)
    }

    func testCollapseWithoutRefIsNoOp() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "a", byteCount: 8_192)
        try writeBlob(repo: repo, name: "b", byteCount: 8_192)
        try materialize(repo: repo, commit: "c1", configBlob: "a", weightsBlob: "a")
        try materialize(repo: repo, commit: "c2", configBlob: "b", weightsBlob: "b")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(reclaimed, 0)
        XCTAssertTrue(exists(snapshot(repo, "c1")))
        XCTAssertTrue(exists(snapshot(repo, "c2")))
    }

    func testCollapsePreservesIncompleteBlobsAndGCsDeadBlobs() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "cfg", byteCount: 4_096)
        try writeBlob(repo: repo, name: "live", byteCount: 8_192)
        try writeBlob(repo: repo, name: "partial.incomplete", byteCount: 4_096)
        try writeBlob(repo: repo, name: "dead", byteCount: 8_192)  // orphan
        try materialize(repo: repo, commit: "c1", configBlob: "cfg", weightsBlob: "live")
        try writeRef(repo: repo, ref: "main", commit: "c1")

        let store = makeStore()
        _ = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        // In-flight partial never reaped; live blobs kept; dead orphan GC'd.
        XCTAssertTrue(exists(blob(repo, "partial.incomplete")))
        XCTAssertTrue(exists(blob(repo, "cfg")))
        XCTAssertTrue(exists(blob(repo, "live")))
        XCTAssertFalse(exists(blob(repo, "dead")))
    }

    /// Data-loss guard: an interrupted update advances `refs/main` to a commit
    /// whose snapshot hasn't been created yet. Collapsing must NOT delete the only
    /// working older snapshot.
    func testCollapseKeepsOldSnapshotWhenRefPointsToAbsentCommit() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "cfg", byteCount: 4_096)
        try writeBlob(repo: repo, name: "weights", byteCount: 8_192)
        try materialize(repo: repo, commit: "v1", configBlob: "cfg", weightsBlob: "weights")
        try writeRef(repo: repo, ref: "main", commit: "v2-not-downloaded-yet")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(reclaimed, 0)
        XCTAssertTrue(exists(snapshot(repo, "v1")))
        XCTAssertTrue(exists(blob(repo, "cfg")))
        XCTAssertTrue(exists(blob(repo, "weights")))
    }

    /// Data-loss guard: `refs/main` points at a snapshot dir that exists but is
    /// only half-downloaded (no `config.json`). Collapse must no-op rather than
    /// clobber the old working snapshot.
    func testCollapseKeepsEverythingWhenReferencedSnapshotIsPartial() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "cfg", byteCount: 4_096)
        try writeBlob(repo: repo, name: "weights", byteCount: 8_192)
        try writeBlob(repo: repo, name: "partialW", byteCount: 8_192)
        try materialize(repo: repo, commit: "v1", configBlob: "cfg", weightsBlob: "weights")
        // v2 present but incomplete: weights only, no config.json.
        try linkSnapshot(repo: repo, commit: "v2", filename: "model.safetensors", toBlob: "partialW")
        try writeRef(repo: repo, ref: "main", commit: "v2")

        let store = makeStore()
        let reclaimed = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        XCTAssertEqual(reclaimed, 0)
        XCTAssertTrue(exists(snapshot(repo, "v1")))
        XCTAssertTrue(exists(snapshot(repo, "v2")))
        XCTAssertTrue(exists(blob(repo, "weights")))
        XCTAssertTrue(exists(blob(repo, "partialW")))
    }

    /// When a surviving snapshot uses HF copy-fallback (real files, not symlinks),
    /// blob GC is skipped — a copy's bytes aren't a mappable `blobs/<etag>`, so
    /// GC'ing against blob names could free a live blob.
    func testCollapseSkipsBlobGCForCopyFallbackSnapshot() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        let blobs = repo.appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try Data(count: 8_192).write(to: blobs.appendingPathComponent("dead"))
        // c1 as real-file copies (no symlinks).
        let c1 = snapshot(repo, "c1")
        try fm.createDirectory(at: c1, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: c1.appendingPathComponent("config.json"))
        try Data(count: 8_192).write(to: c1.appendingPathComponent("model.safetensors"))
        try writeRef(repo: repo, ref: "main", commit: "c1")

        let store = makeStore()
        _ = try await store.collapseSnapshots(hfRepoDirName: "models--Acme--Demo")
        // GC skipped: the orphan blob survives rather than risk a live blob.
        XCTAssertTrue(exists(blob(repo, "dead")))
        XCTAssertTrue(exists(c1))
    }

    // MARK: - Delete

    func testDeleteModelRemovesHubAndModelsDirCopies() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Demo")
        try writeBlob(repo: repo, name: "blobA", byteCount: 32_768)
        try materialize(repo: repo, commit: "c1", configBlob: "blobA", weightsBlob: "blobA")
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

    /// Path-traversal guard: a crafted id that resolves outside the hub directory
    /// is a no-op, never touching the sibling directory.
    func testDeleteModelRejectsPathTraversalId() async throws {
        let secret = root.appendingPathComponent("secret")
        try fm.createDirectory(at: secret, withIntermediateDirectories: true)
        try Data(count: 1_024).write(to: secret.appendingPathComponent("keep.txt"))

        let store = makeStore()
        let reclaimed = try await store.deleteModel(
            hfRepoDirName: "../secret", modelsDirSubdir: nil
        )
        XCTAssertEqual(reclaimed, 0)
        XCTAssertTrue(exists(secret), "traversal id must not escape the hub directory")
    }

    // MARK: - Cleanup migration

    func testCleanupCollapsesAllReposAndDedupesDuplicates() async throws {
        // Repo 1: stacked snapshots -> collapsed.
        let stacked = hub.appendingPathComponent("models--Acme--Stacked")
        try writeBlob(repo: stacked, name: "shared", byteCount: 8_192)
        try writeBlob(repo: stacked, name: "old", byteCount: 16_384)
        try writeBlob(repo: stacked, name: "new", byteCount: 16_384)
        try materialize(repo: stacked, commit: "o", configBlob: "shared", weightsBlob: "old")
        try materialize(repo: stacked, commit: "n", configBlob: "shared", weightsBlob: "new")
        try writeRef(repo: stacked, ref: "main", commit: "n")

        // Repo 2: bundled duplicate -> whole hub repo removed.
        let bundled = hub.appendingPathComponent("models--Acme--Bundled")
        try writeBlob(repo: bundled, name: "b", byteCount: 32_768)
        try materialize(repo: bundled, commit: "c1", configBlob: "b", weightsBlob: "b")
        try writeRef(repo: bundled, ref: "main", commit: "c1")

        // Repo 3: COMPLETE modelsDirectory copy -> hub repo removed.
        let mirrored = hub.appendingPathComponent("models--Acme--Mirrored")
        try writeBlob(repo: mirrored, name: "b", byteCount: 32_768)
        try materialize(repo: mirrored, commit: "c1", configBlob: "b", weightsBlob: "b")
        try writeRef(repo: mirrored, ref: "main", commit: "c1")
        let mirroredCopy = modelsDir.appendingPathComponent("mirrored")
        try fm.createDirectory(at: mirroredCopy, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: mirroredCopy.appendingPathComponent("config.json"))
        try Data(count: 4_096).write(to: mirroredCopy.appendingPathComponent("model.safetensors"))

        // Repo 4: no higher-priority copy -> kept.
        let kept = hub.appendingPathComponent("models--Acme--Kept")
        try writeBlob(repo: kept, name: "b", byteCount: 32_768)
        try materialize(repo: kept, commit: "c1", configBlob: "b", weightsBlob: "b")
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
        XCTAssertFalse(exists(blob(stacked, "old")))
        XCTAssertTrue(exists(blob(stacked, "new")))
        XCTAssertFalse(exists(bundled))
        XCTAssertFalse(exists(mirrored))
        XCTAssertTrue(exists(kept))
    }

    /// Data-loss guard: a half-downloaded modelsDirectory copy (config but no
    /// weights) is NOT a usable higher-priority copy, so the complete hub copy is
    /// kept.
    func testCleanupKeepsHubWhenModelsDirCopyIncomplete() async throws {
        let repo = hub.appendingPathComponent("models--Acme--Q")
        try writeBlob(repo: repo, name: "b", byteCount: 32_768)
        try materialize(repo: repo, commit: "c1", configBlob: "b", weightsBlob: "b")
        try writeRef(repo: repo, ref: "main", commit: "c1")
        // Incomplete copy: config.json present, weights missing.
        let copy = modelsDir.appendingPathComponent("q")
        try fm.createDirectory(at: copy, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: copy.appendingPathComponent("config.json"))

        let descriptors = [
            ModelCacheDescriptor(
                id: "q", module: "llm",
                hfRepoDirName: "models--Acme--Q",
                modelsDirSubdir: "q", isBundled: false
            )
        ]

        let store = makeStore()
        _ = try await store.cleanupAll(descriptors: descriptors)
        XCTAssertTrue(exists(repo), "incomplete models-dir copy must not trigger hub deletion")
    }

    func testCleanupIsIdempotent() async throws {
        let stacked = hub.appendingPathComponent("models--Acme--Stacked")
        try writeBlob(repo: stacked, name: "shared", byteCount: 8_192)
        try writeBlob(repo: stacked, name: "old", byteCount: 16_384)
        try writeBlob(repo: stacked, name: "new", byteCount: 16_384)
        try materialize(repo: stacked, commit: "o", configBlob: "shared", weightsBlob: "old")
        try materialize(repo: stacked, commit: "n", configBlob: "shared", weightsBlob: "new")
        try writeRef(repo: stacked, ref: "main", commit: "n")

        let bundled = hub.appendingPathComponent("models--Acme--Bundled")
        try writeBlob(repo: bundled, name: "b", byteCount: 32_768)
        try materialize(repo: bundled, commit: "c1", configBlob: "b", weightsBlob: "b")
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

    private func allocated(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [
            .totalFileAllocatedSizeKey, .fileSizeKey,
        ])
        return Int64(values?.totalFileAllocatedSize ?? values?.fileSize ?? 0)
    }
}
