// ModelStore.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import OSLog

/// Where a single model's bytes can live on disk, and whether a higher-priority
/// copy exists. Built by the route/module layer (which knows the id -> HF-repo
/// and bundle mappings) and handed to `ModelStore`, keeping all filesystem logic
/// in one pure-Foundation place.
///
/// `hfRepoDirName` is the HuggingFace hub directory name
/// (`models--<namespace>--<repo>`), `modelsDirSubdir` is the long-lived copy under
/// `EngineConfig.modelsDirectory` (or `nil`/`""` for the STT scheme that probes the
/// models directory root). `isBundled` is `true` when the app ships the weights in
/// its bundle, so the resolver prefers that copy and any hub/modelsDirectory copy
/// is reclaimable duplicate.
public struct ModelCacheDescriptor: Sendable, Equatable {
    public let id: String
    public let module: String
    public let hfRepoDirName: String?
    public let modelsDirSubdir: String?
    public let isBundled: Bool

    public init(
        id: String,
        module: String,
        hfRepoDirName: String?,
        modelsDirSubdir: String?,
        isBundled: Bool
    ) {
        self.id = id
        self.module = module
        self.hfRepoDirName = hfRepoDirName
        self.modelsDirSubdir = modelsDirSubdir
        self.isBundled = isBundled
    }

    /// Hub directory name for a HuggingFace id (`namespace/repo` ->
    /// `models--namespace--repo`), matching swift-huggingface's
    /// `HubCache.repoDirectory` naming. Returns `nil` for a malformed id.
    public static func hubRepoDirName(forHuggingFaceID id: String) -> String? {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/"), !trimmed.hasSuffix("/") else {
            return nil
        }
        return "models--" + trimmed.replacingOccurrences(of: "/", with: "--")
    }
}

/// On-disk hygiene for the engine's model caches: real footprint sizing, full
/// delete, superseded-snapshot collapse, and a one-shot cleanup migration.
///
/// Pure Foundation, no MLX / swift-huggingface dependency, so it runs in plain
/// `swift test` and is the single owner of the cache layout knowledge. It is an
/// `actor` purely to serialize destructive filesystem mutations against itself;
/// it holds no model state.
///
/// ## HuggingFace hub layout it operates on
/// ```
/// <hub>/models--<ns>--<repo>/
///   ├── blobs/<etag>           # the real bytes
///   ├── refs/<ref>             # file whose contents are the live commit hash
///   └── snapshots/<commit>/…   # symlinks → ../../blobs/<etag>
/// ```
/// Because snapshot entries are **symlinks**, deleting a snapshot directory frees
/// almost nothing — the bytes are in `blobs/`. Every operation here is therefore
/// blob-aware: size counts each blob once and never double-counts a symlink;
/// collapse deletes superseded snapshot dirs **and** garbage-collects the blobs
/// they orphaned.
public actor ModelStore {
    private let hubCacheDirectory: URL
    private let modelsDirectory: URL
    private let fileManager: FileManager
    private let logger = Logger(subsystem: "live.yooz.engine", category: "ModelStore")

    public init(
        hubCacheDirectory: URL = EngineConfig.huggingFaceCacheDirectory,
        modelsDirectory: URL = EngineConfig.modelsDirectory,
        fileManager: FileManager = .default
    ) {
        self.hubCacheDirectory = hubCacheDirectory
        self.modelsDirectory = modelsDirectory
        self.fileManager = fileManager
    }

    /// Total reclaimable bytes a cleanup pass freed, keyed by hub repo dir name
    /// (plus a synthetic key for modelsDirectory removals).
    public struct CleanupReport: Sendable, Equatable {
        public var totalReclaimedBytes: Int64
        public var perRepo: [String: Int64]

        public init(totalReclaimedBytes: Int64 = 0, perRepo: [String: Int64] = [:]) {
            self.totalReclaimedBytes = totalReclaimedBytes
            self.perRepo = perRepo
        }
    }

    // MARK: - Inventory

    /// Names of every model repo directory present in the hub cache
    /// (`models--<ns>--<repo>`). Drives the disk-first sweep that surfaces
    /// downloaded models (Parakeet, etc.) the catalog layer doesn't enumerate.
    public func cachedRepoDirNames() -> [String] {
        directoryContents(hubCacheDirectory)
            .map(\.lastPathComponent)
            .filter { $0.hasPrefix("models--") }
            .sorted()
    }

    /// Per-LLM-model facts the module layer supplies to `inventory(...)`: the
    /// cache descriptor plus the live display/loaded/active state the pure store
    /// can't read on its own.
    public struct LLMInventoryInput: Sendable, Equatable {
        public let descriptor: ModelCacheDescriptor
        public let displayName: String
        public let loaded: Bool
        public let isActive: Bool

        public init(
            descriptor: ModelCacheDescriptor,
            displayName: String,
            loaded: Bool,
            isActive: Bool
        ) {
            self.descriptor = descriptor
            self.displayName = displayName
            self.loaded = loaded
            self.isActive = isActive
        }
    }

    /// One assembled inventory row (transport-neutral; each transport maps it to
    /// its wire type).
    public struct ManagedModelRow: Sendable, Equatable {
        public let id: String
        public let module: String
        public let displayName: String
        public let sizeBytes: Int64
        public let cached: Bool
        public let loaded: Bool
        public let isActive: Bool
        public let deletable: Bool
    }

    /// Build the model-management inventory: a friendly row per known LLM model
    /// (real on-disk size + bundle awareness), then a disk-first sweep of every
    /// other hub repo (Parakeet, legacy) so nothing consuming disk is hidden.
    /// Only installed/available rows are returned (`cached || size > 0`); a model
    /// that is neither bundled nor downloaded is omitted.
    public func inventory(
        llm: [LLMInventoryInput],
        activeSTTRepoDirName: String?
    ) -> [ManagedModelRow] {
        var rows: [ManagedModelRow] = []
        var coveredRepos = Set<String>()

        for input in llm {
            let descriptor = input.descriptor
            if let repo = descriptor.hfRepoDirName { coveredRepos.insert(repo) }
            let size = onDiskSize(
                hfRepoDirName: descriptor.hfRepoDirName,
                modelsDirSubdir: descriptor.modelsDirSubdir
            )
            let cached = descriptor.isBundled || size > 0
            guard cached || size > 0 else { continue }
            rows.append(ManagedModelRow(
                id: descriptor.id,
                module: descriptor.module,
                displayName: input.displayName,
                sizeBytes: size,
                cached: cached,
                loaded: input.loaded,
                isActive: input.isActive,
                deletable: size > 0 && !input.isActive
            ))
        }

        for repoDir in cachedRepoDirNames() where !coveredRepos.contains(repoDir) {
            let size = onDiskSize(hfRepoDirName: repoDir, modelsDirSubdir: nil)
            let isActive = (repoDir == activeSTTRepoDirName)
            rows.append(ManagedModelRow(
                id: repoDir,
                module: "stt",
                displayName: Self.humanizeRepoDirName(repoDir),
                sizeBytes: size,
                cached: true,
                loaded: isActive,
                isActive: isActive,
                deletable: size > 0 && !isActive
            ))
        }

        return rows
    }

    /// `models--mlx-community--parakeet-tdt-0.6b-v3` -> `parakeet-tdt-0.6b-v3`.
    /// A readable fallback label for a swept repo with no catalog entry.
    static func humanizeRepoDirName(_ dirName: String) -> String {
        guard dirName.hasPrefix("models--") else { return dirName }
        let body = String(dirName.dropFirst("models--".count))
        let parts = body.components(separatedBy: "--")
        return parts.last ?? body
    }

    // MARK: - Size

    /// Reclaimable on-disk footprint for one model: its hub repo (blobs +
    /// non-symlink snapshot copies + refs) plus any `modelsDirectory` copy. The
    /// app-bundle copy is intentionally excluded — it lives in the read-only app
    /// bundle and frees nothing.
    public func onDiskSize(hfRepoDirName: String?, modelsDirSubdir: String?) -> Int64 {
        var total: Int64 = 0
        if let repoDir = repoDirectoryURL(hfRepoDirName) {
            total += directoryAllocatedSize(repoDir)
        }
        if let modelsCopy = modelsDirectoryCopyURL(modelsDirSubdir) {
            total += directoryAllocatedSize(modelsCopy)
        }
        return total
    }

    // MARK: - Delete

    /// Remove every reclaimable copy of a model — the whole hub repo tree (blobs,
    /// snapshots, refs, plus the sibling `.metadata` entry) and the
    /// `modelsDirectory` copy. Returns bytes reclaimed. The caller must unload the
    /// model from memory first. The bundled copy (if any) is untouched, so a
    /// bundled model stays usable after delete.
    @discardableResult
    public func deleteModel(hfRepoDirName: String?, modelsDirSubdir: String?) throws -> Int64 {
        var reclaimed: Int64 = 0

        if let repoDir = repoDirectoryURL(hfRepoDirName) {
            reclaimed += try removeHubRepo(repoDir)
        }

        if let modelsCopy = modelsDirectoryCopyURL(modelsDirSubdir) {
            reclaimed += directoryAllocatedSize(modelsCopy)
            try fileManager.removeItem(at: modelsCopy)
        }

        return reclaimed
    }

    /// Remove a hub repo tree plus its sibling `.metadata/<dirName>` entry (which
    /// HubCache parks outside the repo dir), returning bytes reclaimed. The
    /// metadata removal is best-effort — it's tiny housekeeping, not the model.
    private func removeHubRepo(_ repoDir: URL) throws -> Int64 {
        var reclaimed = directoryAllocatedSize(repoDir)
        try fileManager.removeItem(at: repoDir)
        let metadataDir = hubCacheDirectory
            .appendingPathComponent(".metadata")
            .appendingPathComponent(repoDir.lastPathComponent)
        if fileManager.fileExists(atPath: metadataDir.path) {
            reclaimed += directoryAllocatedSize(metadataDir)
            try? fileManager.removeItem(at: metadataDir)
        }
        return reclaimed
    }

    // MARK: - Collapse superseded snapshots

    /// Delete every snapshot of `hfRepoDirName` that is **not** referenced by a
    /// `refs/*` file (the loader resolves `refs/main` -> a commit -> that
    /// snapshot), then garbage-collect blobs no surviving snapshot points at.
    /// Returns bytes reclaimed.
    ///
    /// Fail-safe by construction — it returns 0 (deletes nothing) rather than risk
    /// removing a live file when it can't prove what is live:
    /// - No ref to anchor a survivor -> no-op.
    /// - The ref points at a commit whose snapshot is absent or not yet fully
    ///   materialized (an interrupted update advances `refs/main` before the new
    ///   snapshot finishes downloading) -> no-op, so the older working snapshot is
    ///   never collapsed out from under a half-finished update.
    /// - A snapshot a ref points at is kept even while still downloading.
    /// All survivor reads happen before any delete, and an unreadable ref or
    /// unresolvable snapshot symlink **throws** (never silently exposes a blob to
    /// GC). Blob GC is skipped entirely when a surviving snapshot uses
    /// copy-fallback files instead of symlinks (their bytes aren't a mappable blob).
    @discardableResult
    public func collapseSnapshots(hfRepoDirName: String) throws -> Int64 {
        guard let repoDir = repoDirectoryURL(hfRepoDirName) else { return 0 }
        let snapshotsDir = repoDir.appendingPathComponent("snapshots")
        let refsDir = repoDir.appendingPathComponent("refs")
        let blobsDir = repoDir.appendingPathComponent("blobs")

        let referencedCommits = try referencedCommits(refsDir: refsDir)
        guard !referencedCommits.isEmpty else { return 0 }

        let present = directoryContents(snapshotsDir)
        // At least one referenced snapshot must be fully materialized on disk
        // (config.json + weights). Otherwise the ref is ahead of an unfinished
        // download and collapsing would delete the only working older snapshot.
        let hasMaterializedSurvivor = referencedCommits.contains { commit in
            isCompleteModelDir(snapshotsDir.appendingPathComponent(commit))
        }
        guard hasMaterializedSurvivor else { return 0 }

        // Superseded = present but unreferenced. Anything a ref points at — even a
        // partial, still-downloading snapshot — is kept.
        let toDelete = present.filter { !referencedCommits.contains($0.lastPathComponent) }
        let surviving = present.filter { referencedCommits.contains($0.lastPathComponent) }

        // Compute referenced blobs from the SURVIVING snapshots BEFORE deleting
        // anything, so an unreadable snapshot / unresolvable symlink aborts the
        // whole collapse with nothing removed.
        var sawCopyFallback = false
        let referencedBlobs = try referencedBlobNames(
            in: surviving, sawCopyFallback: &sawCopyFallback
        )

        var reclaimed: Int64 = 0
        for snapshotDir in toDelete {
            reclaimed += directoryAllocatedSize(snapshotDir)
            try fileManager.removeItem(at: snapshotDir)
        }

        // GC orphan blobs only when every surviving snapshot is symlink-based; a
        // copy-fallback snapshot's bytes live in the snapshot dir, not a mappable
        // `blobs/<etag>`, so GC'ing against blob names could free a live blob.
        if !sawCopyFallback {
            for blob in directoryContents(blobsDir) {
                let name = blob.lastPathComponent
                // Leave partial downloads (`<etag>.incomplete`) alone — a fetch may
                // be mid-flight, and they are never referenced by a snapshot anyway.
                if name.hasSuffix(".incomplete") { continue }
                if referencedBlobs.contains(name) { continue }
                reclaimed += fileAllocatedSize(blob)
                try fileManager.removeItem(at: blob)
            }
        }

        return reclaimed
    }

    // MARK: - Cleanup migration

    /// One-shot disk hygiene for an existing install: collapse superseded
    /// snapshots in **every** hub repo, then remove hub/`modelsDirectory` copies
    /// made redundant by a higher-priority copy (bundled weights, or a complete
    /// `modelsDirectory` copy the resolver prefers over the hub). Idempotent: a
    /// second run finds nothing left to reclaim and returns 0.
    ///
    /// Safety: a hub copy is removed only when a *complete* higher-priority copy
    /// exists — bundled (always complete) or a `modelsDirectory` copy carrying
    /// BOTH `config.json` and weights (`isCompleteModelDir`, matching the loader's
    /// `isModelCached` predicate) — so the resolver's chosen copy is never deleted.
    public func cleanupAll(descriptors: [ModelCacheDescriptor]) throws -> CleanupReport {
        var report = CleanupReport()

        // 1. Collapse stacked snapshots across all model repos in the hub cache. A
        // failure in one repo must not abort the whole pass; log and move on.
        for repoDir in directoryContents(hubCacheDirectory)
        where repoDir.lastPathComponent.hasPrefix("models--") {
            let dirName = repoDir.lastPathComponent
            do {
                let freed = try collapseSnapshots(hfRepoDirName: dirName)
                if freed > 0 { report.add(freed, repo: dirName) }
            } catch {
                logger.warning(
                    "cleanup: snapshot collapse failed for \(dirName, privacy: .public): \(error.localizedDescription, privacy: .public) — skipped"
                )
            }
        }

        // 2. Remove redundant duplicates the resolver would never load from.
        for descriptor in descriptors {
            guard let repoDir = repoDirectoryURL(descriptor.hfRepoDirName) else { continue }
            let modelsCopy = modelsDirectoryCopyURL(descriptor.modelsDirSubdir)
            let higherPriorityCopyExists =
                descriptor.isBundled || (modelsCopy.map(isCompleteModelDir) ?? false)
            guard higherPriorityCopyExists else { continue }

            let freed = try removeHubRepo(repoDir)
            report.add(freed, repo: repoDir.lastPathComponent)
        }

        return report
    }

    // MARK: - Path helpers

    private func repoDirectoryURL(_ hfRepoDirName: String?) -> URL? {
        guard let name = hfRepoDirName,
              let url = confinedURL(hubCacheDirectory, name),
              fileManager.fileExists(atPath: url.path) else { return nil }
        return url
    }

    private func modelsDirectoryCopyURL(_ subdir: String?) -> URL? {
        guard let subdir,
              let url = confinedURL(modelsDirectory, subdir),
              fileManager.fileExists(atPath: url.path) else { return nil }
        return url
    }

    /// Append `component` to `base`, returning `nil` if the resolved path would
    /// escape `base` (path-traversal guard). The public `deleteModel` /
    /// `collapseSnapshots` surface takes arbitrary id strings; this confines every
    /// filesystem op to the hub / models directories even if a caller bypasses the
    /// `hubRepoDirName(forHuggingFaceID:)` factory.
    private func confinedURL(_ base: URL, _ component: String) -> URL? {
        guard !component.isEmpty else { return nil }
        let url = base.appendingPathComponent(component)
        let basePath = base.standardizedFileURL.path
        let prefix = basePath.hasSuffix("/") ? basePath : basePath + "/"
        guard url.standardizedFileURL.path.hasPrefix(prefix) else { return nil }
        return url
    }

    /// Whether a model directory is COMPLETE — both `config.json` and at least one
    /// `*.safetensors` (resolving symlinks). Matches the loader's `isModelCached` /
    /// `firstModelDirectory` predicate, so a half-downloaded copy (config present,
    /// weights missing) is never mistaken for a usable higher-priority copy.
    private func isCompleteModelDir(_ dir: URL) -> Bool {
        guard fileManager.fileExists(
            atPath: dir.appendingPathComponent("config.json").path
        ) else { return false }
        let contents = (try? fileManager.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil
        )) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    /// Commit hashes referenced by any file under `refs/` (refs can nest, e.g.
    /// `refs/pr/5`), each file's trimmed contents being a commit hash. Returns an
    /// empty set when `refs/` is absent. **Throws** when `refs/` exists but can't
    /// be enumerated or a ref file can't be read — a destructive caller must not
    /// proceed on an incomplete survivor set.
    private func referencedCommits(refsDir: URL) throws -> Set<String> {
        guard fileManager.fileExists(atPath: refsDir.path) else { return [] }
        guard let enumerator = fileManager.enumerator(
            at: refsDir,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else { throw ModelStoreError.unreadableRefs(refsDir.path) }

        var commits = Set<String>()
        for case let url as URL in enumerator {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                .isRegularFile == true else { continue }
            guard let raw = try? String(contentsOf: url, encoding: .utf8) else {
                throw ModelStoreError.unreadableRefs(url.path)
            }
            let commit = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !commit.isEmpty { commits.insert(commit) }
        }
        return commits
    }

    /// Blob filenames kept alive by symlinks under the given surviving snapshot
    /// dirs. **Throws** if a snapshot dir can't be enumerated or a symlink can't be
    /// resolved — failing closed so a blob is never wrongly GC'd. Sets
    /// `sawCopyFallback` when a snapshot entry is a real file (HuggingFace
    /// copy-fallback) instead of a symlink; the caller then skips blob GC.
    private func referencedBlobNames(
        in snapshotDirs: [URL], sawCopyFallback: inout Bool
    ) throws -> Set<String> {
        var names = Set<String>()
        for dir in snapshotDirs {
            guard let enumerator = fileManager.enumerator(
                at: dir,
                includingPropertiesForKeys: [.isSymbolicLinkKey, .isRegularFileKey]
            ) else { throw ModelStoreError.unreadableSnapshots(dir.path) }
            for case let url as URL in enumerator {
                let values = try? url.resourceValues(
                    forKeys: [.isSymbolicLinkKey, .isRegularFileKey]
                )
                if values?.isSymbolicLink == true {
                    guard let dest = try? fileManager.destinationOfSymbolicLink(
                        atPath: url.path
                    ) else { throw ModelStoreError.unresolvableSymlink(url.path) }
                    names.insert((dest as NSString).lastPathComponent)
                } else if values?.isRegularFile == true {
                    sawCopyFallback = true
                }
            }
        }
        return names
    }

    // MARK: - Size helpers

    /// Direct (non-recursive) child URLs of a directory; empty if it is missing.
    private func directoryContents(_ dir: URL) -> [URL] {
        (try? fileManager.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
    }

    /// Recursive on-disk size of a directory, counting each real file once and
    /// **skipping symlinks** so blobs referenced via snapshot symlinks aren't
    /// double-counted (the blob itself is counted under `blobs/`). Uses
    /// `.totalFileAllocatedSize` (true on-disk, block-rounded) with `.fileSize`
    /// fallback.
    private func directoryAllocatedSize(_ dir: URL) -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: dir,
            includingPropertiesForKeys: [
                .isRegularFileKey, .isSymbolicLinkKey,
                .totalFileAllocatedSizeKey, .fileSizeKey,
            ],
            options: []
        ) else { return 0 }

        var total: Int64 = 0
        for case let url as URL in enumerator {
            total += fileAllocatedSize(url)
        }
        return total
    }

    /// Allocated size of a single file URL, or 0 for symlinks / directories /
    /// unreadable entries.
    private func fileAllocatedSize(_ url: URL) -> Int64 {
        guard let values = try? url.resourceValues(forKeys: [
            .isRegularFileKey, .isSymbolicLinkKey,
            .totalFileAllocatedSizeKey, .fileSizeKey,
        ]) else { return 0 }
        if values.isSymbolicLink == true { return 0 }
        guard values.isRegularFile == true else { return 0 }
        return Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
    }
}

private extension ModelStore.CleanupReport {
    mutating func add(_ bytes: Int64, repo: String) {
        totalReclaimedBytes += bytes
        perRepo[repo, default: 0] += bytes
    }
}

/// Errors `ModelStore` throws to fail closed on a destructive path rather than
/// risk deleting a live file when the cache can't be read reliably.
public enum ModelStoreError: Error, Equatable {
    /// `refs/` exists but couldn't be enumerated, or a ref file couldn't be read.
    case unreadableRefs(String)
    /// A surviving snapshot directory couldn't be enumerated.
    case unreadableSnapshots(String)
    /// A snapshot symlink couldn't be resolved to its blob target.
    case unresolvableSymlink(String)
}
