// InfiniteCacheProbe.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import HuggingFace
import os.log

private let infiniteCacheLogger = Logger(
    subsystem: "live.yooz.engine",
    category: "InfiniteCacheProbe"
)

enum InfiniteCacheProbe {
    static func isCached(_ descriptor: InfiniteBackendDescriptor) -> Bool {
        guard let repository = descriptor.repository else {
            return false
        }
        guard let repo = repoID(repository.id) else {
            infiniteCacheLogger.error("Malformed Infinite HF id \(repository.id, privacy: .public)")
            return false
        }

        let snapshotsRoot = HubCache().snapshotsDirectory(repo: repo, kind: .model)
        let snapshots: [URL]
        do {
            snapshots = try FileManager.default.contentsOfDirectory(
                at: snapshotsRoot,
                includingPropertiesForKeys: nil
            )
        } catch let error as NSError where error.code == NSFileReadNoSuchFileError {
            return false
        } catch {
            infiniteCacheLogger.warning(
                "Infinite cache probe failed for \(repository.id, privacy: .public): \(error.localizedDescription, privacy: .public)"
            )
            return false
        }

        return snapshots.contains { snapshot in
            snapshot.lastPathComponent == repository.revision &&
            hasRequiredFiles(in: snapshot, required: descriptor.requiredCachedFiles)
        }
    }

    private static func repoID(_ raw: String) -> Repo.ID? {
        let parts = raw.split(
            separator: "/",
            maxSplits: 1,
            omittingEmptySubsequences: true
        )
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            return nil
        }
        return Repo.ID(namespace: String(parts[0]), name: String(parts[1]))
    }

    private static func hasRequiredFiles(in snapshot: URL, required: [String]) -> Bool {
        guard FileManager.default.fileExists(
            atPath: snapshot.appendingPathComponent("config.json").path
        ) else {
            return false
        }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: snapshot,
            includingPropertiesForKeys: nil
        )) ?? []
        for requirement in required where requirement.hasPrefix(".") {
            guard entries.contains(where: { $0.pathExtension == String(requirement.dropFirst()) }) else {
                return false
            }
        }
        return true
    }
}
