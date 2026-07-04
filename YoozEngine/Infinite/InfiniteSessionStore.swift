// InfiniteSessionStore.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import CryptoKit
import Darwin
import Foundation
import os.log

private let sessionStoreLogger = Logger(
    subsystem: "live.yooz.engine",
    category: "InfiniteSessionStore"
)

/// Errors raised by filesystem operations on `InfiniteSessionStore` that
/// aren't already covered by `Foundation`'s own thrown errors (missing
/// files surface as the standard `CocoaError`/`POSIXError` from
/// `FileManager`/`Data`).
public enum InfiniteSessionStoreError: Error, Equatable {
    case checkpointNotFound(session: String, checkpoint: String)
    case checkpointAlreadyExists(session: String, checkpoint: String)
}

/// Raised by the `tokens.bin` codec.
public enum TokensCodecError: Error, Equatable {
    /// A token id didn't fit in the on-disk little-endian `UInt32` slot.
    case tokenIdOutOfRange(Int)
    /// `tokens.bin`'s byte count isn't a multiple of 4.
    case truncatedTokensFile(byteCount: Int)
}

/// Integrity mismatches between a stored `InfiniteSessionManifest` and the
/// live values a resume/fork is about to attach to. Each case carries the
/// expected (stored) and actual (live) value so the caller (a later PR maps
/// these to HTTP error bodies) can render a precise message.
public enum SessionIntegrityError: Error, Equatable, CustomStringConvertible {
    case schemaVersionMismatch(expected: Int, actual: Int)
    case repoIdMismatch(expected: String, actual: String)
    case revisionMismatch(expected: String, actual: String)
    case tokenizerHashMismatch(expected: String, actual: String)
    case tokenIdsSHA256Mismatch(expected: String, actual: String)

    public var description: String {
        switch self {
        case .schemaVersionMismatch(let expected, let actual):
            return "schemaVersion mismatch: expected \(expected), got \(actual)"
        case .repoIdMismatch(let expected, let actual):
            return "repoId mismatch: expected \(expected), got \(actual)"
        case .revisionMismatch(let expected, let actual):
            return "revision mismatch: expected \(expected), got \(actual)"
        case .tokenizerHashMismatch(let expected, let actual):
            return "tokenizerHash mismatch: expected \(expected), got \(actual)"
        case .tokenIdsSHA256Mismatch(let expected, let actual):
            return "tokens.bin sha256 mismatch: expected \(expected), got \(actual)"
        }
    }
}

/// Live-side facts a caller (the MLX backend, wired in a later PR) supplies
/// to `verify(manifest:against:session:checkpoint:)`. Plain data rather than
/// MLX/tokenizer types, so `InfiniteSessionStore` has no MLX import.
public struct LiveSessionFacts: Sendable, Equatable {
    public let schemaVersion: Int
    public let repoId: String
    public let revision: String
    public let tokenizerHash: String

    public init(schemaVersion: Int, repoId: String, revision: String, tokenizerHash: String) {
        self.schemaVersion = schemaVersion
        self.repoId = repoId
        self.revision = revision
        self.tokenizerHash = tokenizerHash
    }
}

/// Filesystem-backed store for durable Infinite session checkpoints.
///
/// Layout: `<root>/<sessionID>/<checkpointID>/{cache.safetensors,
/// manifest.json, tokens.bin}`. `cache.safetensors` is written/read by the
/// MLX backend (a later PR); this type only hands out the checkpoint
/// directory URL for that file. No MLX imports — this type must stay
/// usable from a context that hasn't loaded MLX.
public struct InfiniteSessionStore: Sendable {
    public let root: URL

    public init(root: URL = InfiniteSessionStore.defaultRoot) {
        self.root = root
    }

    /// `~/Library/Application Support/YoozEngine/Infinite/Sessions`, or the
    /// `YOOZ_INFINITE_SESSIONS_DIR` override when set. Mirrors the
    /// `YOOZ_QWEN3_ASR_DIR` override on
    /// `Qwen3ASRModelFetcher.defaultModelDir` (same env-override shape,
    /// same "compute locally, Foundation only" rationale: this file must
    /// stay import-light rather than depending on `EngineConfig`, and the
    /// resolved path still matches the engine's `Application
    /// Support/YoozEngine/...` convention (`EngineConfig.modelsDirectory`
    /// uses `.../YoozEngine/Models`; sessions live in a sibling
    /// `.../YoozEngine/Infinite/Sessions` directory).
    public static var defaultRoot: URL {
        if let override = ProcessInfo.processInfo.environment["YOOZ_INFINITE_SESSIONS_DIR"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                .appendingPathComponent("YoozEngine/Infinite/Sessions", isDirectory: true)
        }
        return appSupport.appendingPathComponent("YoozEngine/Infinite/Sessions", isDirectory: true)
    }

    // MARK: - Layout

    public func sessionDirectory(_ session: String) -> URL {
        root.appendingPathComponent(session, isDirectory: true)
    }

    /// Creates (if needed) and returns `<root>/<session>/<checkpoint>/`.
    @discardableResult
    public func checkpointDirectory(session: String, checkpoint: String) throws -> URL {
        let dir = sessionDirectory(session).appendingPathComponent(checkpoint, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func manifestURL(session: String, checkpoint: String) -> URL {
        sessionDirectory(session)
            .appendingPathComponent(checkpoint, isDirectory: true)
            .appendingPathComponent("manifest.json", isDirectory: false)
    }

    private func tokensURL(session: String, checkpoint: String) -> URL {
        sessionDirectory(session)
            .appendingPathComponent(checkpoint, isDirectory: true)
            .appendingPathComponent("tokens.bin", isDirectory: false)
    }

    // MARK: - Manifest

    public func writeManifest(
        _ manifest: InfiniteSessionManifest,
        session: String,
        checkpoint: String
    ) throws {
        try checkpointDirectory(session: session, checkpoint: checkpoint)
        let encoder = JSONEncoder()
        // `.iso8601` truncates to whole seconds, which is lossy for a
        // manifest round trip (`createdAt` wouldn't compare equal to what
        // was written). `.secondsSince1970` round-trips a `Date` exactly.
        encoder.dateEncodingStrategy = .secondsSince1970
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL(session: session, checkpoint: checkpoint), options: .atomic)
    }

    public func readManifest(session: String, checkpoint: String) throws -> InfiniteSessionManifest {
        let data = try Data(contentsOf: manifestURL(session: session, checkpoint: checkpoint))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        return try decoder.decode(InfiniteSessionManifest.self, from: data)
    }

    // MARK: - Tokens

    public func writeTokens(_ tokenIds: [Int], session: String, checkpoint: String) throws {
        try checkpointDirectory(session: session, checkpoint: checkpoint)
        let data = try Self.encodeTokens(tokenIds)
        try data.write(to: tokensURL(session: session, checkpoint: checkpoint), options: .atomic)
    }

    public func readTokens(session: String, checkpoint: String) throws -> [Int] {
        let data = try Data(contentsOf: tokensURL(session: session, checkpoint: checkpoint))
        return try Self.decodeTokens(data)
    }

    /// Encodes token ids as little-endian `UInt32`s with no header.
    public static func encodeTokens(_ tokenIds: [Int]) throws -> Data {
        var data = Data(capacity: tokenIds.count * MemoryLayout<UInt32>.size)
        for id in tokenIds {
            guard id >= 0, id <= Int(UInt32.max) else {
                throw TokensCodecError.tokenIdOutOfRange(id)
            }
            var little = UInt32(id).littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Decodes the little-endian `UInt32` token id encoding produced by
    /// `encodeTokens(_:)`.
    public static func decodeTokens(_ data: Data) throws -> [Int] {
        let width = MemoryLayout<UInt32>.size
        guard data.count % width == 0 else {
            throw TokensCodecError.truncatedTokensFile(byteCount: data.count)
        }
        var result: [Int] = []
        result.reserveCapacity(data.count / width)
        var index = data.startIndex
        while index < data.endIndex {
            let next = data.index(index, offsetBy: width)
            let littleEndianValue = data[index..<next].withUnsafeBytes { $0.load(as: UInt32.self) }
            result.append(Int(UInt32(littleEndian: littleEndianValue)))
            index = next
        }
        return result
    }

    /// Hex sha256 digest of `data`.
    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Listing

    /// Checkpoints under `session`, sorted oldest-first by
    /// `InfiniteSessionManifest.createdAt`. Returns `[]` for a session with
    /// no directory yet rather than throwing.
    public func listCheckpoints(session: String) throws -> [(id: String, manifest: InfiniteSessionManifest)] {
        let dir = sessionDirectory(session)
        guard FileManager.default.fileExists(atPath: dir.path) else { return [] }

        let entries = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        let checkpoints: [(id: String, manifest: InfiniteSessionManifest)] = try entries
            .filter { entry in
                var isDirectory: ObjCBool = false
                return FileManager.default.fileExists(atPath: entry.path, isDirectory: &isDirectory)
                    && isDirectory.boolValue
            }
            .map { entry in
                let id = entry.lastPathComponent
                return (id, try readManifest(session: session, checkpoint: id))
            }

        return checkpoints.sorted { $0.manifest.createdAt < $1.manifest.createdAt }
    }

    /// The most recently created checkpoint under `session`, or `nil` if
    /// none exist.
    public func latestCheckpoint(session: String) throws -> (id: String, manifest: InfiniteSessionManifest)? {
        try listCheckpoints(session: session).last
    }

    // MARK: - Fork

    /// Clones `session/checkpoint`'s on-disk contents into a new checkpoint
    /// (`newCheckpointId`) under `intoSession` (which may be `session`
    /// itself, for a sibling branch, or a different session id), then
    /// rewrites the cloned manifest's `parentCheckpointId`, `label`, and
    /// `createdAt`.
    ///
    /// Tries `clonefile(2)` first (an APFS copy-on-write clone of the whole
    /// checkpoint directory in one syscall; cheap even for a multi-GB
    /// `cache.safetensors`), falling back to `FileManager.copyItem` (logged)
    /// when the volume doesn't support it.
    public func fork(
        session: String,
        checkpoint: String,
        intoSession newSession: String,
        newCheckpointId: String,
        label: String?
    ) throws {
        let sourceDir = sessionDirectory(session).appendingPathComponent(checkpoint, isDirectory: true)
        guard FileManager.default.fileExists(atPath: sourceDir.path) else {
            throw InfiniteSessionStoreError.checkpointNotFound(session: session, checkpoint: checkpoint)
        }

        let destSessionDir = sessionDirectory(newSession)
        try FileManager.default.createDirectory(at: destSessionDir, withIntermediateDirectories: true)
        let destDir = destSessionDir.appendingPathComponent(newCheckpointId, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: destDir.path) else {
            throw InfiniteSessionStoreError.checkpointAlreadyExists(
                session: newSession,
                checkpoint: newCheckpointId
            )
        }

        if !cloneDirectory(from: sourceDir, to: destDir) {
            sessionStoreLogger.notice(
                "clonefile(2) unavailable for \(sourceDir.path, privacy: .public); falling back to FileManager.copyItem"
            )
            try FileManager.default.copyItem(at: sourceDir, to: destDir)
        }

        let original = try readManifest(session: session, checkpoint: checkpoint)
        let forked = InfiniteSessionManifest(
            schemaVersion: original.schemaVersion,
            model: original.model,
            tokenizerHash: original.tokenizerHash,
            tokenCount: original.tokenCount,
            pendingTokenId: original.pendingTokenId,
            tokenIdsSHA256: original.tokenIdsSHA256,
            cacheConfig: original.cacheConfig,
            createdAt: Date(),
            parentCheckpointId: checkpoint,
            label: label
        )
        try writeManifest(forked, session: newSession, checkpoint: newCheckpointId)
    }

    /// Attempts an APFS `clonefile(2)` of `source` (recursively, if a
    /// directory) to `destination`. Returns `false` on any failure (cross-
    /// volume, unsupported filesystem, `destination` already existing) so
    /// the caller can fall back to a regular copy.
    private func cloneDirectory(from source: URL, to destination: URL) -> Bool {
        source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                clonefile(sourcePath, destinationPath, 0) == 0
            }
        }
    }

    // MARK: - Delete

    /// Removes the entire session tree. A no-op (not an error) when the
    /// session has no directory.
    public func deleteSession(_ session: String) throws {
        let dir = sessionDirectory(session)
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        try FileManager.default.removeItem(at: dir)
    }

    // MARK: - Integrity

    /// Verifies `manifest` (as read from `session/checkpoint`) against the
    /// live values a resume/fork is about to attach, and recomputes
    /// `tokens.bin`'s sha256 from disk to compare against
    /// `manifest.tokenIdsSHA256`. Throws the first `SessionIntegrityError`
    /// found; the caller (a later PR) maps these to HTTP error bodies.
    public func verify(
        manifest: InfiniteSessionManifest,
        against live: LiveSessionFacts,
        session: String,
        checkpoint: String
    ) throws {
        guard manifest.schemaVersion == live.schemaVersion else {
            throw SessionIntegrityError.schemaVersionMismatch(
                expected: manifest.schemaVersion,
                actual: live.schemaVersion
            )
        }
        guard manifest.model.repoId == live.repoId else {
            throw SessionIntegrityError.repoIdMismatch(expected: manifest.model.repoId, actual: live.repoId)
        }
        guard manifest.model.revision == live.revision else {
            throw SessionIntegrityError.revisionMismatch(expected: manifest.model.revision, actual: live.revision)
        }
        guard manifest.tokenizerHash == live.tokenizerHash else {
            throw SessionIntegrityError.tokenizerHashMismatch(
                expected: manifest.tokenizerHash,
                actual: live.tokenizerHash
            )
        }

        let tokensData = try Data(contentsOf: tokensURL(session: session, checkpoint: checkpoint))
        let actualTokenIdsSHA256 = Self.sha256Hex(tokensData)
        guard manifest.tokenIdsSHA256 == actualTokenIdsSHA256 else {
            throw SessionIntegrityError.tokenIdsSHA256Mismatch(
                expected: manifest.tokenIdsSHA256,
                actual: actualTokenIdsSHA256
            )
        }
    }
}
