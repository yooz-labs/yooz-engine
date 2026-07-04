// InfiniteSessionManifest.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Model identity pinned into a session manifest at checkpoint time so a
/// later resume/fork can detect a mismatch against a different model or
/// revision than the one the checkpoint was written against (see
/// `InfiniteSessionStore.verify(manifest:against:session:checkpoint:)`).
public struct ModelIdentity: Codable, Sendable, Equatable {
    /// Wire id from `InfiniteModelSelection` (e.g. `"gemma4-e4b-1m"`).
    public let selectionId: String
    /// HuggingFace repo id, e.g. `"mlx-community/gemma-4-e4b-it-1m-4bit"`.
    public let repoId: String
    /// Pinned HuggingFace revision (commit sha).
    public let revision: String

    public init(selectionId: String, repoId: String, revision: String) {
        self.selectionId = selectionId
        self.repoId = repoId
        self.revision = revision
    }
}

/// Cache-shape and turn-handling knobs pinned into a checkpoint so a
/// resumed session reconstructs the same KV cache configuration it was
/// checkpointed with.
public struct SessionKnobs: Codable, Sendable, Equatable {
    public let kvBits: Int?
    public let kvGroupSize: Int?
    public let kvScheme: String?
    public let turnPolicy: String

    public init(
        kvBits: Int? = nil,
        kvGroupSize: Int? = nil,
        kvScheme: String? = nil,
        turnPolicy: String = "turn_commit"
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.kvScheme = kvScheme
        self.turnPolicy = turnPolicy
    }
}

/// On-disk manifest for one Infinite session checkpoint.
///
/// `InfiniteSessionStore` writes/reads this as `manifest.json` alongside
/// `cache.safetensors` (written and read by the MLX backend, added in a
/// later PR) and `tokens.bin` (this PR) inside one checkpoint directory.
/// No MLX imports here: the manifest is plain `Codable` data so the store
/// stays usable from contexts that don't link MLX.
public struct InfiniteSessionManifest: Codable, Sendable, Equatable {
    /// Bumped on breaking on-disk format changes.
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let model: ModelIdentity
    /// Hex sha256 identifying the tokenizer in use, pinned so a resume
    /// can't silently attach a different tokenizer than the one the
    /// token ids in `tokens.bin` were produced with.
    public let tokenizerHash: String
    public let tokenCount: Int
    /// Token id awaiting a turn-commit decision (see
    /// `SessionKnobs.turnPolicy`); nil when there is no pending token.
    public let pendingTokenId: Int?
    /// Hex sha256 of `tokens.bin`'s raw bytes, recomputed and compared by
    /// `InfiniteSessionStore.verify(manifest:against:session:checkpoint:)`.
    public let tokenIdsSHA256: String
    public let cacheConfig: SessionKnobs
    /// Creation time as whole milliseconds since the Unix epoch. Stored as
    /// an integer so a written manifest always compares equal to its own
    /// read-back: round-tripping a `Date` through JSON is not bit-exact
    /// (reference-date conversion plus decimal text), which made the
    /// round-trip test flake roughly 1-in-3 before this.
    public let createdAtMs: Int64

    public var createdAt: Date {
        Date(timeIntervalSince1970: Double(createdAtMs) / 1000)
    }
    /// Checkpoint this one was forked/derived from; nil for a session's
    /// root checkpoint.
    public let parentCheckpointId: String?
    public let label: String?

    public init(
        schemaVersion: Int = InfiniteSessionManifest.currentSchemaVersion,
        model: ModelIdentity,
        tokenizerHash: String,
        tokenCount: Int,
        pendingTokenId: Int? = nil,
        tokenIdsSHA256: String,
        cacheConfig: SessionKnobs,
        createdAt: Date = Date(),
        parentCheckpointId: String? = nil,
        label: String? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.model = model
        self.tokenizerHash = tokenizerHash
        self.tokenCount = tokenCount
        self.pendingTokenId = pendingTokenId
        self.tokenIdsSHA256 = tokenIdsSHA256
        self.cacheConfig = cacheConfig
        self.createdAtMs = Int64((createdAt.timeIntervalSince1970 * 1000).rounded())
        self.parentCheckpointId = parentCheckpointId
        self.label = label
    }
}
