// InfiniteSessionTypes.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

public struct InfiniteResourceMetrics: Codable, Sendable, Equatable {
    public let physicalMemoryBytes: Int64
    public let wiredMemoryLimitBytes: Int64
    public let requiredRAMTier: String
    public let peakMemoryBytes: Int64?
    public let prefillTokensPerSecond: Double?
    public let decodeTokensPerSecond: Double?
    public let draftAcceptanceRate: Double?

    public init(
        physicalMemoryBytes: Int64,
        wiredMemoryLimitBytes: Int64,
        requiredRAMTier: String,
        peakMemoryBytes: Int64? = nil,
        prefillTokensPerSecond: Double? = nil,
        decodeTokensPerSecond: Double? = nil,
        draftAcceptanceRate: Double? = nil
    ) {
        self.physicalMemoryBytes = physicalMemoryBytes
        self.wiredMemoryLimitBytes = wiredMemoryLimitBytes
        self.requiredRAMTier = requiredRAMTier
        self.peakMemoryBytes = peakMemoryBytes
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.draftAcceptanceRate = draftAcceptanceRate
    }
}

public struct InfiniteSessionInfo: Codable, Sendable, Equatable {
    public let id: String
    public let modelId: String
    public let label: String?
    public let state: String
    public let createdAt: String
    public let updatedAt: String
    public let contextWindowTokens: Int
    public let inputCharacters: Int
    public let estimatedInputTokens: Int
    public let checkpointCount: Int
    public let cleanupPolicy: String
    public let resources: InfiniteResourceMetrics

    public init(
        id: String,
        modelId: String,
        label: String?,
        state: String,
        createdAt: String,
        updatedAt: String,
        contextWindowTokens: Int,
        inputCharacters: Int,
        estimatedInputTokens: Int,
        checkpointCount: Int,
        cleanupPolicy: String,
        resources: InfiniteResourceMetrics
    ) {
        self.id = id
        self.modelId = modelId
        self.label = label
        self.state = state
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.contextWindowTokens = contextWindowTokens
        self.inputCharacters = inputCharacters
        self.estimatedInputTokens = estimatedInputTokens
        self.checkpointCount = checkpointCount
        self.cleanupPolicy = cleanupPolicy
        self.resources = resources
    }
}

public struct InfiniteSessionsResponse: Codable, Sendable, Equatable {
    public let sessions: [InfiniteSessionInfo]

    public init(sessions: [InfiniteSessionInfo]) {
        self.sessions = sessions
    }
}

public struct InfiniteCreateSessionRequest: Codable, Sendable, Equatable {
    public let modelId: String?
    public let label: String?
    /// `"turn_commit"` (default) or `"thinking_in_session"` — see
    /// `SessionKnobs.turnPolicy`/`InfiniteTurnPolicy` (engine#267). `nil`/
    /// empty defaults to `"turn_commit"`; any other value is rejected.
    public let turnPolicy: String?

    public init(modelId: String? = nil, label: String? = nil, turnPolicy: String? = nil) {
        self.modelId = modelId
        self.label = label
        self.turnPolicy = turnPolicy
    }
}

public struct InfiniteAppendSessionRequest: Codable, Sendable, Equatable {
    public let text: String

    public init(text: String) {
        self.text = text
    }
}

public struct InfiniteAppendSessionResponse: Codable, Sendable, Equatable {
    public let session: InfiniteSessionInfo
    public let appendedCharacters: Int
    public let estimatedAppendedTokens: Int

    public init(
        session: InfiniteSessionInfo,
        appendedCharacters: Int,
        estimatedAppendedTokens: Int
    ) {
        self.session = session
        self.appendedCharacters = appendedCharacters
        self.estimatedAppendedTokens = estimatedAppendedTokens
    }
}

public struct InfiniteGenerateSessionRequest: Codable, Sendable, Equatable {
    public let prompt: String?
    public let maxTokens: Int?
    /// Sampling temperature; `nil` preserves the production default (0.7).
    /// `0` selects greedy decoding — used by the live session parity test
    /// and any caller that needs deterministic output (engine#265).
    public let temperature: Double?

    public init(prompt: String? = nil, maxTokens: Int? = nil, temperature: Double? = nil) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

public struct InfiniteGenerateSessionResponse: Codable, Sendable, Equatable {
    public let sessionId: String
    public let text: String
    public let finishReason: String
    public let resources: InfiniteResourceMetrics
    /// Turn-commit (engine#267) stats — `nil` for `"thinking_in_session"`
    /// sessions, where there is no separate reasoning/commit bucket.
    /// Reasoning-side token count, approximated by re-encoding the split-out
    /// reasoning text.
    public let thinkingTokens: Int?
    /// Exact token count committed to the durable cache this call (user
    /// turn plus the stable-framed answer).
    public let committedTokens: Int?
    /// Wall-clock seconds spent chunk-prefilling the commit onto the
    /// durable cache.
    public let commitSeconds: Double?

    public init(
        sessionId: String,
        text: String,
        finishReason: String,
        resources: InfiniteResourceMetrics,
        thinkingTokens: Int? = nil,
        committedTokens: Int? = nil,
        commitSeconds: Double? = nil
    ) {
        self.sessionId = sessionId
        self.text = text
        self.finishReason = finishReason
        self.resources = resources
        self.thinkingTokens = thinkingTokens
        self.committedTokens = committedTokens
        self.commitSeconds = commitSeconds
    }
}

public struct InfiniteCheckpointSessionRequest: Codable, Sendable, Equatable {
    public let label: String?
    /// When `true`, the engine releases the session's live KV cache from
    /// RAM after checkpointing (session `state` becomes `"parked"`).
    /// `false`/`nil` leaves the session hot.
    public let park: Bool?

    public init(label: String? = nil, park: Bool? = nil) {
        self.label = label
        self.park = park
    }
}

public struct InfiniteSessionCheckpoint: Codable, Sendable, Equatable {
    public let id: String
    public let label: String?
    public let createdAt: String
    public let inputCharacters: Int
    public let estimatedInputTokens: Int
    public let resources: InfiniteResourceMetrics

    public init(
        id: String,
        label: String?,
        createdAt: String,
        inputCharacters: Int,
        estimatedInputTokens: Int,
        resources: InfiniteResourceMetrics
    ) {
        self.id = id
        self.label = label
        self.createdAt = createdAt
        self.inputCharacters = inputCharacters
        self.estimatedInputTokens = estimatedInputTokens
        self.resources = resources
    }
}

public struct InfiniteCheckpointSessionResponse: Codable, Sendable, Equatable {
    public let session: InfiniteSessionInfo
    public let checkpoint: InfiniteSessionCheckpoint
    /// Bytes written to `cache.safetensors` for this checkpoint.
    public let sizeBytes: Int64
    /// `tokenRecord.count` at checkpoint time (durable tokens plus the one
    /// pending token, if any) — the exact figure the manifest was written
    /// with, duplicated here so a caller doesn't need to cross-reference
    /// `checkpoint.estimatedInputTokens`.
    public let tokenCount: Int
    /// Wall-clock seconds spent branching the live KV cache and writing
    /// `cache.safetensors` (excludes tokens.bin/manifest.json, which are
    /// negligible by comparison).
    public let durationSeconds: Double
    /// The checkpoint this one supersedes for the same session, if any.
    public let parentCheckpointId: String?

    public init(
        session: InfiniteSessionInfo,
        checkpoint: InfiniteSessionCheckpoint,
        sizeBytes: Int64,
        tokenCount: Int,
        durationSeconds: Double,
        parentCheckpointId: String? = nil
    ) {
        self.session = session
        self.checkpoint = checkpoint
        self.sizeBytes = sizeBytes
        self.tokenCount = tokenCount
        self.durationSeconds = durationSeconds
        self.parentCheckpointId = parentCheckpointId
    }
}

/// Body for `POST /v1/infinite/sessions/:id/resume`.
public struct InfiniteResumeSessionRequest: Codable, Sendable, Equatable {
    /// Checkpoint to resume from; defaults to the session's latest
    /// checkpoint when omitted.
    public let checkpointId: String?

    public init(checkpointId: String? = nil) {
        self.checkpointId = checkpointId
    }
}

/// Body for `POST /v1/infinite/sessions/:id/fork`.
public struct InfiniteForkSessionRequest: Codable, Sendable, Equatable {
    /// Checkpoint to fork from; defaults to the source session's latest
    /// checkpoint when omitted (a hot, never-checkpointed source takes an
    /// implicit checkpoint first).
    public let checkpointId: String?
    public let label: String?

    public init(checkpointId: String? = nil, label: String? = nil) {
        self.checkpointId = checkpointId
        self.label = label
    }
}

public struct InfiniteDeleteSessionResponse: Codable, Sendable, Equatable {
    public let sessionId: String
    public let deleted: Bool

    public init(sessionId: String, deleted: Bool) {
        self.sessionId = sessionId
        self.deleted = deleted
    }
}
