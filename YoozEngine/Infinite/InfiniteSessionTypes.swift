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

    public init(modelId: String? = nil, label: String? = nil) {
        self.modelId = modelId
        self.label = label
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

    public init(prompt: String? = nil, maxTokens: Int? = nil) {
        self.prompt = prompt
        self.maxTokens = maxTokens
    }
}

public struct InfiniteGenerateSessionResponse: Codable, Sendable, Equatable {
    public let sessionId: String
    public let text: String
    public let finishReason: String
    public let resources: InfiniteResourceMetrics

    public init(
        sessionId: String,
        text: String,
        finishReason: String,
        resources: InfiniteResourceMetrics
    ) {
        self.sessionId = sessionId
        self.text = text
        self.finishReason = finishReason
        self.resources = resources
    }
}

public struct InfiniteCheckpointSessionRequest: Codable, Sendable, Equatable {
    public let label: String?

    public init(label: String? = nil) {
        self.label = label
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

    public init(
        session: InfiniteSessionInfo,
        checkpoint: InfiniteSessionCheckpoint
    ) {
        self.session = session
        self.checkpoint = checkpoint
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
