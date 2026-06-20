// InfinitePickerTypes.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// One row in the Infinite model picker.
///
/// The canonical picker fields match the engine-wide pattern. The
/// Infinite-specific fields are optional extension metadata for hosts
/// that want to display context length, RAM tier, and benchmark provenance.
public struct InfiniteModelInfo: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let description: String
    public let tier: ModelTier
    public let sizeBytes: Int64?
    public let loadState: ModelLoadState
    public let isActive: Bool
    public let maxContextTokens: Int?
    public let nativeContextTokens: Int?
    public let ramTier: String?
    public let backendKind: String?
    public let adapterKind: String?
    public let huggingFaceID: String?
    public let revision: String?
    public let requiresAppleSilicon: Bool
    public let evidenceRef: String?

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: ModelTier,
        sizeBytes: Int64?,
        loadState: ModelLoadState,
        isActive: Bool,
        maxContextTokens: Int?,
        nativeContextTokens: Int?,
        ramTier: String?,
        backendKind: String?,
        adapterKind: String?,
        huggingFaceID: String?,
        revision: String?,
        requiresAppleSilicon: Bool,
        evidenceRef: String?
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.tier = tier
        self.sizeBytes = sizeBytes
        self.loadState = loadState
        self.isActive = isActive
        self.maxContextTokens = maxContextTokens
        self.nativeContextTokens = nativeContextTokens
        self.ramTier = ramTier
        self.backendKind = backendKind
        self.adapterKind = adapterKind
        self.huggingFaceID = huggingFaceID
        self.revision = revision
        self.requiresAppleSilicon = requiresAppleSilicon
        self.evidenceRef = evidenceRef
    }
}

public struct InfiniteModelsResponse: Codable, Sendable, Equatable {
    public let models: [InfiniteModelInfo]
    public let activeId: String

    public init(models: [InfiniteModelInfo], activeId: String) {
        self.models = models
        self.activeId = activeId
    }
}

public struct InfiniteSetModelRequest: Codable, Sendable, Equatable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}

public struct InfiniteStatus: Codable, Sendable, Equatable {
    public let loaded: Bool
    public let modelId: String
    public let progress: Double?
    public let state: String
    public let activeSessions: Int
    public let maxContextTokens: Int?
    public let ramTier: String?
    public let backendKind: String?
    public let cleanupPolicy: String?
    public let resources: InfiniteResourceMetrics?
    public let lastError: String?

    public init(
        loaded: Bool,
        modelId: String,
        progress: Double?,
        state: String,
        activeSessions: Int,
        maxContextTokens: Int?,
        ramTier: String?,
        backendKind: String?,
        cleanupPolicy: String? = nil,
        resources: InfiniteResourceMetrics? = nil,
        lastError: String? = nil
    ) {
        self.loaded = loaded
        self.modelId = modelId
        self.progress = progress
        self.state = state
        self.activeSessions = activeSessions
        self.maxContextTokens = maxContextTokens
        self.ramTier = ramTier
        self.backendKind = backendKind
        self.cleanupPolicy = cleanupPolicy
        self.resources = resources
        self.lastError = lastError
    }
}
