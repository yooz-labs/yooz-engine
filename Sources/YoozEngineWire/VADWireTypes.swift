// VADWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

public struct SpeechSegment: Codable, Sendable, Equatable {
    public let startMs: Int
    public let endMs: Int
    public let probability: Float

    public init(startMs: Int, endMs: Int, probability: Float) {
        self.startMs = startMs
        self.endMs = endMs
        self.probability = probability
    }
}

/// Response for `POST /v1/vad/detect`.
public struct VADResponse: Codable, Sendable, Equatable {
    public let segments: [SpeechSegment]

    public init(segments: [SpeechSegment]) {
        self.segments = segments
    }
}
