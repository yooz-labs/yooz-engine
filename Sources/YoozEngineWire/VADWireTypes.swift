// VADWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Request body for `POST /v1/vad/detect`. Single definition shared by the
/// SDK's `VADClient` (encode) and the server / in-process transport
/// (decode) — previously three independent structs (#225).
public struct VADRequest: Codable, Sendable, Equatable {
    public let samples: [Float]
    /// When true (or unset), resets the RNN hidden/cell state before
    /// detection. Set to false when sending consecutive chunks from the
    /// same recording to preserve inter-frame state continuity.
    public let reset: Bool?

    public init(samples: [Float], reset: Bool? = nil) {
        self.samples = samples
        self.reset = reset
    }
}

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
