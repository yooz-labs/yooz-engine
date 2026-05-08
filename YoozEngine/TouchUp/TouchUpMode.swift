// TouchUpMode.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Processing intensity for the `/v1/touchup` pipeline.
///
/// Module-owned enum deliberately distinct from the HTTP wire type
/// (`ServerTouchUpMode` in the app target's `APITypes.swift`). `APIServer`
/// translates between the two at the call site. Keeping the domain enum
/// here lets `LLMModule` stay independent of the server's request DTOs,
/// mirroring the Grammar module pattern (primitives at the boundary) and
/// the VAD pattern (module-owned result type, server-owned wire type).
public enum TouchUpMode: String, Sendable {
    /// Skip LLM entirely; regex voice-commands + whitespace only.
    case off
    /// Yooz-Light (0.5B): contractions, capitalization, simple grammar.
    case light
    /// Yooz-Quality when available, else Yooz-Light: standard proofreading.
    case standard
    /// Yooz-Quality comprehensive cleanup: self-corrections, fragments.
    case full
}
