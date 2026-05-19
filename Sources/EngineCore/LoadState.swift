// LoadState.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Lifecycle state for an in-flight model load (engine#125).
///
/// Lives in `EngineCore` so both the engine app target and the
/// STT/LLM modules can produce values of this type, plus the
/// `YoozEngineClient` SDK consumes the wire shape with matching
/// `rawValue`s. Decode-safe: pre-#125 servers omit the field
/// entirely, so SDK consumers see `state == nil` and infer state
/// from `loaded` + `progress`.
///
/// Transitions per load:
/// - `.idle → .loading` when a load is enqueued
/// - `.loading → .ready` when the load succeeds
/// - `.loading → .failed` when the load throws
/// - `.loading → .idle` when the load is cancelled
/// - `.failed | .ready → .loading` when a new load is enqueued
/// - `any → .idle` on `stop()` / `unload()`
public enum LoadState: String, Codable, Sendable, Equatable {
    /// No model is loaded and no load is in flight. A picker may
    /// start a fresh load.
    case idle
    /// A load (download + init) is currently in flight.
    case loading
    /// The active model is loaded and serving requests.
    case ready
    /// The last load attempt failed. The accompanying `lastError`
    /// field on the response carries the human-readable message;
    /// the next load enqueue clears the failed state and retries.
    case failed
}
