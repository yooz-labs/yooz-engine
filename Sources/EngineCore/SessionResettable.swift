// SessionResettable.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Protocol any stateful backend conforms to so it participates in the
/// per-recording-session reset boundary (engine issue #114).
///
/// In v0.6.0 the engine became out-of-process. State that used to be reset
/// implicitly by the consumer app's actor lifecycle (one whisper process =
/// one session of LLM/STT state) is now persistent across recordings because
/// the helper survives them. Concrete consequences:
///
/// - **LLM**: KV / system-prompt caches leak previous-recording context into
///   the next call, producing echoed output or aggressive summarization.
/// - **STT**: streaming transcribers accumulate audio + token buffers across
///   recordings.
/// - **Future modules** (TTS, RAG, multi-turn): inherit the same shape.
///
/// Per-model patches don't scale — N+ models and M+ modules over time would
/// each need their own endpoint and client call site. Instead, every
/// stateful backend conforms to `SessionResettable` and the server fans out
/// `POST /v1/session/begin` and `POST /v1/session/end` to all conformers.
///
/// ## Contract
///
/// - **Idempotent.** Safe to call on a backend that has no state yet, or to
///   call repeatedly. Implementations should treat the call as "drop
///   anything that depends on the previous recording".
/// - **Cheap.** Resetting state is in the call path of every recording
///   (begin) and every paste (end). Implementations must not block on
///   network or model loads.
/// - **Must NOT unload models or invalidate long-lived weights.** This
///   boundary is per-*recording*, not per-process. Tearing down a model
///   here would force a re-load on the very next call. Drop KV caches,
///   partial buffers, streaming context — keep weights.
public protocol SessionResettable: Sendable {
    /// Drop per-recording-session state (KV caches, partial buffers,
    /// streaming context, etc.). Idempotent. MUST NOT unload models or
    /// invalidate long-lived weights.
    func resetForNewSession() async
}
