// SessionCoordinator.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Result of `SessionCoordinator.begin()`: the wire payload for
/// `POST /v1/session/begin` plus the fan-out count for caller-side logging.
///
/// `sessionId` is a fresh UUID per call. Engine state itself doesn't pin to
/// the value — `begin` is idempotent and unconditionally fans out
/// `resetForNewSession()` to every `SessionResettable` module — but
/// returning a UUID lets consumer apps tag their own logs / metrics so a
/// recording can be correlated end-to-end across engine + client traces.
/// `ts` is an ISO-8601 UTC timestamp captured server-side at fan-out start.
///
/// Deliberately NOT `Codable`: `fanoutCount` is caller-side telemetry, not
/// part of the wire contract. Each transport encodes only `sessionId` + `ts`
/// through its own wire type (`SessionBeginResponse` on the loopback server,
/// `SessionBeginBody` in `InProcessTransport`), so conforming this struct to
/// `Codable` and encoding it directly would leak `fanoutCount` onto the wire.
public struct SessionBeginResult: Sendable {
    public let sessionId: String
    public let ts: String
    public let fanoutCount: Int
}

/// Single source of truth for the per-recording session-reset fan-out
/// (engine issue #114 / #222). Every transport that serves
/// `POST /v1/session/{begin,end}` — the loopback HTTP server today, the
/// in-process facade, and the future XPC service — calls this instead of
/// re-implementing the fan-out, so the wire contract can't drift per
/// transport. See `docs/engine-app-packaging.md`, "What the engine must
/// add", item 3.
///
/// Resets are cheap and weight-preserving by contract; they MUST NOT unload
/// models. See `SessionResettable`.
public enum SessionCoordinator {
    /// Shared formatter for the `ts` field. `ISO8601DateFormatter` is
    /// thread-safe and the default options produce a UTC `Z`-suffixed
    /// `yyyy-MM-ddTHH:mm:ssZ` string. Cached statically so each `begin` call
    /// reuses one formatter instance instead of allocating + configuring a
    /// fresh one per request.
    private static let timestampFormatter = ISO8601DateFormatter()

    /// Fan out `resetForNewSession()` to every registered `SessionResettable`
    /// module and mint a fresh session id + timestamp for
    /// `POST /v1/session/begin`.
    public static func begin() async -> SessionBeginResult {
        let sessionId = UUID().uuidString
        let ts = timestampFormatter.string(from: Date())
        let fanoutCount = await resetAll()
        return SessionBeginResult(sessionId: sessionId, ts: ts, fanoutCount: fanoutCount)
    }

    /// Fan out `resetForNewSession()` for `POST /v1/session/end`. No wire
    /// payload — callers return an empty/204-equivalent response. Returns
    /// the fan-out count for caller-side logging.
    @discardableResult
    public static func end() async -> Int {
        await resetAll()
    }

    private static func resetAll() async -> Int {
        let resettables = await ModuleRegistry.shared.allResettable()
        for module in resettables {
            await module.resetForNewSession()
        }
        return resettables.count
    }
}
