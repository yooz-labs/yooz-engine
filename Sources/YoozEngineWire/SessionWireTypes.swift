// SessionWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Response for `POST /v1/session/begin` (engine issue #114).
///
/// `sessionId` is a fresh UUID per call. Engine state itself doesn't pin to
/// the value — `begin` is idempotent and unconditionally fans out
/// `resetForNewSession()` to every `SessionResettable` module — but
/// returning a UUID lets consumer apps tag their own logs / metrics so a
/// recording can be correlated end-to-end across engine + client traces.
/// `ts` is an ISO-8601 UTC timestamp captured server-side at fan-out start.
///
/// Single definition shared by the loopback server (`APIServer`) and
/// `InProcessTransport` — previously two independent structs (#225).
public struct SessionBeginResponse: Codable, Sendable, Equatable {
    public let sessionId: String
    public let ts: String

    public init(sessionId: String, ts: String) {
        self.sessionId = sessionId
        self.ts = ts
    }
}
