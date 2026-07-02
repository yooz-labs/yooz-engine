// SessionEndpoints.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os

private let logger = Logger(subsystem: "live.yooz.engine", category: "session-endpoints")

/// Table endpoints for the `/v1/session/*` family (engine#225 Phase B).
///
/// The session family's only dependency is `SessionCoordinator` (EngineCore),
/// so — unlike the module-backed families — the handler bodies can live here
/// and be truly single-homed: the loopback server and the in-process
/// transport bind the exact same closures. Fitting, since the `/v1/session`
/// in-process gap (#222) is the drift this whole table exists to prevent.
public enum SessionEndpoints {
    public static func endpoints() -> [Endpoint] {
        [
            // `POST /v1/session/begin` (engine#114/#222): idempotent
            // per-recording reset fan-out to every `SessionResettable`
            // module; returns `{sessionId, ts}` so consumer apps can tag
            // their logs/metrics for end-to-end correlation.
            Endpoint(EndpointSpecs.sessionBegin) { _ in
                let result = await SessionCoordinator.begin()
                logger.debug(
                    "Session begin: id=\(result.sessionId) fanout=\(result.fanoutCount)"
                )
                return try WireResponse.json(SessionBeginResponse(
                    sessionId: result.sessionId,
                    ts: result.ts
                ))
            },
            // `POST /v1/session/end`: same fan-out; 204 No Content on the
            // wire (the in-process transport surfaces that as empty `Data`,
            // matching what `HTTPTransport` produces for a 204).
            Endpoint(EndpointSpecs.sessionEnd) { _ in
                let fanoutCount = await SessionCoordinator.end()
                logger.debug("Session end: fanout=\(fanoutCount)")
                return .noContent
            },
        ]
    }
}
