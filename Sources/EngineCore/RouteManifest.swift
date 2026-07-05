// RouteManifest.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// The wire method of a `RouteManifestEntry`. `websocket` routes are not
/// dispatched through the `EngineTransport` REST verbs (`get`/`post`/
/// `delete`); see `RouteManifestEntry.inProcessEquivalent`.
public enum RouteMethod: String, Sendable, CaseIterable {
    case get = "GET"
    case post = "POST"
    case delete = "DELETE"
    case websocket = "WS"
}

/// One `(method, path-pattern)` entry as registered on the loopback
/// `APIServer` (`YoozEngine/Server/APIServer.swift`). Path parameters use the
/// same `:name` placeholder Hummingbird's router uses (e.g.
/// `/v1/infinite/sessions/:sessionID`) so an entry reads like the router
/// registration it mirrors.
public struct RouteManifestEntry: Sendable, Equatable {
    public let method: RouteMethod
    public let path: String

    /// Non-nil only for a route that is reachable in-process through a
    /// different call shape than `EngineTransport.get/post/delete` — today
    /// just the STT WebSocket stream, whose in-process equivalent is
    /// `InProcessTransport.openSTTStream(language:mode:)`. Documentation
    /// only: the parity test never reads this string — its dispatch
    /// mechanism is the hardcoded `.websocket` case in
    /// `RouteParityTests.dispatch()`. The field exists so the manifest
    /// itself records WHY the route needs no `RouteParityAllowlist` entry:
    /// it genuinely IS served in-process, just not via a
    /// `route(method:path:)`-shaped call.
    public let inProcessEquivalent: String?

    public init(_ method: RouteMethod, _ path: String, inProcessEquivalent: String? = nil) {
        self.method = method
        self.path = path
        self.inProcessEquivalent = inProcessEquivalent
    }

    /// Stable identity for set membership / dictionary keys across the
    /// manifest and the allowlist. Intentionally derived from `method` +
    /// `path` only — unlike the synthesized `==`, it ignores
    /// `inProcessEquivalent` — so two entries for the same wire route always
    /// collide on `key`, which is what the duplicate-detection test checks.
    public var key: String { "\(method.rawValue) \(path)" }
}

/// The route manifest (#223): every route `APIServer` registers on the
/// loopback Hummingbird server.
/// `Tests/YoozEngineInProcessTests/RouteParityTests.swift` walks this list
/// and fails if a non-allowlisted entry is not reachable through
/// `InProcessTransport` — so a new `APIServer` route with no in-process
/// handler and no `RouteParityAllowlist` entry breaks the build instead of
/// silently drifting (the `/v1/session/*` gap that motivated this issue).
///
/// Since #225 Phase B this is a PROJECTION of the `EndpointSpecs` catalog —
/// the single declaration of every REST route — plus the one WebSocket
/// route, which is not a REST-dispatchable endpoint (no
/// `EndpointTable`-shaped handler; its in-process equivalent is
/// `InProcessTransport.openSTTStream(language:mode:)`). Add or rename a
/// route in `EndpointSpecs`, not here; the manifest, the endpoint table,
/// and the parity test all follow automatically.
public enum RouteManifest {
    public static let all: [RouteManifestEntry] =
        EndpointSpecs.all.map { RouteManifestEntry($0.method, $0.path) } + [
            // The in-process transport has no socket, so there is no literal
            // WS upgrade to dispatch. `openSTTStream` is the equivalent call
            // for every backend except qwen3-preview (see
            // `RouteParityAllowlist` doc comment below for why that gap is
            // not its own manifest entry).
            RouteManifestEntry(
                .websocket, "/v1/stt/stream",
                inProcessEquivalent: "InProcessTransport.openSTTStream(language:mode:)"
            ),
            // `/v1/events` (engine#226): same shape as the STT stream entry
            // above — a WebSocket upgrade on loopback, served in-process by
            // `openEvents()` (an `EngineTransport` requirement mirroring
            // `openSTTStream`) rather than a literal socket. Reachable on
            // ALL THREE transports as of engine#244 (`XPCTransport.openEvents()`
            // bridges it over the callback-proxy shape `XPCServiceHandler`
            // already uses for STT streaming), so it needs no
            // `RouteParityAllowlist` entry — that allowlist only governs
            // in-process parity, and XPC has no equivalent gate since parity
            // there is proven by `XPCRoundTripTests`/`XPCStreamingTests`
            // instead of a manifest-driven check.
            RouteManifestEntry(
                .websocket, "/v1/events",
                inProcessEquivalent: "InProcessTransport.openEvents()"
            ),
        ]
}

/// One `loopbackOnly` declaration: a manifest entry `InProcessTransport`
/// intentionally does not (yet, or ever) serve, with the one-line reason a
/// reviewer needs to judge whether the gap is still justified.
public struct LoopbackOnlyRoute: Sendable {
    public let entry: RouteManifestEntry
    public let reason: String

    public init(_ entry: RouteManifestEntry, reason: String) {
        self.entry = entry
        self.reason = reason
    }
}

/// Routes intentionally NOT served by `InProcessTransport`. Every entry here
/// is a deliberate, reviewed decision — not a silent gap. Adding a new
/// `APIServer` route with no in-process handler and no allowlist line here
/// fails `RouteParityTests` (`Tests/YoozEngineInProcessTests`).
///
/// ## Why "qwen3 streaming preview" has no manifest entry of its own
///
/// `InProcessTransport.openSTTStream` throws `unsupportedOperation` when the
/// active STT backend is `.qwen3ASRPreview` (loopback/dev only; unstable,
/// engine#154) — a real in-process gap, but a backend-conditional one WITHIN
/// the single `/v1/stt/stream` route, not a distinct wire route `APIServer`
/// registers. Representing it as its own manifest entry would need a
/// synthetic path with no matching `router.ws(...)` registration, breaking
/// the manifest's 1:1 correspondence to real `APIServer` registrations that
/// the (future) app-target half of this check relies on. The gap is instead
/// pinned by `InProcessTransportTests.testInProcessQwen3StreamingIsUnsupported`
/// and called out here for visibility. The same applies to the `.qwen3ASRPreview`
/// branch of `POST /v1/stt/load` ("load qwen3 preview").
public enum RouteParityAllowlist {
    /// Shared justification for every `/v1/infinite/*` entry — one string so
    /// the rationale can never drift between entries on a partial edit.
    private static let infiniteReason =
        "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"

    public static let loopbackOnly: [LoopbackOnlyRoute] = [
        // /v1/session/{begin,end} were allowlisted here while #222 was in
        // flight; that PR landed (SessionCoordinator fan-out shared by both
        // transports), so the routes are asserted as in-process-reachable by
        // the parity test like any other.

        // Infinite is loopback-only by design: its only consumer is the
        // super-yooz host, which always talks loopback HTTP.
        // `YoozEngineInProcess` does not even depend on the Infinite module
        // (Package.swift ~200-203).
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/models"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/model"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/status"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/sessions"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/sessions/:sessionID"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/append"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/generate"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/checkpoint"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/resume"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/fork"),
            reason: infiniteReason
        ),
        LoopbackOnlyRoute(
            .init(.delete, "/v1/infinite/sessions/:sessionID"),
            reason: infiniteReason
        ),
    ]
}
