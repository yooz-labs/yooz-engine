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
    /// `InProcessTransport.openSTTStream(language:mode:)`. Documents the
    /// equivalence so the route-parity test can certify the route as covered
    /// WITHOUT adding it to `RouteParityAllowlist` — it genuinely IS served
    /// in-process, just not via a `route(method:path:)`-shaped call.
    public let inProcessEquivalent: String?

    public init(_ method: RouteMethod, _ path: String, inProcessEquivalent: String? = nil) {
        self.method = method
        self.path = path
        self.inProcessEquivalent = inProcessEquivalent
    }

    /// Stable identity for set membership / dictionary keys across the
    /// manifest and the allowlist.
    public var key: String { "\(method.rawValue) \(path)" }
}

/// The tactical route manifest (#223): every route `APIServer` registers on
/// the loopback Hummingbird server. `Tests/YoozEngineInProcessTests/RouteParityTests.swift`
/// walks this list and fails if a non-allowlisted entry is not reachable
/// through `InProcessTransport` — so a new `APIServer` route with no
/// in-process handler and no `RouteParityAllowlist` entry breaks the build
/// instead of silently drifting (the `/v1/session/*` gap that motivated this
/// issue).
///
/// KEEP IN SYNC BY HAND with `APIServer`'s `router.get/post/delete/ws(...)`
/// calls when adding, removing, or renaming a route. There is no automated
/// check today that this list matches `APIServer`'s registrations (the
/// `APIServer` half of the parity check needs the app-hosted Xcode test
/// target; blocked on #105 / #202 — see the comment above `buildRouter()` in
/// `APIServer.swift`). This SPM-side manifest is the source of truth for the
/// in-process half, which SPM tests do enforce automatically.
public enum RouteManifest {
    public static let all: [RouteManifestEntry] = [
        .init(.get, "/v1/health"),
        .init(.get, "/v1/modules"),
        .init(.get, "/v1/models"),
        .init(.delete, "/v1/models/:id"),
        .init(.post, "/v1/models/cleanup"),

        .init(.post, "/v1/session/begin"),
        .init(.post, "/v1/session/end"),

        .init(.post, "/v1/llm/generate"),
        .init(.get, "/v1/llm/models"),
        .init(.get, "/v1/llm/status"),
        .init(.post, "/v1/llm/model"),
        .init(.post, "/v1/llm/preload"),
        .init(.post, "/v1/llm/unload"),

        .init(.get, "/v1/infinite/models"),
        .init(.post, "/v1/infinite/model"),
        .init(.get, "/v1/infinite/status"),
        .init(.get, "/v1/infinite/sessions"),
        .init(.post, "/v1/infinite/sessions"),
        .init(.get, "/v1/infinite/sessions/:sessionID"),
        .init(.post, "/v1/infinite/sessions/:sessionID/append"),
        .init(.post, "/v1/infinite/sessions/:sessionID/generate"),
        .init(.post, "/v1/infinite/sessions/:sessionID/checkpoint"),
        .init(.delete, "/v1/infinite/sessions/:sessionID"),

        .init(.post, "/v1/touchup"),
        .init(.get, "/v1/touchup/models"),
        .init(.post, "/v1/touchup/model"),

        .init(.post, "/v1/grammar/check"),

        .init(.post, "/v1/vad/detect"),

        .init(.get, "/v1/stt/engine"),
        .init(.post, "/v1/stt/engine"),
        .init(.get, "/v1/stt/languages"),
        .init(.get, "/v1/stt/status"),
        .init(.post, "/v1/stt/load"),
        .init(.post, "/v1/stt/batch"),
        // The in-process transport has no socket, so there is no literal WS
        // upgrade to dispatch. `openSTTStream` is the equivalent call for
        // every backend except qwen3-preview (see `RouteParityAllowlist`
        // doc comment below for why that gap is not its own manifest entry).
        .init(
            .websocket, "/v1/stt/stream",
            inProcessEquivalent: "InProcessTransport.openSTTStream(language:mode:)"
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
    public static let loopbackOnly: [LoopbackOnlyRoute] = [
        // #222 (in flight, concurrent branch) adds /v1/session/{begin,end} to
        // InProcessTransport. Remove exactly these two entries when that PR
        // lands — whichever of #222 / #223 lands second rebases past the
        // other, per the coordination note on both issues.
        LoopbackOnlyRoute(
            .init(.post, "/v1/session/begin"),
            reason: "per-recording session reset not yet routed in-process (#222 in flight)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/session/end"),
            reason: "per-recording session reset not yet routed in-process (#222 in flight)"
        ),

        // Infinite is loopback-only by design: its only consumer is the
        // super-yooz host, which always talks loopback HTTP.
        // `YoozEngineInProcess` does not even depend on the Infinite module
        // (Package.swift ~200-203).
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/models"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/model"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/status"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/sessions"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.get, "/v1/infinite/sessions/:sessionID"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/append"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/generate"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.post, "/v1/infinite/sessions/:sessionID/checkpoint"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
        LoopbackOnlyRoute(
            .init(.delete, "/v1/infinite/sessions/:sessionID"),
            reason: "Infinite is loopback-only by design; consumer is the super-yooz host (Package.swift ~200-203)"
        ),
    ]
}
