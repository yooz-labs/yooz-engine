// EndpointSpecs.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// The single declaration of every REST route's wire location (engine#225
/// Phase B). `RouteManifest.all` is a projection of this catalog (plus the
/// one WebSocket route, which is not REST-dispatchable — see
/// `RouteManifest`), so adding, removing, or renaming a route happens HERE
/// and nowhere else.
///
/// Split into `converted` (routes whose handler is also declared once, as a
/// table `Endpoint` both transports derive from) and `legacy` (routes still
/// hand-implemented per transport — `APIServer` registration + the
/// `InProcessTransport` switch). The Phase B conversion moves specs from
/// `legacy` to `converted` family by family; the remaining-work list lives
/// on engine#225.
public enum EndpointSpecs {
    // MARK: - Converted (table-dispatched on both transports)

    // Session family — handlers in `SessionEndpoints` (EngineCore).
    public static let sessionBegin = EndpointSpec(.post, "/v1/session/begin")
    public static let sessionEnd = EndpointSpec(.post, "/v1/session/end")

    // TouchUp picker family — handlers in `TouchUpEndpoints` (LLMModule).
    public static let touchUpModels = EndpointSpec(.get, "/v1/touchup/models")
    public static let touchUpSetModel = EndpointSpec(.post, "/v1/touchup/model")

    // Model-management family — handlers in `ModelManagementEndpoints`
    // (LLMModule; the one STT-owned input is injected).
    public static let modelsInventory = EndpointSpec(.get, "/v1/models")
    public static let modelsDelete = EndpointSpec(.delete, "/v1/models/:id")
    public static let modelsCleanup = EndpointSpec(.post, "/v1/models/cleanup")

    // Engine state family (engine#226) — handlers in `EngineStateEndpoints`
    // (LLMModule; today's only contributor is the TouchUp picker). `/v1/events`
    // is NOT here: it is a WebSocket upgrade, not a REST body, so it is not
    // table-dispatchable — see `RouteManifest`'s manual entry, mirroring
    // `/v1/stt/stream`.
    public static let engineState = EndpointSpec(.get, "/v1/state")

    /// Every converted spec. Transport bindings (`APIServer.buildRouter`,
    /// `InProcessTransport`'s table) must cover exactly this set —
    /// `EndpointTableTests` pins the table side.
    public static let converted: [EndpointSpec] = [
        sessionBegin, sessionEnd,
        touchUpModels, touchUpSetModel,
        modelsInventory, modelsDelete, modelsCleanup,
        engineState,
    ]

    // MARK: - Legacy (hand-implemented per transport; conversion tracked on #225)

    // Internal: `legacy` and `all` are read only inside EngineCore (the
    // `RouteManifest` projection) and by `EngineCoreTests` via `@testable`;
    // only `converted` is a cross-module contract (the transport bindings).
    static let legacy: [EndpointSpec] = [
        EndpointSpec(.get, "/v1/health"),
        EndpointSpec(.get, "/v1/modules"),

        EndpointSpec(.post, "/v1/llm/generate"),
        EndpointSpec(.get, "/v1/llm/models"),
        EndpointSpec(.get, "/v1/llm/status"),
        EndpointSpec(.post, "/v1/llm/model"),
        EndpointSpec(.post, "/v1/llm/preload"),
        EndpointSpec(.post, "/v1/llm/unload"),

        EndpointSpec(.get, "/v1/infinite/models"),
        EndpointSpec(.post, "/v1/infinite/model"),
        EndpointSpec(.get, "/v1/infinite/status"),
        EndpointSpec(.get, "/v1/infinite/sessions"),
        EndpointSpec(.post, "/v1/infinite/sessions"),
        EndpointSpec(.get, "/v1/infinite/sessions/:sessionID"),
        EndpointSpec(.post, "/v1/infinite/sessions/:sessionID/append"),
        EndpointSpec(.post, "/v1/infinite/sessions/:sessionID/generate"),
        EndpointSpec(.post, "/v1/infinite/sessions/:sessionID/checkpoint"),
        EndpointSpec(.delete, "/v1/infinite/sessions/:sessionID"),

        EndpointSpec(.post, "/v1/touchup"),

        EndpointSpec(.post, "/v1/grammar/check"),

        EndpointSpec(.post, "/v1/vad/detect"),

        EndpointSpec(.get, "/v1/stt/engine"),
        EndpointSpec(.post, "/v1/stt/engine"),
        EndpointSpec(.get, "/v1/stt/languages"),
        EndpointSpec(.get, "/v1/stt/status"),
        EndpointSpec(.post, "/v1/stt/load"),
        EndpointSpec(.post, "/v1/stt/batch"),
    ]

    /// The full REST surface, converted + legacy.
    static let all: [EndpointSpec] = converted + legacy
}
