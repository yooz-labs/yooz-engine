// EndpointTable.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// The typed endpoint table (engine#225 Phase B).
//
// Each converted route is declared exactly once as an `Endpoint` — a spec
// (method + path) bound to a transport-agnostic `WireHandler` closure that
// works directly against the module actors. The loopback `APIServer`
// registers converted routes FROM the table (a generic Hummingbird adapter
// per entry) and `InProcessTransport` dispatches through
// `EndpointTable.match` before falling back to its legacy switch — so both
// transports derive from the single declaration instead of hand-mirroring
// each other, which is what allowed the `/v1/session` drift (#222) that
// motivated this table.
//
// Layering: the handler CLOSURES for module-backed families live in the
// module that owns the actor (e.g. `TouchUpEndpoints` /
// `ModelManagementEndpoints` in LLMModule) — both transports compile those
// same sources, so binding the table in each transport references one shared
// implementation. Handlers whose dependencies live entirely in `EngineCore`
// (the session family) are provided here. The spec catalog
// (`EndpointSpecs`) is total over the REST surface so `RouteManifest` can be
// a projection of it, even for routes whose handlers are not yet converted.

/// One `(method, path)` route identity. The single declaration of a route's
/// wire location — `RouteManifest` projects from the catalog of these, and
/// an `Endpoint` binds one to its handler.
public struct EndpointSpec: Sendable, Hashable {
    public let method: RouteMethod
    /// Path pattern with Hummingbird-style `:name` parameter placeholders
    /// (e.g. `/v1/models/:id`).
    public let path: String

    public init(_ method: RouteMethod, _ path: String) {
        self.method = method
        self.path = path
    }

    /// Stable identity string, same shape as `RouteManifestEntry.key`.
    public var key: String { "\(method.rawValue) \(path)" }

    /// The `:name` parameter names appearing in `path`, in order.
    public var parameterNames: [String] {
        path.split(separator: "/")
            .filter { $0.hasPrefix(":") }
            .map { String($0.dropFirst()) }
    }
}

/// Wire-level request as seen by a table handler: the raw body plus routing
/// context, already transport-neutral. The loopback adapter fills it from
/// the Hummingbird request; the in-process adapter from the SDK call.
public struct WireRequest: Sendable {
    public let body: Data
    /// Captured `:name` path parameters, percent-decoded.
    public let pathParameters: [String: String]
    /// Raw query string (no leading `?`), or nil when absent. Carried for
    /// handlers that honor flags like `?wait=true`; converted families today
    /// don't read it, but the shape supports them without a signature break.
    public let query: String?

    public init(body: Data = Data(), pathParameters: [String: String] = [:], query: String? = nil) {
        self.body = body
        self.pathParameters = pathParameters
        self.query = query
    }
}

/// Wire-level response from a table handler: an HTTP status plus a JSON body
/// (empty for 204). The loopback adapter maps it onto a Hummingbird
/// `Response`; the in-process adapter returns the body `Data` (its transport
/// has no status channel — same as before the table).
public struct WireResponse: Sendable {
    public let status: Int
    public let body: Data

    public init(status: Int = 200, body: Data = Data()) {
        self.status = status
        self.body = body
    }

    /// Encode `value` with the same default `JSONEncoder` both transports
    /// have always used for these routes.
    public static func json<T: Encodable>(_ value: T, status: Int = 200) throws -> WireResponse {
        WireResponse(status: status, body: try JSONEncoder().encode(value))
    }

    public static let noContent = WireResponse(status: 204)
}

/// Typed wire error thrown by table handlers. The loopback adapter renders
/// it as the standard `{"error": message, "code": code}` body with `status`;
/// the in-process adapter rethrows it as
/// `YoozEngineError.serverError(statusCode:code:message:)` so the SDK error
/// surface is unchanged. One error vocabulary, two transport renderings.
public struct WireError: Error, Sendable {
    public let status: Int
    public let code: String
    public let message: String

    public init(status: Int, code: String, message: String) {
        self.status = status
        self.code = code
        self.message = message
    }

    /// The canonical body-failed-to-decode error every route maps
    /// identically (AGENTS.md "Wire codes").
    public static func invalidRequest(_ underlying: Error) -> WireError {
        WireError(
            status: 400,
            code: "invalid_request",
            message: "Invalid request body: \(underlying.localizedDescription)"
        )
    }
}

/// A table handler: decode the request, call the module actors, encode the
/// response. Throws `WireError` for typed wire failures; anything else is a
/// programming error surfaced by the transport's generic error path.
public typealias WireHandler = @Sendable (WireRequest) async throws -> WireResponse

/// One converted route: its single spec + its single handler.
public struct Endpoint: Sendable {
    public let spec: EndpointSpec
    public let handler: WireHandler

    public init(_ spec: EndpointSpec, handler: @escaping WireHandler) {
        self.spec = spec
        self.handler = handler
    }
}

/// The dispatch table: validated collection of `Endpoint`s with
/// `(method, path)` matching, including `:name` parameter capture.
public struct EndpointTable: Sendable {
    /// Exact-path endpoints keyed by `spec.key`.
    private let exact: [String: Endpoint]
    /// Endpoints whose path contains `:name` segments, matched per segment.
    private let parameterized: [Endpoint]

    public let endpoints: [Endpoint]

    /// - Precondition: no two endpoints share a `(method, path)` spec. A
    ///   duplicate is a programming error (two declarations of one route —
    ///   exactly what the table exists to prevent), so it traps at
    ///   construction rather than silently shadowing.
    public init(_ endpoints: [Endpoint]) {
        var exact: [String: Endpoint] = [:]
        var parameterized: [Endpoint] = []
        var seen = Set<String>()
        for endpoint in endpoints {
            precondition(
                seen.insert(endpoint.spec.key).inserted,
                "duplicate endpoint declaration: \(endpoint.spec.key)"
            )
            if endpoint.spec.path.contains(":") {
                parameterized.append(endpoint)
            } else {
                exact[endpoint.spec.key] = endpoint
            }
        }
        self.exact = exact
        self.parameterized = parameterized
        self.endpoints = endpoints
    }

    public var specs: [EndpointSpec] { endpoints.map(\.spec) }

    /// Match a concrete request path (query already stripped) against the
    /// table. Returns the endpoint plus captured, percent-decoded path
    /// parameters.
    public func match(
        method: RouteMethod, path: String
    ) -> (endpoint: Endpoint, pathParameters: [String: String])? {
        if let endpoint = exact["\(method.rawValue) \(path)"] {
            return (endpoint, [:])
        }
        let pathSegments = path.split(separator: "/", omittingEmptySubsequences: true)
        for endpoint in parameterized where endpoint.spec.method == method {
            let patternSegments = endpoint.spec.path.split(
                separator: "/", omittingEmptySubsequences: true
            )
            guard patternSegments.count == pathSegments.count else { continue }
            var captured: [String: String] = [:]
            var matched = true
            for (pattern, segment) in zip(patternSegments, pathSegments) {
                if pattern.hasPrefix(":") {
                    let raw = String(segment)
                    captured[String(pattern.dropFirst())] = raw.removingPercentEncoding ?? raw
                } else if pattern != segment {
                    matched = false
                    break
                }
            }
            if matched {
                return (endpoint, captured)
            }
        }
        return nil
    }
}
