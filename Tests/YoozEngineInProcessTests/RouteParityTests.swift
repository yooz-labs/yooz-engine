// RouteParityTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// Route-parity gate (#223): every route `APIServer` registers on the
/// loopback server is either reachable through `InProcessTransport` or
/// explicitly, reviewably declared loopback-only. `RouteManifest.all`
/// (EngineCore) is the tactical list of `(method, path)` entries mirroring
/// `APIServer`'s registrations; `RouteParityAllowlist.loopbackOnly` is the
/// reviewed set of intentional gaps.
///
/// This suite is a **dispatch-reachability** check, not a behavior test: each
/// request is deliberately minimal or invalid so a route that IS wired fails
/// fast on validation (a 400, or a JSON decode error — decoding is always the
/// first statement in each handler) rather than doing real model-load or
/// network work. The only signal under test is whether the call falls through
/// to `YoozEngineError.unsupportedOperation` — the "this route doesn't exist
/// here" branch of `InProcessTransport.get/post/delete` (and, for the one
/// WebSocket entry, `openSTTStream`).
final class RouteParityTests: XCTestCase {
    private func makeTransport() async throws -> InProcessTransport {
        let transport = InProcessTransport()
        try await transport.connect()
        return transport
    }

    // MARK: - Allowlist hygiene

    /// A `loopbackOnly` entry with no matching manifest entry (e.g. after a
    /// path rename) silently allowlists nothing and stops protecting anything.
    func testAllowlistEntriesExistInManifest() {
        let manifestKeys = Set(RouteManifest.all.map(\.key))
        for allowed in RouteParityAllowlist.loopbackOnly {
            XCTAssertTrue(
                manifestKeys.contains(allowed.entry.key),
                "loopbackOnly entry \(allowed.entry.key) has no matching RouteManifest.all entry"
            )
        }
    }

    /// Every allowlist entry must carry a real justification, not a stub.
    func testAllowlistEntriesHaveAReason() {
        for allowed in RouteParityAllowlist.loopbackOnly {
            XCTAssertFalse(
                allowed.reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "\(allowed.entry.key) has no justification"
            )
        }
    }

    /// The manifest itself must not carry accidental duplicate entries.
    func testManifestHasNoDuplicateEntries() {
        let keys = RouteManifest.all.map(\.key)
        XCTAssertEqual(keys.count, Set(keys).count, "RouteManifest.all has duplicate (method, path) entries")
    }

    // MARK: - Core parity gate

    /// Every non-allowlisted manifest entry must be reachable in-process.
    /// Adding a new `APIServer` route without in-process handling and without
    /// a `loopbackOnly` entry fails here (#223 acceptance criterion 1).
    func testEveryNonAllowlistedRouteIsReachableInProcess() async throws {
        let allowlisted = Set(RouteParityAllowlist.loopbackOnly.map { $0.entry.key })
        let transport = try await makeTransport()

        for entry in RouteManifest.all where !allowlisted.contains(entry.key) {
            do {
                try await dispatch(entry, transport: transport)
            } catch let error as YoozEngineError {
                if case .unsupportedOperation = error {
                    XCTFail(
                        """
                        \(entry.key) is not reachable in-process (falls through to \
                        unsupportedOperation) — add in-process handling in \
                        InProcessTransport, or add a RouteParityAllowlist entry with a reason.
                        """
                    )
                }
                // Any other YoozEngineError (400 validation, 404, ...) proves the
                // request reached a real handler branch.
            } catch is DecodingError {
                // The handler's first statement is `JSONDecoder().decode(...)`;
                // reaching a decode failure on a deliberately-empty body proves
                // the switch matched this route rather than the default case.
            } catch {
                // Any other error surfaced past validation still proves the
                // route is wired; only unsupportedOperation above is a failure.
            }
        }
    }

    /// Inverse check: every allowlisted entry is, today, genuinely unsupported
    /// in-process. If this starts failing, a route was wired up without
    /// shrinking the allowlist — remove the now-stale entry (see the
    /// coordination note in `RouteManifest.swift` for the #222 session-route
    /// entries specifically).
    func testEveryAllowlistedRouteStillThrowsUnsupported() async throws {
        let transport = try await makeTransport()
        for allowed in RouteParityAllowlist.loopbackOnly {
            do {
                try await dispatch(allowed.entry, transport: transport)
                XCTFail("\(allowed.entry.key) is reachable in-process now — remove its loopbackOnly entry")
            } catch let error as YoozEngineError {
                guard case .unsupportedOperation = error else {
                    XCTFail("\(allowed.entry.key) threw \(error), expected unsupportedOperation")
                    continue
                }
            }
        }
    }

    // MARK: - Dispatch helpers

    /// Route a manifest entry to the matching `InProcessTransport` call.
    /// Bodies are intentionally minimal/invalid (see the type doc comment);
    /// `/v1/models/cleanup` is the one route with no decode gate ahead of real
    /// filesystem work, so it gets a redirected `HF_HOME` to keep this test
    /// from touching the machine's real model cache.
    private func dispatch(_ entry: RouteManifestEntry, transport: InProcessTransport) async throws {
        switch entry.method {
        case .get:
            _ = try await transport.get(entry.concretePath)

        case .post where entry.path == "/v1/models/cleanup":
            try await withRedirectedHFHome {
                _ = try await transport.post(entry.concretePath, body: Data())
            }

        case .post:
            _ = try await transport.post(entry.concretePath, body: Data())

        case .delete:
            _ = try await transport.delete(entry.concretePath)

        case .websocket:
            guard #available(macOS 14.0, iOS 17.0, *) else { return }
            // An unrecognized language code fails validation before any
            // backend/model work, regardless of the currently-active STT
            // backend (the guard runs before the backend switch) — cheap and
            // deterministic proof `openSTTStream` is wired.
            _ = try await transport.openSTTStream(language: "zz-route-parity-invalid", mode: "normal")
        }
    }

    /// Point `HF_HOME` at an empty temp directory for the duration of `body`,
    /// then restore it. Mirrors `InProcessTransportTests.testInProcessCleanupCollapsesRedirectedCache`.
    private func withRedirectedHFHome(_ body: () async throws -> Void) async throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("route-parity-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let saved = ProcessInfo.processInfo.environment["HF_HOME"]
        setenv("HF_HOME", dir.path, 1)
        defer {
            if let saved { setenv("HF_HOME", saved, 1) } else { unsetenv("HF_HOME") }
            try? fm.removeItem(at: dir)
        }
        try await body()
    }
}

private extension RouteManifestEntry {
    /// Substitute Hummingbird `:param` path segments with a fixed placeholder
    /// for dispatch. The placeholder deliberately does not match any real id,
    /// so routes that resolve it (e.g. `DELETE /v1/models/:id`) take their
    /// "unknown id" branch (404) rather than mutating real state.
    var concretePath: String {
        path.split(separator: "/", omittingEmptySubsequences: false)
            .map { $0.hasPrefix(":") ? "route-parity-placeholder" : String($0) }
            .joined(separator: "/")
    }
}
