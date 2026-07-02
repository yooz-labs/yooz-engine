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
/// fast on validation (a 400, or a JSON decode error — every in-process
/// handler validates before doing real work, see the handler contract above
/// `InProcessTransport.post`) rather than doing model-load or network work.
/// The only signal under test is whether the call falls through to
/// `YoozEngineError.unsupportedOperation` — the "this route doesn't exist
/// here" branch of `InProcessTransport.get/post/delete` (and, for the one
/// WebSocket entry, `openSTTStream`).
///
/// As defense in depth, every dispatch runs with the HuggingFace cache env
/// (`HF_HOME` / `HF_HUB_CACHE`) redirected to an empty temp directory, so
/// even a handler with no validation gate (`POST /v1/models/cleanup` and
/// `POST /v1/session/*` today) can never touch the machine's real model
/// cache from this suite. (The session handlers run their real fan-out —
/// `SessionCoordinator` reset of in-process module state — which is cheap
/// and safe in a bare test process.)
final class RouteParityTests: XCTestCase {
    private func makeTransport() async throws -> InProcessTransport {
        let transport = InProcessTransport()
        try await transport.connect()
        return transport
    }

    // MARK: - Manifest / allowlist hygiene

    /// A truncated or accidentally-emptied manifest would make the core
    /// parity loop pass vacuously; anchor it on known-stable routes.
    func testManifestIsNonEmptyAndAnchored() {
        let keys = Set(RouteManifest.all.map(\.key))
        XCTAssertTrue(keys.contains("GET /v1/health"), "manifest lost its /v1/health anchor")
        XCTAssertTrue(keys.contains("WS /v1/stt/stream"), "manifest lost its WS anchor")
    }

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

        try await withRedirectedHFCache {
            for entry in RouteManifest.all where !allowlisted.contains(entry.key) {
                let reachable = await self.isReachableInProcess(entry, transport: transport)
                XCTAssertTrue(
                    reachable,
                    """
                    \(entry.key) is not reachable in-process (falls through to \
                    unsupportedOperation) — add in-process handling in \
                    InProcessTransport, or add a RouteParityAllowlist entry with a reason.
                    """
                )
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
        try await withRedirectedHFCache {
            for allowed in RouteParityAllowlist.loopbackOnly {
                let reachable = await self.isReachableInProcess(allowed.entry, transport: transport)
                XCTAssertFalse(
                    reachable,
                    "\(allowed.entry.key) is reachable in-process now — remove its loopbackOnly entry"
                )
            }
        }
    }

    /// Self-test of the detector: a route that exists nowhere must be
    /// classified as unreachable for every REST verb. Automates the
    /// "temporarily inject a bogus manifest entry and watch the gate fail"
    /// manual verification, so a regression in the classification logic
    /// itself (e.g. the unsupportedOperation match breaking) cannot silently
    /// neuter the whole suite.
    func testDetectorClassifiesUnknownRouteAsUnreachable() async throws {
        let transport = try await makeTransport()
        try await withRedirectedHFCache {
            for method: RouteMethod in [.get, .post, .delete] {
                let bogus = RouteManifestEntry(method, "/v1/route-parity-nonexistent")
                let reachable = await self.isReachableInProcess(bogus, transport: transport)
                XCTAssertFalse(reachable, "\(bogus.key) should be classified unreachable")
            }
        }
    }

    // MARK: - Dispatch helpers

    /// Classify a manifest entry: `true` when dispatch reached a real handler
    /// branch, `false` when it fell through to `unsupportedOperation`.
    ///
    /// Reaching ANY other outcome — success, a `DecodingError` from the
    /// deliberately-empty body (each handler validates first), a validation
    /// 400/404, even an unexpected runtime error — proves the switch matched
    /// this route rather than the default case, which is the entire claim
    /// under test. `unsupportedOperation` has exactly one producer in the
    /// transport: the unrouted-path branches.
    private func isReachableInProcess(
        _ entry: RouteManifestEntry, transport: InProcessTransport
    ) async -> Bool {
        do {
            try await dispatch(entry, transport: transport)
            return true
        } catch let error as YoozEngineError {
            if case .unsupportedOperation = error { return false }
            return true
        } catch is DecodingError {
            return true
        } catch {
            // Anything else surfaced past validation still proves the route
            // is wired. Log it so an unexpected error is diagnosable when
            // someone needs to debug a parity run.
            NSLog(
                "RouteParityTests: %@ threw non-routing error: %@",
                entry.key, String(describing: error)
            )
            return true
        }
    }

    /// Route a manifest entry to the matching `InProcessTransport` call.
    /// Bodies are intentionally minimal/invalid (see the type doc comment).
    private func dispatch(_ entry: RouteManifestEntry, transport: InProcessTransport) async throws {
        switch entry.method {
        case .get:
            _ = try await transport.get(entry.concretePath)

        case .post:
            _ = try await transport.post(entry.concretePath, body: Data())

        case .delete:
            _ = try await transport.delete(entry.concretePath)

        case .websocket:
            // The package's minimum platforms (macOS 14 / iOS 17) satisfy
            // `openSTTStream`'s availability annotation, so no runtime guard
            // is needed — the WS entry is always genuinely exercised.
            // An unrecognized language code fails validation before any
            // backend/model work, regardless of the currently-active STT
            // backend (the guard runs before the backend switch) — cheap and
            // deterministic proof `openSTTStream` is wired.
            _ = try await transport.openSTTStream(language: "zz-route-parity-invalid", mode: "normal")
        }
    }

    /// Point the HuggingFace cache env at an empty temp directory for the
    /// duration of `body`, then restore it. Both variables are handled:
    /// `HF_HUB_CACHE` takes precedence over `HF_HOME` in
    /// `EngineConfig.huggingFaceCacheDirectory`, so it must be cleared or a
    /// developer's ambient `HF_HUB_CACHE` would silently defeat the redirect.
    private func withRedirectedHFCache(_ body: () async throws -> Void) async throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("route-parity-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let env = ProcessInfo.processInfo.environment
        let savedHome = env["HF_HOME"]
        let savedHubCache = env["HF_HUB_CACHE"]
        setenv("HF_HOME", dir.path, 1)
        unsetenv("HF_HUB_CACHE")
        defer {
            if let savedHome { setenv("HF_HOME", savedHome, 1) } else { unsetenv("HF_HOME") }
            if let savedHubCache {
                setenv("HF_HUB_CACHE", savedHubCache, 1)
            } else {
                unsetenv("HF_HUB_CACHE")
            }
            do {
                try fm.removeItem(at: dir)
            } catch {
                NSLog(
                    "RouteParityTests: temp HF cache cleanup failed at %@: %@",
                    dir.path, String(describing: error)
                )
            }
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
