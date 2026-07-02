// AppGroupWeightsLocation.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import OSLog

/// Resolves the shared weights cache location for the sandboxed XPC service
/// packaging (engine#227; `../yooz/docs/engine-app-packaging.md` "Standalone
/// packaging — sandboxed XPC service" -> "Weights"). A consumer app and its
/// `.xpc` service each declare `com.apple.security.application-groups` with
/// the SAME team-ID-prefixed group id, so both processes resolve the same
/// container and never duplicate a multi-GB HuggingFace (HF) download.
///
/// Group id convention: `<TeamID>.live.yooz.<app>.shared` (e.g. Whisper's
/// app + its XPC service both declare
/// `$(TeamIdentifierPrefix)live.yooz.whisper.shared`; substitute your app's
/// name for `<app>`). This repo's own `YoozEngineXPC` target + dev harness
/// use `live.yooz.engine.shared` as the illustrative id — there is no real
/// shipped app here, just the reference packaging. See
/// `docs/CONSUMER_INTEGRATION.md` "Weights/app-group wiring" for the full
/// recipe, including the Info.plist trick that resolves
/// `$(TeamIdentifierPrefix)` at build time instead of guessing the running
/// process's team id at runtime.
///
/// Scope: this redirects the HF hub cache only (`HF_HUB_CACHE`, the highest
/// -priority branch of `EngineConfig.huggingFaceCacheDirectory`'s resolution
/// order) — the large downloads (STT/LLM weights) all land there per
/// AGENTS.md "HF model auto-download". `EngineConfig.modelsDirectory` (a few
/// smaller LLM/STT artifacts) is a separate, non-overridable
/// `Application Support` path scoped to each sandboxed process's own
/// container; it is NOT yet app-group-aware — tracked as a known follow-up,
/// out of scope for the XPC packaging epic itself.
public enum AppGroupWeightsLocation {
    private static let logger = Logger(subsystem: "live.yooz.engine", category: "AppGroupWeightsLocation")

    /// Pure path construction: the HF hub-cache directory inside an
    /// already-resolved app-group container. Exposed separately from
    /// `redirectHuggingFaceCache` so this is unit-testable without a real
    /// sandboxed process / provisioning profile.
    ///
    /// Application Support, NOT Caches — the OS purges container Caches
    /// under disk pressure, which would force a multi-GB re-download (the
    /// exact failure mode the packaging doc calls out).
    public static func huggingFaceHubCacheURL(inContainer container: URL) -> URL {
        container
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)
            .appendingPathComponent("YoozEngine", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
    }

    /// Resolve `groupIdentifier`'s shared container and point `HF_HUB_CACHE`
    /// at its hub-cache subdirectory, creating it if needed. Call this once,
    /// early in process startup, BEFORE any module touches the HF
    /// downloader — `EngineConfig.huggingFaceCacheDirectory` is computed (not
    /// cached), but the underlying `HubApi`/downloader clients it feeds may
    /// resolve their own cache root at construction time.
    ///
    /// `fileManager` is injectable (defaults to `.default`) so a test double
    /// can exercise the "container resolves but the subdirectory can't be
    /// created" branch deterministically — `FileManager` is an open class,
    /// not a protocol, so the double is a thin subclass overriding just the
    /// two methods this function calls, delegating everything else to a real
    /// `FileManager` (same "backend double, not a mock" shape as
    /// `CannedStreamTransport` in `XPCStreamingTests.swift`).
    ///
    /// Returns `nil` (a no-op — never crashes, never throws) when the
    /// directory can't be resolved or created, logging WHY at debug level so
    /// a misconfigured group id doesn't fail silently and invisibly (a
    /// consumer app that gets the id wrong would otherwise see permanent,
    /// undiagnosable duplicate multi-GB downloads with no trace in the log).
    /// Distinguishes three causes in the log: empty identifier, unresolvable
    /// container, and an uncreatable subdirectory. Callers fall through to
    /// `EngineConfig.huggingFaceCacheDirectory`'s own sandboxed-container
    /// fallback in every case, so the engine still works — just without
    /// cross-process cache sharing.
    ///
    /// Note `containerURL(forSecurityApplicationGroupIdentifier:)` itself
    /// does NOT require the `application-groups` entitlement to resolve a
    /// path when the calling process is unsandboxed (a local dev run, or a
    /// consumer app before it adds the App Sandbox capability) — macOS only
    /// enforces group-entitlement matching for a genuinely sandboxed
    /// caller. An unsandboxed run therefore typically DOES succeed here,
    /// landing under `~/Library/Group Containers/<groupIdentifier>/`, which
    /// is a real, private-to-that-string location — harmless, just not
    /// actually shared with anything unless a sandboxed process with the
    /// matching entitlement also resolves it.
    @discardableResult
    public static func redirectHuggingFaceCache(
        groupIdentifier: String,
        fileManager: FileManager = .default
    ) -> URL? {
        guard !groupIdentifier.isEmpty else {
            logger.debug("redirectHuggingFaceCache: empty group identifier, skipping")
            return nil
        }
        guard let container = fileManager.containerURL(
            forSecurityApplicationGroupIdentifier: groupIdentifier
        ) else {
            logger.debug("redirectHuggingFaceCache: could not resolve container for group '\(groupIdentifier, privacy: .public)'")
            return nil
        }
        let hubCache = huggingFaceHubCacheURL(inContainer: container)
        do {
            try fileManager.createDirectory(at: hubCache, withIntermediateDirectories: true)
        } catch {
            logger.debug(
                "redirectHuggingFaceCache: could not create '\(hubCache.path, privacy: .public)': \(error.localizedDescription, privacy: .public)"
            )
            return nil
        }
        setenv("HF_HUB_CACHE", hubCache.path, 1)
        return hubCache
    }
}
