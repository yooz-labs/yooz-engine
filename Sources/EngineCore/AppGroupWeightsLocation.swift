// AppGroupWeightsLocation.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

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
    /// Returns `false` (a no-op — never crashes, never throws) when the
    /// group directory can't be resolved or created — an empty identifier,
    /// or (rare in practice) a filesystem failure creating the
    /// subdirectory. Callers fall through to
    /// `EngineConfig.huggingFaceCacheDirectory`'s own sandboxed-container
    /// fallback in that case, so the engine still works — just without
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
    public static func redirectHuggingFaceCache(groupIdentifier: String) -> Bool {
        guard !groupIdentifier.isEmpty, let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: groupIdentifier
        ) else {
            return false
        }
        let hubCache = huggingFaceHubCacheURL(inContainer: container)
        guard (try? FileManager.default.createDirectory(
            at: hubCache, withIntermediateDirectories: true
        )) != nil else {
            return false
        }
        setenv("HF_HUB_CACHE", hubCache.path, 1)
        return true
    }
}
