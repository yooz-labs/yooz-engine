// ModuleRegistry.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Process-wide registry of `AIModule` instances loaded into this build variant.
///
/// Modules register themselves at app launch (see `EngineAppDelegate`). The
/// server consults the registry to decide whether a route is backed by a
/// bundled module or should return HTTP 501.
///
/// The registry is intentionally simple: register by `name` at startup, look
/// up by `name` at request time. No runtime unloading, no lazy load — modules
/// that need explicit model loading handle that internally.
public actor ModuleRegistry {
    public static let shared = ModuleRegistry()

    private var modules: [String: any AIModule] = [:]

    private init() {}

    /// Register a module instance. Safe to call multiple times; later
    /// registrations replace earlier ones for the same `name`.
    public func register(_ module: any AIModule) {
        let name = type(of: module).name
        modules[name] = module
    }

    /// Remove a previously-registered module by name. No-op if absent.
    ///
    /// Registration normally happens once at app launch and persists for the
    /// process lifetime; this exists for tests that exercise the not-bundled
    /// path (HTTP 501) by starting a server without a given module present.
    /// Scoped to a single name on purpose — there is no drop-all, so a stray
    /// call can only affect the one module it names.
    public func unregister(_ name: String) {
        modules.removeValue(forKey: name)
    }

    /// Whether a module with the given name is bundled in this build variant.
    public func isBundled(_ name: String) -> Bool {
        modules[name] != nil
    }

    /// Look up a registered module by name. Returns nil if not bundled.
    public func module(_ name: String) -> (any AIModule)? {
        modules[name]
    }

    /// All registered modules, sorted by name for stable ordering in
    /// `/v1/modules` responses.
    public func all() -> [any AIModule] {
        modules.values.sorted { type(of: $0).name < type(of: $1).name }
    }

    /// All registered modules that participate in the per-recording-session
    /// reset boundary (engine issue #114). Sorted by module name for stable
    /// fan-out order in `/v1/session/begin` and `/v1/session/end`.
    ///
    /// A module opts in by conforming to `SessionResettable`; the registry
    /// is the single source of truth for which modules are currently active,
    /// so the session-reset fan-out automatically picks up new modules with
    /// zero wiring per model.
    public func allResettable() -> [any SessionResettable] {
        modules.values
            .sorted { type(of: $0).name < type(of: $1).name }
            .compactMap { $0 as? any SessionResettable }
    }
}
