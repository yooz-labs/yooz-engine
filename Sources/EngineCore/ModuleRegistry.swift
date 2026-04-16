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
}
