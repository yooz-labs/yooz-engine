// ModelSelectionStore.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os

private let logger = Logger(subsystem: "live.yooz.engine", category: "model-selection-store")

/// Engine-owned persistence for each module's active model selection
/// (engine#226). The engine, not the consuming app, is the source of truth
/// for model-selection INTENT: a module's active id survives an engine
/// restart with no consumer involvement, closing the class of bugs the
/// engine#226 issue catalogs (whisper's `LLMModelPickerStore` reconciliation
/// latches, the STT rollback ladder, ...) — those exist only because the
/// engine used to forget the selection on every restart.
///
/// Backing store: a small JSON file (`[module: activeId]`) under
/// `EngineConfig.stateDirectory`. Migration is implicit: a module with no
/// stored entry (first run, or a module added after this file existed)
/// simply finds nothing and the caller falls back to its compiled-in
/// default — see `TouchUpEngine.restorePersistedSelectionIfNeeded()`.
///
/// An actor (not a plain struct + lock) so concurrent `setActiveId` calls
/// from different modules serialize without a separate lock, and so the
/// lazily-loaded in-memory cache can't be read torn mid-load.
public actor ModelSelectionStore {
    /// Production instance, backed by `EngineConfig.modelSelectionFileURL`
    /// (respecting `YOOZ_ENGINE_STATE_DIR`). Every module actor's `.shared`
    /// singleton should route through this instance so persistence is
    /// actually shared process-wide; module actors constructed fresh for
    /// tests should inject their own instance (see the `fileURL:` init)
    /// rather than default to this one, so a test process never touches a
    /// developer's real Application Support directory.
    public static let shared = ModelSelectionStore()

    private let fileURL: URL
    /// nil until the first read/write touches disk (lazy — actor `init`
    /// cannot `await` a file read).
    private var cache: [String: String]?

    public init(fileURL: URL = EngineConfig.modelSelectionFileURL) {
        self.fileURL = fileURL
    }

    /// The persisted active model id for `module`, or nil if nothing has
    /// been persisted for it yet.
    public func activeId(for module: String) -> String? {
        loadIfNeeded()
        return cache?[module]
    }

    /// Persist `id` as the active model for `module`. Best-effort: a disk
    /// failure (full disk, sandbox denial) is logged, not thrown — an
    /// engine-side persistence hiccup must never block the in-memory
    /// selection from taking effect for the rest of the process lifetime.
    public func setActiveId(_ id: String, for module: String) {
        loadIfNeeded()
        cache?[module] = id
        persist()
    }

    private func loadIfNeeded() {
        guard cache == nil else { return }
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([String: String].self, from: data)
        else {
            cache = [:]
            return
        }
        cache = decoded
    }

    private func persist() {
        guard let cache else { return }
        do {
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONEncoder().encode(cache)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            let path = self.fileURL.path
            logger.error(
                "ModelSelectionStore: persist to \(path, privacy: .public) failed: \(error.localizedDescription, privacy: .public)"
            )
        }
    }
}
