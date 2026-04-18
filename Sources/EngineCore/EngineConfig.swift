// EngineConfig.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Process-wide engine configuration.
///
/// Lives in `EngineCore` so every module target (STT, LLM, VAD, Grammar)
/// can read ports, version, and on-disk locations without pulling in the
/// `YoozEngine` app target.
public enum EngineConfig {
    public static let port: Int = 19920
    public static let host: String = "127.0.0.1"
    public static let version: String = "0.6.0"

    /// Environment variable host apps set when launching the engine as an
    /// embedded helper. When `YOOZ_ENGINE_HEADLESS=1` the app suppresses its
    /// menu-bar UI and Settings scene, but still starts the API server and
    /// writes to the same OSLog subsystem. This contract is consumed by
    /// `YoozEngineApp` and `EngineAppDelegate`; host apps (e.g. yooz-whisper's
    /// `EngineHelperController`) must pass the variable through
    /// `NSWorkspace.OpenConfiguration.environment` when spawning the helper.
    public static let headlessEnvVar = "YOOZ_ENGINE_HEADLESS"

    /// `true` when the engine process was launched with
    /// `YOOZ_ENGINE_HEADLESS=1`. Used to gate menu-bar/Settings UI, modal
    /// alerts, and any other host-app-inappropriate behaviour.
    public static var isHelper: Bool {
        ProcessInfo.processInfo.environment[headlessEnvVar] == "1"
    }

    /// `~/Library/Application Support/YoozEngine/Models` — long-lived model
    /// artifacts keyed by `LLMModelType.rawValue` or STT model identifier.
    public static let modelsDirectory: URL = {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Application Support directory not found")
        }
        return appSupport.appendingPathComponent("YoozEngine/Models")
    }()

    /// `~/Library/Caches/live.yooz.engine` — ephemeral downloads, tarballs,
    /// partial models. Safe for the OS to evict.
    public static let cacheDirectory: URL = {
        guard let caches = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Caches directory not found")
        }
        return caches.appendingPathComponent("live.yooz.engine")
    }()
}
