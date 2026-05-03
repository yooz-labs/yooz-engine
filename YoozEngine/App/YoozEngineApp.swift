import SwiftUI

/// SwiftUI application entry for the standalone (non-helper) engine.
///
/// Constructed only when `EngineConfig.isHelper` is `false`. Helper-mode
/// processes bypass SwiftUI entirely via `main.swift`, which never calls
/// `YoozEngineApp.main()` — that's the only way to keep `MenuBarExtra`
/// from registering an `NSStatusItem` (see #42). The process entry point
/// lives in `main.swift` so the branch happens *before* the SwiftUI scene
/// graph is built; that's also why this struct no longer carries `@main`.
struct YoozEngineApp: App {
    @NSApplicationDelegateAdaptor(EngineAppDelegate.self) var appDelegate

    var body: some Scene {
        MenuBarExtra("Yooz Engine", systemImage: "brain.filled.head.profile") {
            EngineMenuView(server: appDelegate.server)
        }
        Settings {
            EngineSettingsView()
        }
    }
}
