// main.swift
// YoozEngine
//
// Process entry point. Branches between two app shapes before SwiftUI is
// loaded:
//
// * **Standalone (menu-bar) mode** — when neither the helper env var nor
//   the helper argv flag is present, delegate to `YoozEngineApp.main()`.
//   SwiftUI builds the `MenuBarExtra` scene, the user sees the brain
//   icon, and `EngineSettingsView` is reachable via the standard
//   Settings menu.
//
// * **Helper mode** — when host apps (e.g. yooz-whisper) launch us with
//   `YOOZ_ENGINE_HEADLESS=1` OR pass `--headless` on the command line,
//   do **not** construct any SwiftUI scene. `MenuBarExtra` registers its
//   `NSStatusItem` as soon as the SwiftUI scene graph is evaluated,
//   which happens before any `NSApplicationDelegate` hook fires;
//   flipping `setActivationPolicy(.prohibited)` afterwards leaves a
//   ghost icon (issue #42). Bypassing SwiftUI entirely is the only
//   reliable way to guarantee no menu-bar presence.
//
//   Both signals are accepted because `NSWorkspace.OpenConfiguration.environment`
//   is NOT reliably propagated to nested helper bundles on macOS 26
//   (#117 / whisper#179), so host apps spawn us via
//   `OpenConfiguration.arguments = ["--headless"]` instead. The env-var
//   channel is preserved for shell exec, scripts, and test harnesses.
//   `EngineConfig.isHelper` consults both sources.
//
// In helper mode we drive `NSApplication` directly: install the same
// `EngineAppDelegate` (so the API server start/stop logic stays shared),
// pin the activation policy to `.prohibited`, and run the AppKit event
// loop. The delegate's `applicationDidFinishLaunching` boots the
// Hummingbird server exactly as it does in standalone mode.

import AppKit
import EngineCore

if EngineConfig.isHelper {
    // The whole helper bootstrap is `@MainActor`-isolated:
    // `EngineAppDelegate` is `@MainActor`, and `NSApplication` APIs need
    // main-thread affinity. Top-level Swift code is not automatically
    // `@MainActor`, so we hop explicitly. `assumeIsolated` is sound here
    // because process entry runs on the main thread.
    MainActor.assumeIsolated {
        let app = NSApplication.shared
        // Pin the activation policy to `.prohibited` *before* the run
        // loop starts. Combined with skipping SwiftUI, this guarantees
        // no menu bar / dock / cmd-tab presence even if a future code
        // path were to create an `NSStatusItem` imperatively.
        app.setActivationPolicy(.prohibited)
        // `NSApplication.delegate` is `weak`. We retain `delegate` here
        // for the run-loop's lifetime; `app.run()` does not return.
        let delegate = EngineAppDelegate()
        app.delegate = delegate
        app.run()
        // `NSApplication.run()` only returns after `NSApp.terminate(_:)`
        // (which calls `exit(0)` itself). Keeping `delegate` alive past
        // `run()` is moot in practice, but `withExtendedLifetime` makes
        // the intent explicit and silences the "never read" lint.
        withExtendedLifetime(delegate) { }
    }
} else {
    YoozEngineApp.main()
}
