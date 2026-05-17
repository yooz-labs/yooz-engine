// main.swift
// YoozEngine
//
// Process entry point. Branches between two app shapes before SwiftUI is
// loaded.
//
// ## Two variant classes
//
// * **Embedded helper variants (`VARIANT_WHISPER`, `VARIANT_LITE`)** — built
//   exclusively to be dropped into a host app's `Contents/Helpers/` and
//   launched via `NSWorkspace.openApplication`. These targets have NO
//   user-mode purpose; there is no scenario in which the menu-bar UI or
//   Settings scene should ever surface. The helper invariant is therefore
//   pinned at **compile time** (#128): `isHelperMode = true` unconditionally,
//   the `YoozEngineApp` SwiftUI scene is never invoked, and the `MenuBarExtra`
//   never has the chance to register an `NSStatusItem`.
//
//   Why compile-time and not runtime: on macOS 26, LaunchServices strips
//   both `OpenConfiguration.environment` AND `OpenConfiguration.arguments`
//   from helper processes launched via `openApplication`. Verified empirically
//   with `ps -E` and `lsappinfo info -only LSLaunchArguments`. There is no
//   recoverable runtime channel by which the host can signal "be headless"
//   through that API. Compile-time invariant is the correct architectural
//   fix because it matches the actual truth: an embedded variant cannot
//   meaningfully run in any other mode.
//
// * **Standalone variant (full `YoozEngine`)** — dev/diagnostic builds where
//   double-clicking the `.app` should still produce a menu-bar icon and
//   Settings scene. Helper mode is opt-in via the runtime signals from
//   #117/#119:
//
//   - `YOOZ_ENGINE_HEADLESS=1` env var (works for direct shell exec,
//     scripts, test harnesses, and any spawn API that propagates env).
//   - `--headless` argv flag (works for any spawn that propagates argv).
//
//   Both runtime channels remain in `EngineConfig.isHelper` as forward-compat
//   for any future host that wants to launch the full standalone build in
//   helper mode (e.g. an integration harness, a developer running the engine
//   manually via `open --args`, etc.).
//
// ## Helper bootstrap
//
// In helper mode we drive `NSApplication` directly: install the same
// `EngineAppDelegate` (so the API server start/stop logic stays shared),
// pin the activation policy to `.prohibited`, and run the AppKit event
// loop. The delegate's `applicationDidFinishLaunching` boots the
// Hummingbird server exactly as it does in standalone mode.

import AppKit
import EngineCore

#if VARIANT_WHISPER || VARIANT_LITE
// Compile-time invariant for embedded helper variants. The `YoozEngineApp`
// SwiftUI scene is never reachable from this build — `MenuBarExtra` cannot
// register an `NSStatusItem` because the scene graph is never evaluated.
// See file header for the LaunchServices-strips-env-and-argv rationale (#128).
let isHelperMode = true
#else
// Standalone variant: respect the runtime helper signal so developers and
// future hosts can still launch us headless when they need to.
let isHelperMode = EngineConfig.isHelper
#endif

if isHelperMode {
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
