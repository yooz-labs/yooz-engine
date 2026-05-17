import EngineCore
import SwiftUI

/// SwiftUI application entry for the standalone (non-helper) engine.
///
/// Constructed only when the full `YoozEngine` variant is running and
/// `EngineConfig.isHelper` is `false`. Embedded variants (`VARIANT_WHISPER`,
/// `VARIANT_LITE`) never call `YoozEngineApp.main()` — the compile-time
/// branch in `main.swift` makes them headless unconditionally, so
/// `MenuBarExtra` cannot register an `NSStatusItem` regardless of any
/// runtime signal (#128). Helper-mode processes bypass SwiftUI entirely
/// via `main.swift` — that's the only way to keep `MenuBarExtra` from
/// registering an `NSStatusItem` (see #42 for the original SwiftUI race,
/// #128 for the compile-time-vs-runtime architectural rationale). The
/// process entry point lives in `main.swift` so the branch happens *before*
/// the SwiftUI scene graph is built; that's also why this struct no
/// longer carries `@main`.
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

/// Compile-time variant marker.
///
/// `true` when the current build is an embedded helper variant
/// (`VARIANT_WHISPER` or `VARIANT_LITE`) whose `main.swift` pins
/// `isHelperMode = true` unconditionally and never invokes
/// `YoozEngineApp.main()`. `false` for the full standalone `YoozEngine`
/// build, where helper mode is opt-in via the runtime
/// `EngineConfig.isHelper` signal.
///
/// Exposed so tests can pin the compile-time invariant — when these
/// targets eventually get their own test bundles, `XCTAssertTrue(...)`
/// against this constant will catch any accidental removal of the
/// `VARIANT_WHISPER` / `VARIANT_LITE` compile condition from
/// `project.yml`. See #128 for the architectural rationale (LaunchServices
/// strips both env and argv from `openApplication`-spawned helpers, so the
/// helper invariant must live at compile time, not runtime).
enum HelperVariantInvariant {
    #if VARIANT_WHISPER || VARIANT_LITE
    static let isCompileTimeHelperOnly = true
    #else
    static let isCompileTimeHelperOnly = false
    #endif
}
