import AppKit
#if canImport(AppleSTTModule)
import AppleSTTModule
#endif
import EngineCore
import GrammarModule
import SwiftUI
#if canImport(LLMModule)
import LLMModule
#endif
#if canImport(STTModule)
import STTModule
#endif
#if canImport(VADModule)
import VADModule
#endif

@MainActor
final class EngineAppDelegate: NSObject, NSApplicationDelegate {
    let server = APIServer()
    private var crashObserver: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Helper mode: remove any UI presence. `MenuBarExtra(isInserted:)`
        // with a constant-false binding leaves a ghost slot in some macOS
        // versions; `.prohibited` activation policy guarantees no menu-bar
        // icon, no dock tile, no ⌘-Tab entry.
        if EngineConfig.isHelper {
            NSApp.setActivationPolicy(.prohibited)
        }

        // Surface server crashes in the UI. The APIServer posts the
        // notification from its crash watcher; we log it here so ops
        // can see it in Console.app independent of the menu bar view.
        let serverRef = server
        crashObserver = NotificationCenter.default.addObserver(
            forName: APIServer.crashedNotification,
            object: server,
            queue: .main
        ) { note in
            let reason = (note.userInfo?[APIServer.crashErrorKey] as? String) ?? "unknown"
            // serverRef is captured by value (class reference). Its
            // logger is a swift-log Logger — safe to call from any
            // thread; swift-log synchronises internally.
            serverRef.logger.error("Engine server crashed: \(reason)")
        }

        Task {
            await registerModules()

            do {
                try await server.start()
            } catch {
                server.logger.error("Failed to start server: \(error)")
                // In helper mode the host app owns user-facing UI; a modal
                // alert from an embedded helper is very bad UX (and can hang
                // if the helper's NSApplication has no key window). Log-only
                // when headless; exit nonzero so the host app's launch
                // sequence can detect the failure via process termination.
                if EngineConfig.isHelper {
                    // Shut any partially-initialised Hummingbird resources
                    // down cleanly before exiting. `NSApplication.terminate(_:)`
                    // posts an async event and does not guarantee the exit
                    // code, which would make whisper's crash observer see a
                    // clean (zero-status) shutdown. `exit(EX_SOFTWARE)` is a
                    // synchronous non-zero exit whisper can distinguish from a
                    // normal termination.
                    await server.stop()
                    exit(EX_SOFTWARE)
                } else {
                    let alert = NSAlert()
                    alert.messageText = "Yooz Engine Failed to Start"
                    alert.informativeText = error.localizedDescription
                    alert.alertStyle = .critical
                    alert.runModal()
                }
            }
        }
    }

    // The app delegate lives for the app's lifetime; we do not need to
    // remove the crash observer in deinit. Keeping the token reference
    // alive until process exit is sufficient.

    /// Register every module compiled into this build variant.
    ///
    /// `#if canImport(...)` lets unlisted modules compile out silently for
    /// slim variants (e.g. Notes may exclude STT). Today all modules ship
    /// in every variant, but the scaffolding is in place for the split.
    private func registerModules() async {
        #if canImport(GrammarModule)
        await ModuleRegistry.shared.register(GrammarEngine.shared)
        #endif
        #if canImport(LLMModule)
        await ModuleRegistry.shared.register(TouchUpEngine.shared)
        #endif
        #if canImport(STTModule)
        await ModuleRegistry.shared.register(YoozSTTEngine.shared)
        #endif
        #if canImport(AppleSTTModule)
        await ModuleRegistry.shared.register(AppleSTTEngine.shared)
        #endif
        #if canImport(VADModule)
        await ModuleRegistry.shared.register(VADEngine.shared)
        #endif
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Task { @MainActor in
            await server.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
