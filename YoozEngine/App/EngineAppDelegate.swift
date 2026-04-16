import AppKit
import EngineCore
import GrammarModule
import SwiftUI
#if canImport(VADModule)
import VADModule
#endif

@MainActor
final class EngineAppDelegate: NSObject, NSApplicationDelegate {
    let server = APIServer()

    func applicationDidFinishLaunching(_ notification: Notification) {
        Task {
            await registerModules()

            do {
                try await server.start()
            } catch {
                server.logger.error("Failed to start server: \(error)")
                let alert = NSAlert()
                alert.messageText = "Yooz Engine Failed to Start"
                alert.informativeText = error.localizedDescription
                alert.alertStyle = .critical
                alert.runModal()
            }
        }
    }

    /// Register every module compiled into this build variant.
    ///
    /// `#if canImport(...)` lets unlisted modules compile out silently for
    /// slim variants (e.g. Notes may exclude STT). Today all modules ship
    /// in every variant, but the scaffolding is in place for the split.
    private func registerModules() async {
        #if canImport(GrammarModule)
        await ModuleRegistry.shared.register(GrammarEngine.shared)
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
