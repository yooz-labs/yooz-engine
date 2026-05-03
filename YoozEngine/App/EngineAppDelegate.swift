import AppKit
import SwiftUI

@MainActor
final class EngineAppDelegate: NSObject, NSApplicationDelegate {
    let server = APIServer()

    func applicationWillFinishLaunching(_ notification: Notification) {
        // Belt-and-suspenders: helper-mode processes already pin
        // `.prohibited` in `main.swift` *before* the run loop starts, and
        // they bypass SwiftUI entirely so no `MenuBarExtra` scene is ever
        // built. Re-asserting the policy here costs nothing and keeps the
        // contract explicit if a future entry point ever forgets to set
        // it (e.g. an XCTest host that instantiates the delegate directly).
        // See #42 for why setting the policy alone isn't sufficient: by
        // the time `MenuBarExtra` registers its `NSStatusItem`, the icon
        // is already in the menu bar regardless of activation policy.
        if EngineConfig.isHelper {
            NSApp.setActivationPolicy(.prohibited)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Task {
            do {
                // `server.start()` boots the HTTP server and (in
                // production) kicks off the variant-aware module
                // eager-load. The eager-load runs in the background;
                // this `await` returns once the server is listening
                // and the kickoff Task is spawned, not when the
                // modules finish loading.
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

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Task { @MainActor in
            await server.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
