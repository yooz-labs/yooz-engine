import AppKit
import SwiftUI

@MainActor
final class EngineAppDelegate: NSObject, NSApplicationDelegate {
    let server = APIServer()

    func applicationDidFinishLaunching(_ notification: Notification) {
        Task {
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

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Task { @MainActor in
            await server.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
