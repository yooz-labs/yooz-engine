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
                return
            }
            // Kick off the variant-aware module eager-load AFTER the
            // server is listening. Modules load concurrently in the
            // background; `/v1/health` and `/v1/modules` report
            // `loading` -> `ready` / `error` as each finishes. The
            // server is fully responsive while loads run.
            //
            // Skip on test runs (`XCTest` sets `YOOZ_DISABLE_EAGER_LOAD`)
            // so unit tests don't drag in MLX weight loads on every
            // boot of the in-process server.
            if EngineConfig.eagerLoadOnLaunch {
                await ModuleEagerLoader.shared.kickoff(
                    variant: EngineConfig.variant
                )
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
