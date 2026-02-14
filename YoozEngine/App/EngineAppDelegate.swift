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
                print("[YoozEngine] Failed to start server: \(error)")
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        Task {
            await server.stop()
        }
    }
}
