import SwiftUI

@main
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
