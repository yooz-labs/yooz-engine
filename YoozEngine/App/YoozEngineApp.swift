import EngineCore
import SwiftUI

@main
struct YoozEngineApp: App {
    @NSApplicationDelegateAdaptor(EngineAppDelegate.self) var appDelegate

    var body: some Scene {
        // Helper mode: the host app (e.g. yooz-whisper) owns the menu bar
        // and Settings UI. `LSUIElement=true` hides the dock/menu icon;
        // `MenuBarExtra` is gated below so no menu-bar icon appears. The
        // `Settings` scene is inert when nothing opens it.
        MenuBarExtra("Yooz Engine", systemImage: "brain.filled.head.profile", isInserted: menuBarInserted) {
            EngineMenuView(server: appDelegate.server)
        }
        Settings {
            EngineSettingsView()
        }
    }

    private var menuBarInserted: Binding<Bool> {
        .constant(!EngineConfig.isHelper)
    }
}
