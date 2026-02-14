import SwiftUI

struct EngineMenuView: View {
    @ObservedObject var server: APIServer

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(server.isRunning ? .green : .red)
                    .frame(width: 8, height: 8)
                Text(server.isRunning ? "Running on port \(EngineConfig.port)" : "Stopped")
                    .font(.caption)
            }
        }
        .padding(.horizontal, 8)

        Divider()

        if server.isRunning {
            Button("Stop Server") {
                Task { await server.stop() }
            }
        } else {
            Button("Start Server") {
                Task { try? await server.start() }
            }
        }

        Divider()

        Button("Settings...") {
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        }
        .keyboardShortcut(",", modifiers: .command)

        Button("Quit Yooz Engine") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q", modifiers: .command)
    }
}
