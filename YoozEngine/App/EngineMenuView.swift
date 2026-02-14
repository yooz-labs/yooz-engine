import SwiftUI

struct EngineMenuView: View {
    @ObservedObject var server: APIServer

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(server.isRunning ? .green : .red)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.caption)
            }

            if let error = server.lastError {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 8)

        if server.isRunning {
            let sttEngine = YoozSTTEngine.shared
            VStack(alignment: .leading, spacing: 2) {
                Text("Modules")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                HStack(spacing: 4) {
                    Circle()
                        .fill(sttEngine.isRunning ? .green : .gray)
                        .frame(width: 6, height: 6)
                    Text("STT")
                        .font(.caption)
                    if sttEngine.isRunning {
                        Text("(\(sttEngine.currentLanguage.displayName))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 8)
        }

        Divider()

        if server.isRunning {
            Button("Stop Server") {
                Task { await server.stop() }
            }
        } else if server.state == .stopped {
            Button("Start Server") {
                Task {
                    do {
                        try await server.start()
                    } catch {
                        server.logger.error("Failed to start: \(error)")
                    }
                }
            }
        } else {
            Text(server.state == .starting ? "Starting..." : "Stopping...")
                .font(.caption)
                .foregroundStyle(.secondary)
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

    private var statusText: String {
        switch server.state {
        case .running: return "Running on port \(EngineConfig.port)"
        case .starting: return "Starting..."
        case .stopping: return "Stopping..."
        case .stopped: return "Stopped"
        }
    }
}
