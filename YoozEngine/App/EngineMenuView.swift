import EngineCore
#if canImport(AppleSTTModule)
import AppleSTTModule
#endif
#if canImport(STTModule)
import STTModule
#endif
import SwiftUI

struct EngineMenuView: View {
    @ObservedObject var server: APIServer

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(statusColor)
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
            VStack(alignment: .leading, spacing: 2) {
                Text("Modules")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                HStack(spacing: 4) {
                    #if canImport(STTModule)
                    let sttEngine = YoozSTTEngine.shared
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
                    #elseif canImport(AppleSTTModule)
                    AppleSTTStatusRow()
                    #else
                    Circle()
                        .fill(.gray)
                        .frame(width: 6, height: 6)
                    Text("STT")
                        .font(.caption)
                    #endif
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
        } else if server.state == .crashed {
            Button("Restart Engine") {
                Task {
                    do {
                        try await server.restart()
                    } catch {
                        server.logger.error("Failed to restart: \(error)")
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

    private var statusColor: Color {
        switch server.state {
        case .running: return .green
        case .crashed: return .orange
        case .starting, .stopping: return .yellow
        case .stopped: return .red
        }
    }

    private var statusText: String {
        switch server.state {
        case .running: return "Running on port \(EngineConfig.port)"
        case .starting: return "Starting..."
        case .stopping: return "Stopping..."
        case .stopped: return "Stopped"
        case .crashed: return "Crashed — restart required"
        }
    }
}

#if canImport(AppleSTTModule)
private struct AppleSTTStatusRow: View {
    @State private var isLoaded = false
    @State private var language = AppleSTTLanguage.english

    var body: some View {
        Group {
            Circle()
                .fill(isLoaded ? .green : .gray)
                .frame(width: 6, height: 6)
            Text("Apple STT")
                .font(.caption)
            if isLoaded {
                Text("(\(language.rawValue.uppercased()))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .task { await refresh() }
    }

    private func refresh() async {
        let engine = AppleSTTEngine.shared
        isLoaded = await engine.isLoaded
        language = await engine.currentLanguage
    }
}
#endif
