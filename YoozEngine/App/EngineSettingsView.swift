import SwiftUI

struct EngineSettingsView: View {
    var body: some View {
        Form {
            Section("Server") {
                LabeledContent("Port", value: "\(EngineConfig.port)")
                LabeledContent("Version", value: EngineConfig.version)
            }

            Section("Models") {
                Text("No models loaded yet")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 400, height: 300)
    }
}
