// HarnessMain.swift
// YoozEngineXPCHarness
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Dev-only harness (engine#227) — NOT shipped, NOT an XCTest target. Proves
// the packaged `YoozEngineXPC.xpc` service is reachable end-to-end through
// `XPCTransport` once embedded under this app's own `Contents/XPCServices/`.
// Round-trips `GET /v1/health` and a streaming STT open/send/receive/close
// cycle.
//
// Why not XCTest: XPC services are launchd-managed with no GUI test-runner
// involved, and this repo's headless build environment cannot attach an
// app-hosted XCTest runner. A plain executable sidesteps that entirely —
// build it, then run the binary directly:
//
//   xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngineXPCHarness \
//     -configuration Debug -skipMacroValidation -derivedDataPath build build
//   "build/Build/Products/Debug/YoozEngineXPCHarness.app/Contents/MacOS/YoozEngineXPCHarness"
//
// Exit code 0 means the harness completed its checks (health round-tripped,
// and the streaming call either succeeded or came back as a well-formed
// typed `YoozEngineError` — both prove the XPC plumbing works; only a
// connection-level failure or a hang indicates a packaging problem). Exit
// code 1 means `/v1/health` itself failed to round-trip — the packaging is
// broken.
//
// Forces the Apple STT backend before streaming: the default backend
// (Parakeet, an MLX model) needs a multi-hundred-MB HuggingFace download on
// first use, which this harness has no business triggering just to prove
// wiring. Apple STT needs no download; on a machine without Speech
// Recognition authorization already granted it fails fast with a typed
// error instead of downloading anything or hanging on a permission prompt
// (`AppleSTTEngine.start` only reads `SFSpeechRecognizer.authorizationStatus`,
// it never calls `requestAuthorization` itself) — see "Verifying the round
// trip" in docs/CONSUMER_INTEGRATION.md for what a passing run looks like
// with permission granted vs. not.

import Foundation
import YoozEngineClient

@main
struct HarnessMain {
    /// Must equal `YoozEngineXPC`'s `PRODUCT_BUNDLE_IDENTIFIER` — the XPC
    /// service is addressed by bundle id, not a port.
    static let serviceName = "live.yooz.engine.xpc"

    static func main() async {
        let transport = XPCTransport(serviceName: serviceName)
        let client = YoozEngineClient(transport: transport)

        do {
            let health = try await client.health()
            log("HEALTH_OK status=\(health.status) version=\(health.version)")
        } catch {
            log("HEALTH_FAIL \(error)")
            exit(1)
        }

        do {
            try await client.stt.setEngine(id: "apple_stt", preload: false)
            let stream = try await client.stt.startStream(language: .english, mode: .normal)
            try await stream.sendAudio([Float](repeating: 0, count: 1_600))
            let result = try await stream.receive()
            stream.close()
            log("STREAM_OK result=\(String(describing: result))")
        } catch let error as YoozEngineError {
            // A typed engine error still proves the round trip: the request
            // crossed the XPC wire, the service dispatched it, and a
            // structured failure came back through `XPCErrorBridge` rather
            // than a hang or a raw connection-level error.
            log("STREAM_TYPED_ERROR \(error)")
        } catch {
            // Anything that ISN'T a `YoozEngineError` (a raw connection
            // error `XPCErrorBridge` failed to bridge, a decoding crash,
            // etc.) is a genuine packaging/wiring problem, not proof the
            // round trip works — fail loudly instead of reporting success.
            log("STREAM_UNEXPECTED_ERROR \(error)")
            exit(1)
        }

        log("HARNESS_DONE")
        exit(0)
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
