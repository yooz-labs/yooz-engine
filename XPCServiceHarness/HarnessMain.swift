// HarnessMain.swift
// YoozEngineXPCHarness
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Dev-only harness (engine#227) — NOT shipped, NOT an XCTest target. Proves
// the packaged `YoozEngineXPC.xpc` service is reachable end-to-end through
// `XPCTransport` once embedded under this app's own `Contents/XPCServices/`.
// Round-trips `GET /v1/health`, a streaming STT open/send/receive/close
// cycle, and (engine#244) an `/v1/events` open/publish/receive cycle.
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
// Exit code 0 means the harness completed its checks: health round-tripped,
// the streaming call either succeeded or came back as a well-formed typed
// `YoozEngineError` (both prove the XPC plumbing; Apple STT legitimately
// fails typed on a machine without Speech Recognition authorization), and
// the `/v1/events` round trip delivered a real `modelChanged` frame. Exit
// code 1 means a genuine packaging problem: `/v1/health` failed, the
// streaming call failed with a NON-engine error, or the events check failed
// in ANY way (timeout, typed error, or unexpected error — unlike STT, the
// events path has no permission/hardware excuse, so every failure there is
// a regression).
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

        // `/v1/events` (engine#244): open the push channel, then trigger a
        // REAL engine-side publish (`POST /v1/touchup/model`, preload:false
        // so no download/MLX load is involved — same reasoning as forcing
        // Apple STT above, keep this harness fast and hermetic) and confirm
        // the resulting `modelChanged` frame arrives back over the XPC
        // callback proxy. Unlike the STT check, NOTHING on this path has a
        // legitimate environment-dependent failure mode (no permission
        // prompt, no hardware dependency, no download), so EVERY failure —
        // timeout, typed engine error, or unexpected error — is a packaging
        // regression and a hard `exit(1)`. (That's also why this block
        // bounds its wait with a timeout while the STT block awaits
        // unbounded: STT's silence can mean "authorization dialog territory";
        // events' silence can only mean broken plumbing.)
        do {
            let stream = try await client.openEvents()

            try await client.touchUp.setModel(id: "yooz-quality-v2", preload: false)

            let matched = await Self.firstMatchingEvent(stream, timeoutSeconds: 5) {
                $0.kind == .modelChanged && $0.module == "touchup" && $0.modelId == "yooz-quality-v2"
            }
            // Restore the default selection so this harness run doesn't
            // leave the persisted `ModelSelectionStore` state changed for
            // whatever runs next against the same weights directory.
            try? await client.touchUp.setModel(id: "yooz-light-v2", preload: false)

            if let matched {
                log("EVENTS_OK event=\(matched)")
            } else {
                log("EVENTS_TIMEOUT no modelChanged frame arrived over /v1/events within 5s")
                exit(1)
            }
        } catch {
            // No typed-vs-unexpected split here, unlike the STT block: a
            // typed `YoozEngineError` from openEvents/setModel has no
            // legitimate excuse on this path, so it is just as much a hard
            // failure as a raw connection error (PR #245 review).
            log("EVENTS_FAIL \(error)")
            exit(1)
        }

        log("HARNESS_DONE")
        exit(0)
    }

    /// Scan `stream` for the first event matching `predicate`, bounded by
    /// `timeoutSeconds` so a genuine regression (no frame ever arrives)
    /// exits the harness promptly instead of hanging forever. The scanning
    /// `for await` runs in its own child task, so it owns the stream's
    /// iterator for the whole call — no cross-task iterator sharing needed.
    /// Losing the race (timeout fires first) cancels that task, which ends
    /// its `for await` loop via `AsyncStream`'s cancellation-aware `next()`.
    private static func firstMatchingEvent(
        _ stream: AsyncStream<EngineEvent>,
        timeoutSeconds: Double,
        where predicate: @escaping @Sendable (EngineEvent) -> Bool
    ) async -> EngineEvent? {
        await withTaskGroup(of: Optional<EngineEvent>.self) { group in
            group.addTask {
                for await event in stream where predicate(event) {
                    return event
                }
                return nil  // the stream ended before a match arrived
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(timeoutSeconds))
                return nil
            }
            defer { group.cancelAll() }
            return await group.next() ?? nil
        }
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
