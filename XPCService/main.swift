// main.swift
// YoozEngineXPC
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import STTModule
import YoozEngineClient
import YoozEngineInProcess

// Sandboxed XPC service entry point (engine#227, epic #192 Phase 3
// packaging). System-launched by launchd/xpcproxy from an app's
// `Contents/XPCServices/YoozEngineXPC.xpc` — never run directly.
//
// Forwards every request to `InProcessTransport`, exactly like a standalone
// app's in-process build: there is no third route table. The service is the
// SAME dispatch surface as `YoozEngineInProcess`, just reached over
// `NSXPCConnection` instead of a direct call, so the listener setup below
// stays a few lines on purpose — it mirrors the delegate/listener wiring in
// `XPCServiceHandler.swift`'s doc comment verbatim; the app-group redirect
// step ahead of it is this file's one addition beyond that reference shape.

// Redirect the HF weights cache into the shared app-group container
// (docs/engine-app-packaging.md "Weights") BEFORE any module touches the HF
// downloader, so the very first fetch lands in the right place. The group id
// comes from this bundle's own Info.plist (`YoozAppGroupIdentifier`,
// Xcode-substituted `$(TeamIdentifierPrefix)...` at build time — no runtime
// team-id lookup needed). No-ops silently when unresolvable (no entitlement /
// no provisioning profile / unsandboxed dev run) — see
// `AppGroupWeightsLocation`'s documented fallback contract.
if let groupID = Bundle.main.object(forInfoDictionaryKey: "YoozAppGroupIdentifier") as? String {
    AppGroupWeightsLocation.redirectHuggingFaceCache(groupIdentifier: groupID)
}

// Proactive STT warmup (engine#252, yooz-labs/yooz-whisper#280 follow-up).
// This service is on-demand (launchd tears it down once the last connection
// closes) and never stays warm across app launches the way the loopback host
// does. Measured: the first Parakeet batch call in a freshly-launched
// process took ~105s for 111s of audio (first-ever Metal/MLX shader JIT
// compilation for this binary), vs. ~4s for a second call in the same,
// now-warm process — matching the loopback baseline exactly. Pay that cost
// here, off the user's first real request, instead of on it.
//
// Fire-and-forget: does NOT block `listener.resume()` below, so the service
// starts accepting connections immediately; `/v1/health` etc. are unaffected
// while warmup runs. Mechanics (cache-check before any download, the
// enqueueLoad coalescing, the bounded dummy inference, backend gating) all
// live in `YoozSTTEngine.warmupIfNeeded` — this call site is just language
// resolution + the fire-and-forget kickoff. Runs unconditionally on every
// spawn, including connections that never touch STT: at ~2s of warm-cost
// once cached (0 cost, no download, when not cached — see
// `warmupIfNeeded`'s doc), that is an acceptable tradeoff against the ~105s
// cold-start alternative for the STT connections that do arrive.
Task {
    let language = await YoozSTTEngine.resolveWarmupLanguage()
    YoozSTTEngine.shared.warmupIfNeeded(language: language)
}

// `NSXPCListener.service()`'s `resume()` does not return under normal
// operation — the process lives for the lifetime of the connection(s)
// launchd hands it.
let delegate = XPCServiceListenerDelegate {
    XPCServiceHandler(transport: InProcessTransport())
}
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
