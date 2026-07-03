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
//
// Batch mode (yooz-labs/yooz-whisper#280 regression gate): `--batch-wav
// <path> [--expect-sentences N] [--concurrent-stream]` feeds a 16kHz
// float32 mono WAV through the packaged service's `/v1/stt/batch` (Parakeet,
// since that is the backend #280 was reported against) and prints
// `BATCH_OK <chars> <elapsedMs>` plus the full text, exiting nonzero if the
// text comes back empty. `--concurrent-stream` opens a second, live Parakeet
// streaming session that feeds audio for the duration of the batch call —
// reproducing the "batch fires while a stream is still open" usage pattern
// #280 flagged as untested on the loopback exoneration runs. Requires a
// cached Parakeet checkpoint (set `HF_HUB_CACHE` to a pre-populated cache to
// avoid a 2.5GB download at harness start).

import AVFoundation
import Foundation
import YoozEngineClient

@main
struct HarnessMain {
    /// Must equal `YoozEngineXPC`'s `PRODUCT_BUNDLE_IDENTIFIER` — the XPC
    /// service is addressed by bundle id, not a port.
    static let serviceName = "live.yooz.engine.xpc"

    static func main() async {
        if CommandLine.arguments.contains("--batch-wav") {
            await runBatchMode()
            return
        }

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

    // MARK: - Batch mode (#280)

    private static func runBatchMode() async {
        let arguments = CommandLine.arguments
        guard let wavIndex = arguments.firstIndex(of: "--batch-wav"), wavIndex + 1 < arguments.count else {
            log("BATCH_USAGE --batch-wav <path> [--expect-sentences N] [--concurrent-stream]")
            exit(1)
        }
        let wavPath = arguments[wavIndex + 1]
        let expectSentences = intArgument(arguments, flag: "--expect-sentences")
        let concurrentStream = arguments.contains("--concurrent-stream")
        // The XPC service is on-demand (launchd tears it down once the last
        // connection closes) — a single-shot harness process pays a cold
        // model-load/Metal-shader-JIT tax on its first call that a
        // long-lived consumer app (one NSXPCConnection open for the whole
        // app lifetime, many PTT recordings across it) only pays once.
        // `--warm-runs N` does N throwaway transcribe calls over THIS
        // connection before the measured run, so cold-start cost can be
        // isolated from steady-state latency instead of contaminating
        // every measurement.
        let warmRuns = intArgument(arguments, flag: "--warm-runs") ?? 0

        let samples: [Float]
        do {
            samples = try loadFloat32Samples(path: wavPath)
        } catch {
            log("BATCH_LOAD_FAIL \(error)")
            exit(1)
        }
        log("BATCH_LOADED samples=\(samples.count) path=\(wavPath) concurrentStream=\(concurrentStream) warmRuns=\(warmRuns)")

        let transport = XPCTransport(serviceName: serviceName)
        let client = YoozEngineClient(transport: transport)

        // Parakeet, not Apple STT: #280 was reported against `sttEngine=yooz-parakeet`,
        // and the concurrent-stream hypothesis specifically needs both the
        // batch call and the stream to share `YoozSTTEngine`'s single MLX
        // `ParakeetModel` instance — Apple STT routes through a wholly
        // separate actor and would not exercise that shared state at all.
        do {
            try await client.stt.setEngine(id: "parakeet", preload: false)
        } catch {
            log("BATCH_SET_ENGINE_FAIL \(error)")
            exit(1)
        }

        for i in 0..<warmRuns {
            let warmStart = ContinuousClock.now
            do {
                let warmResult = try await client.stt.transcribe(
                    audioSamples: samples, language: .english, mode: .normal
                )
                let warmElapsedMs = warmStart.duration(to: .now).milliseconds
                log("WARM_RUN \(i) chars=\(warmResult.text.count) elapsedMs=\(warmElapsedMs)")
            } catch {
                log("WARM_RUN \(i) FAILED elapsedMs=\(warmStart.duration(to: .now).milliseconds) error=\(error)")
            }
        }

        var streamTask: Task<Void, Never>?
        if concurrentStream {
            streamTask = Task { await runConcurrentParakeetStream(client: client) }
            // Let the stream actually open (model load if not already
            // resident) before racing the batch call against it — otherwise
            // this just measures two sequential cold-loads instead of a
            // genuine interleave.
            try? await Task.sleep(for: .milliseconds(300))
        }

        let start = ContinuousClock.now
        do {
            let result = try await client.stt.transcribe(
                audioSamples: samples, language: .english, mode: .normal
            )
            let elapsedMs = start.duration(to: .now).milliseconds
            streamTask?.cancel()
            log("BATCH_OK chars=\(result.text.count) elapsedMs=\(elapsedMs)")
            log("BATCH_TEXT: \(result.text)")
            if let expectSentences {
                let covered = sentenceCoverage(result.text, upTo: expectSentences)
                log("SENTENCE_COVERAGE \(covered)/\(expectSentences)")
            }
            if result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                log("BATCH_EMPTY")
                exit(1)
            }
            exit(0)
        } catch {
            let elapsedMs = start.duration(to: .now).milliseconds
            streamTask?.cancel()
            log("BATCH_FAIL elapsedMs=\(elapsedMs) error=\(error)")
            exit(1)
        }
    }

    /// Opens a live Parakeet streaming session and feeds it small audio
    /// chunks in a loop (silence is enough — the point is to keep the
    /// shared `StreamingTranscriber`/`ParakeetModel` busy with MLX
    /// submissions concurrently with the batch call, not to produce a
    /// meaningful transcript) until cancelled by the batch call completing.
    /// Drains `receive()` concurrently so the session's result queue can't
    /// grow unbounded over the run.
    private static func runConcurrentParakeetStream(client: YoozEngineClient) async {
        do {
            let stream = try await client.stt.startStream(language: .english, mode: .normal)
            defer { stream.close() }
            await withTaskGroup(of: Void.self) { group in
                group.addTask {
                    // ~64ms frames at 16kHz, matching the real capture cadence
                    // documented on `StreamingTranscriber`'s partial-emission
                    // cadence floor.
                    let frame = [Float](repeating: 0, count: 1_024)
                    while !Task.isCancelled {
                        do {
                            try await stream.sendAudio(frame)
                        } catch {
                            return
                        }
                        try? await Task.sleep(for: .milliseconds(64))
                    }
                }
                group.addTask {
                    while !Task.isCancelled {
                        guard let result = try? await stream.receive() else { return }
                        _ = result
                    }
                }
                await group.next()
                group.cancelAll()
            }
        } catch {
            log("CONCURRENT_STREAM_FAIL \(error)")
        }
    }

    /// Loads a WAV file's audio as `[Float]` via `AVAudioFile`, converting to
    /// 16kHz mono Float32 if the file isn't already in that format (the test
    /// corpus is, but this keeps the harness usable against arbitrary WAVs).
    private static func loadFloat32Samples(path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let file = try AVAudioFile(forReading: url)
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false
        )!

        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw HarnessError.audioBufferAllocationFailed
        }
        try file.read(into: sourceBuffer)

        if file.processingFormat.sampleRate == 16_000,
           file.processingFormat.channelCount == 1,
           file.processingFormat.commonFormat == .pcmFormatFloat32 {
            return Array(UnsafeBufferPointer(start: sourceBuffer.floatChannelData![0], count: Int(sourceBuffer.frameLength)))
        }

        guard let converter = AVAudioConverter(from: file.processingFormat, to: targetFormat) else {
            throw HarnessError.converterCreationFailed
        }
        let ratio = targetFormat.sampleRate / file.processingFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio) + 1_024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            throw HarnessError.audioBufferAllocationFailed
        }
        var error: NSError?
        var suppliedSource = false
        converter.convert(to: outBuffer, error: &error) { _, statusPointer in
            if suppliedSource {
                statusPointer.pointee = .noDataNow
                return nil
            }
            suppliedSource = true
            statusPointer.pointee = .haveData
            return sourceBuffer
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: outBuffer.floatChannelData![0], count: Int(outBuffer.frameLength)))
    }

    private static func intArgument(_ arguments: [String], flag: String) -> Int? {
        guard let idx = arguments.firstIndex(of: flag), idx + 1 < arguments.count else { return nil }
        return Int(arguments[idx + 1])
    }

    /// Coarse coverage metric: counts how many of the numbered-sentence
    /// markers ("number one" / "number 1" style ordinals, matching the
    /// synthesized test corpus's script) appear in `text`, out of
    /// `expected`. Deliberately loose (substring, not exact ordinal
    /// spelling match) since Parakeet's own ordinal transcription varies
    /// ("11" vs "eleven") — see the corpus's own `script.txt` ground truth.
    private static func sentenceCoverage(_ text: String, upTo expected: Int) -> Int {
        let lowered = text.lowercased()
        var covered = 0
        for n in 1...expected {
            let digitForm = "number \(n)."
            let digitFormComma = "number \(n),"
            if lowered.contains(digitForm) || lowered.contains(digitFormComma) {
                covered += 1
                continue
            }
            if let word = Self.ordinalWords[n], lowered.contains("number \(word)") {
                covered += 1
            }
        }
        return covered
    }

    private static let ordinalWords: [Int: String] = [
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 21: "twenty one", 22: "twenty two", 23: "twenty three",
        24: "twenty four",
    ]

    private enum HarnessError: Error {
        case audioBufferAllocationFailed
        case converterCreationFailed
    }
}
