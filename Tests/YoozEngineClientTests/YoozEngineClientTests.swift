import XCTest
@testable import YoozEngineClient

final class YoozEngineClientTests: XCTestCase {

    func testClientInitialization() {
        let client = YoozEngineClient()
        XCTAssertEqual(client.baseURL.absoluteString, "http://127.0.0.1:19920")
    }

    func testClientCustomPort() {
        let client = YoozEngineClient(port: 8080)
        XCTAssertEqual(client.baseURL.absoluteString, "http://127.0.0.1:8080")
    }

    func testDefaultClientUsesHTTPTransport() {
        // Backward compat: the no-arg / host+port initializers must still build
        // a loopback HTTP transport (epic #192 Phase 2 seam).
        let client = YoozEngineClient()
        XCTAssertTrue(client.transport is HTTPTransport)
        XCTAssertEqual(client.port, 19920)
    }

    /// The core seam guarantee: every SDK call routes through the injected
    /// `EngineTransport`, not a hardcoded HTTP path. A spy records the path
    /// and returns canned JSON; if `YoozEngineClient` ignored its transport
    /// this would not see the request.
    func testClientDelegatesToInjectedTransport() async throws {
        let spy = SpyTransport()
        let client = YoozEngineClient(transport: spy)
        XCTAssertEqual(client.baseURL.absoluteString, "spy://test")
        XCTAssertEqual(client.port, 0)

        let health = try await client.health()
        XCTAssertTrue(health.isHealthy)
        XCTAssertEqual(spy.getPaths, ["/v1/health"])
    }

    // Helper-launch internals moved from `YoozEngineClient` to `HTTPTransport`
    // behind the transport seam (epic #192 Phase 2). The contract is unchanged;
    // the tests now target the transport directly.
    #if canImport(AppKit)
    func testHelperLaunchEnvironmentIncludesCustomPort() {
        let transport = HTTPTransport(port: 19921)
        XCTAssertEqual(transport.helperLaunchEnvironment[HTTPTransport.headlessEnvVar], "1")
        XCTAssertEqual(transport.helperLaunchEnvironment[HTTPTransport.portEnvVar], "19921")
    }

    /// The argv channel is the reliable headless signal on macOS 26
    /// (`OpenConfiguration.arguments` IS propagated by LaunchServices,
    /// while `OpenConfiguration.environment` is NOT). Pin the flag
    /// spelling so the SDK and the engine-side detector
    /// (`EngineConfig.helperModeArg`) cannot drift apart silently.
    func testHelperLaunchArgumentsCarryHeadlessFlag() {
        let transport = HTTPTransport(port: 19921)
        XCTAssertEqual(transport.helperLaunchArguments, [HTTPTransport.helperModeArg])
        XCTAssertEqual(HTTPTransport.helperModeArg, "--headless")
    }

    func testBundledHelperLaunchConfigurationCanCreateNewInstance() {
        let transport = HTTPTransport(port: 19921)
        let config = transport.helperOpenConfiguration(createsNewInstance: true)
        XCTAssertFalse(config.activates)
        XCTAssertTrue(config.createsNewApplicationInstance)
        XCTAssertEqual(config.environment[HTTPTransport.portEnvVar], "19921")
    }

    /// `helperOpenConfiguration` must populate BOTH channels — the
    /// env-var channel for backward compat with engine builds that
    /// pre-date the argv path, and the argv channel for the reliable
    /// macOS 26 launch path. Verifies the belt-and-suspenders contract
    /// the engine's `EngineConfig.isHelperMode` expects (#117).
    func testHelperOpenConfigurationPopulatesBothHeadlessChannels() {
        let transport = HTTPTransport(port: 19921)
        let config = transport.helperOpenConfiguration(createsNewInstance: true)
        XCTAssertEqual(
            config.environment[HTTPTransport.headlessEnvVar],
            "1",
            "env channel kept for backward compat with engine builds pre-#117"
        )
        XCTAssertTrue(
            config.arguments.contains(HTTPTransport.helperModeArg),
            "argv channel is the reliable headless signal on macOS 26 (#117)"
        )
    }
    #endif

    func testHealthStatusDecoding() throws {
        let json = """
        {
            "status": "ok",
            "version": "0.1.0",
            "modules": {
                "stt": true,
                "llm": false,
                "touchup": false,
                "grammar": true,
                "vad": false,
                "tts": false
            }
        }
        """
        let data = json.data(using: .utf8)!
        let health = try JSONDecoder().decode(HealthStatus.self, from: data)
        XCTAssertTrue(health.isHealthy)
        XCTAssertEqual(health.version, "0.1.0")
        XCTAssertTrue(health.modules.stt)
        XCTAssertFalse(health.modules.llm)
        // Back-compat: older engines omit `infinite`, which decodes to nil.
        XCTAssertNil(health.modules.infinite)
    }

    func testHealthStatusDecodesInfiniteField() throws {
        func decodeInfinite(_ value: String) throws -> Bool? {
            let json = """
            {
                "status": "ok",
                "version": "0.1.0",
                "modules": {
                    "stt": false, "llm": false, "touchup": false,
                    "grammar": false, "vad": false, "tts": false,
                    "infinite": \(value)
                }
            }
            """
            let health = try JSONDecoder().decode(HealthStatus.self, from: Data(json.utf8))
            return health.modules.infinite
        }
        XCTAssertEqual(try decodeInfinite("true"), true)
        XCTAssertEqual(try decodeInfinite("false"), false)
    }

    func testTouchUpRequestEncoding() throws {
        let request = TouchUpRequest(text: "hello world", mode: .standard)
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(TouchUpRequest.self, from: data)
        XCTAssertEqual(decoded.text, "hello world")
        XCTAssertEqual(decoded.mode, .standard)
    }

    // MARK: - STT Types

    func testSTTLanguageAllCases() {
        XCTAssertEqual(STTLanguage.allCases.count, 17)
        XCTAssertEqual(STTLanguage.english.rawValue, "en")
        XCTAssertEqual(STTLanguage.arabic.rawValue, "ar")
        XCTAssertEqual(STTLanguage.persian.rawValue, "fa")
    }

    func testTranscriptionResultDecoding() throws {
        let json = """
        {
            "text": "hello world",
            "finalized": "hello",
            "draft": "world",
            "language": "en"
        }
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.finalized, "hello")
        XCTAssertEqual(result.draft, "world")
        XCTAssertEqual(result.language, "en")
    }

    func testTranscriptionResultDecodingNullLanguage() throws {
        // Language omitted from JSON (optional field)
        let json = """
        {"text": "hello", "finalized": "hello", "draft": ""}
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.text, "hello")
        XCTAssertNil(result.language)
    }

    // MARK: - AlignedToken + TranscriptionResult token roundtrip (engine#34)

    func testAlignedTokenEncodeDecodeRoundTrip() throws {
        let token = AlignedToken(text: " hello", start: 0.12, end: 0.48)
        let data = try JSONEncoder().encode(token)
        let decoded = try JSONDecoder().decode(AlignedToken.self, from: data)
        XCTAssertEqual(decoded, token)
    }

    func testAlignedTokenJSONShape() throws {
        // Wire shape must be `{text, start, end}` — used by the server at
        // /v1/stt/batch?aligned=true and by whisper's chunk-boundary dedup.
        let token = AlignedToken(text: "world", start: 1.5, end: 2.25)
        let data = try JSONEncoder().encode(token)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["text"] as? String, "world")
        XCTAssertEqual(try XCTUnwrap(json["start"] as? Double), 1.5, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(json["end"] as? Double), 2.25, accuracy: 0.001)
        XCTAssertNil(json["duration"],
                     "wire format uses `end`, not `duration`, for backend portability")
    }

    func testTranscriptionResultWithTokensRoundTrip() throws {
        let original = TranscriptionResult(
            text: "hello world",
            finalized: "hello world",
            draft: "",
            language: "en",
            tokens: [
                AlignedToken(text: "hello", start: 0.0, end: 0.4),
                AlignedToken(text: " world", start: 0.5, end: 0.9)
            ]
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(decoded.text, original.text)
        XCTAssertEqual(decoded.finalized, original.finalized)
        XCTAssertEqual(decoded.draft, original.draft)
        XCTAssertEqual(decoded.language, original.language)
        XCTAssertEqual(decoded.tokens, original.tokens)
    }

    func testTranscriptionResultDecodesOldJSONWithoutTokens() throws {
        // Back-compat: v0.5.x response bodies omit the `tokens` key. The
        // SDK must still decode them cleanly — old whisper clients keep
        // working against a newer engine, and the new SDK keeps working
        // against an older engine (`tokens` is nil).
        let json = """
        {"text":"hi","finalized":"hi","draft":"","language":"en"}
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.text, "hi")
        XCTAssertNil(result.tokens,
                     "absent tokens key must decode to nil, not an empty array")
    }

    func testTranscriptionResultDecodesJSONWithEmptyTokenArray() throws {
        // Aligned silent audio: `tokens: []` is distinct from `tokens: null`.
        // The empty array should round-trip exactly.
        let json = """
        {"text":"","finalized":"","draft":"","language":"en","tokens":[]}
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.tokens, [])
    }

    func testTranscriptionResultDecodesJSONWithTokens() throws {
        // Exact shape the /v1/stt/batch?aligned=true handler emits for
        // Parakeet/FastConformer/AppleSTT. Locks the wire contract.
        let json = """
        {
          "text": "quick brown fox",
          "finalized": "quick brown fox",
          "draft": "",
          "language": "en",
          "tokens": [
            {"text":"quick","start":0.0,"end":0.3},
            {"text":" brown","start":0.35,"end":0.6},
            {"text":" fox","start":0.65,"end":0.9}
          ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.tokens?.count, 3)
        XCTAssertEqual(result.tokens?[0].text, "quick")
        XCTAssertEqual(result.tokens?[0].start, 0.0)
        XCTAssertEqual(try XCTUnwrap(result.tokens?[0].end), 0.3, accuracy: 0.001)
        // Monotonic ordering: start times are non-decreasing in the
        // per-chunk dedup use case. Locks the server's emission order.
        let starts = result.tokens!.map(\.start)
        XCTAssertEqual(starts, starts.sorted(),
                       "server must emit tokens in monotonically non-decreasing start order")
    }

    func testTranscriptionResultEncodingOmitsTokensWhenNil() throws {
        // Backward-compat on the wire: old servers that don't populate
        // tokens must see the SDK emit JSON byte-identical with v0.5.x.
        let result = TranscriptionResult(
            text: "hi",
            finalized: "hi",
            draft: "",
            language: "en",
            tokens: nil
        )
        let data = try JSONEncoder().encode(result)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        // Swift's default keyed encoder emits `null` for nil optionals; we
        // accept either absence or null here to keep this test decoupled
        // from the Codable synthesis strategy.
        let tokensField = json["tokens"]
        XCTAssertTrue(
            tokensField == nil || tokensField is NSNull,
            "tokens must be absent or null when constructed with tokens: nil"
        )
    }

    func testBatchSTTRequestAlignedFlagEncoding() throws {
        // STTClient.batchTranscribeAligned sets aligned=true; verify it
        // lands on the wire where the server route expects it.
        let request = BatchSTTRequest(
            samples: [0.1, 0.2],
            language: "en",
            mode: "normal",
            aligned: true
        )
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["aligned"] as? Bool, true)
    }

    func testBatchSTTRequestAlignedFlagOmittedWhenNil() throws {
        // STTClient.transcribe (non-aligned path) must produce JSON that
        // is byte-identical with v0.5.x traffic. The `aligned` key must be
        // absent, not `null`.
        let request = BatchSTTRequest(
            samples: [0.1],
            language: "en",
            mode: "normal"
        )
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertNil(json["aligned"],
                     "non-aligned requests must not emit the aligned key")
    }

    func testBatchSTTRequestDecodingTolerantToAlignedFlag() throws {
        // The server decodes request bodies with the same type shape; lock
        // both encode and decode paths round-trip through the aligned flag.
        let request = BatchSTTRequest(
            samples: [0.1, 0.2, 0.3],
            language: "en",
            mode: "normal",
            aligned: true
        )
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(BatchSTTRequest.self, from: data)
        XCTAssertEqual(decoded.aligned, true)
        XCTAssertEqual(decoded.samples, [0.1, 0.2, 0.3])
    }

    func testSTTStatusDecoding() throws {
        let json = """
        {
            "loaded": true,
            "language": "en",
            "streaming": false
        }
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(STTStatus.self, from: data)
        XCTAssertTrue(status.loaded)
        XCTAssertEqual(status.language, "en")
        XCTAssertFalse(status.streaming)
    }

    func testSTTStatusDecodingNullLanguage() throws {
        // When no model is loaded, language is null
        let json = """
        {"loaded": false, "language": null, "streaming": false}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(STTStatus.self, from: data)
        XCTAssertFalse(status.loaded)
        XCTAssertNil(status.language)
    }

    /// Forward compat: pre-#41 servers omit `progress`, so the SDK
    /// must still decode the legacy 3-field shape with `progress = nil`.
    func testSTTStatusDecodingOmittedProgressIsNil() throws {
        let json = """
        {"loaded": true, "language": "en", "streaming": false}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(STTStatus.self, from: data)
        XCTAssertNil(status.progress, "Older servers omit progress entirely")
    }

    /// New #41 servers report `progress` as a fraction; the SDK
    /// surfaces it untouched. Without this, polling clients silently
    /// drop the download-percent UX after the server starts emitting it.
    func testSTTStatusDecodingProgressFraction() throws {
        let json = """
        {"loaded": false, "language": "en", "streaming": false, "progress": 0.42}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(STTStatus.self, from: data)
        XCTAssertEqual(try XCTUnwrap(status.progress), 0.42, accuracy: 1e-9)
    }

    // MARK: - LLMStatus (engine #124)

    /// Locked decode for the canonical `/v1/llm/status` wire shape so the
    /// SDK can't silently drift from the server's `LLMStatusResponse`.
    /// Whisper's download-progress banner reads `progress` directly.
    func testLLMStatusDecoding() throws {
        let json = """
        {"loaded": false, "modelId": "yooz-light-v2", "progress": 0.42}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(LLMStatus.self, from: data)
        XCTAssertFalse(status.loaded)
        XCTAssertEqual(status.modelId, "yooz-light-v2")
        XCTAssertEqual(try XCTUnwrap(status.progress), 0.42, accuracy: 1e-9)
    }

    /// Idle / already-loaded shape: server omits `progress`, banner hides.
    func testLLMStatusDecodingOmittedProgressIsNil() throws {
        let json = """
        {"loaded": true, "modelId": "yooz-light-v2", "progress": null}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(LLMStatus.self, from: data)
        XCTAssertTrue(status.loaded)
        XCTAssertNil(status.progress)
    }

    /// Forward-compat: older / minimal server builds may omit `progress`
    /// and `modelId` entirely (not just send `null`). Decoder must not
    /// throw on the missing keys.
    func testLLMStatusDecodingOmittedFieldsAreNil() throws {
        let json = """
        {"loaded": false}
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(LLMStatus.self, from: data)
        XCTAssertFalse(status.loaded)
        XCTAssertNil(status.modelId)
        XCTAssertNil(status.progress)
    }

    /// Round-trip so the encoder produces a body the server's
    /// decoder accepts (e.g. for future client-side preview tooling).
    func testLLMStatusCodableRoundTrip() throws {
        let original = LLMStatus(
            loaded: false,
            modelId: "yooz-quality-v2",
            progress: 0.73
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(LLMStatus.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    /// SDK round-trip for the canonical picker shape (issue #97).
    /// Pinning the wire keys catches an accidental rename on either
    /// side that would cause silent picker breakage in apps.
    func testTouchUpModelInfoCodableRoundTrip() throws {
        let info = TouchUpModelInfo(
            id: "yooz-light-v2",
            displayName: "Yooz-Light",
            description: "Fast proofreading (~200ms)",
            tier: .light,
            sizeBytes: 276 * 1024 * 1024,
            loadState: .loaded,
            isActive: true
        )
        let encoded = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(TouchUpModelInfo.self, from: encoded)
        XCTAssertEqual(decoded, info)
    }

    /// Boundary test (drift catch): the engine app target and the
    /// SDK module both define `TouchUpModelInfo` independently. If
    /// either side renames a JSON key or changes a field type, this
    /// literal JSON sample — the *current* engine wire shape —
    /// fails to decode on the SDK side. Cheaper than a single
    /// shared type and catches the drift that would otherwise
    /// surface as a runtime decode error in production picker UIs.
    func testTouchUpModelsResponseDecoding() throws {
        let json = """
        {
            "models": [
                {
                    "id": "yooz-light-v2",
                    "displayName": "Yooz-Light",
                    "description": "Fast",
                    "tier": "light",
                    "sizeBytes": 289406976,
                    "loadState": "loaded",
                    "isActive": true
                },
                {
                    "id": "yooz-quality-v2",
                    "displayName": "Yooz-Quality",
                    "description": "High quality",
                    "tier": "quality",
                    "sizeBytes": 444596224,
                    "loadState": "available",
                    "isActive": false
                }
            ],
            "activeId": "yooz-light-v2"
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TouchUpModelsResponse.self, from: data)
        XCTAssertEqual(response.models.count, 2)
        XCTAssertEqual(response.activeId, "yooz-light-v2")
        XCTAssertEqual(response.models.first?.id, "yooz-light-v2")
        XCTAssertEqual(response.models.first?.loadState, .loaded)
        XCTAssertTrue(response.models.first?.isActive ?? false)
    }

    /// Forward compat: an SDK consumer running against a newer
    /// engine that ships an unrecognised tier (e.g. `"reserved"`)
    /// must decode `.unknown` instead of failing the whole picker
    /// fetch. Same contract for `loadState` falling back to
    /// `.unavailable`. Without this, a future engine field
    /// addition would brick every shipped SDK consumer.
    func testTouchUpModelTierAndLoadStateForwardCompat() throws {
        let json = """
        {
            "id": "future-model-v9",
            "displayName": "Future",
            "description": "Forward compat",
            "tier": "totally-new-tier",
            "sizeBytes": null,
            "loadState": "totally-new-state",
            "isActive": false
        }
        """
        let data = json.data(using: .utf8)!
        let info = try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
        XCTAssertEqual(info.tier, .unknown)
        XCTAssertEqual(info.loadState, .unavailable)
    }

    // MARK: - STT picker (canonical adopter #2, #99)

    /// Round-trip the new STT picker shape so SDK consumers
    /// (whisper, notes) detect drift between engine and SDK at
    /// build time, not in production.
    func testSTTBackendInfoCodableRoundTrip() throws {
        let info = STTBackendInfo(
            id: "parakeet",
            displayName: "Parakeet (Recommended)",
            description: "Multilingual Latin / European",
            tier: .quality,
            sizeBytes: nil,
            loadState: .available,
            isActive: true,
            supportsBatch: true,
            supportsStreaming: true,
            supportedLanguages: ["en", "es", "fr"]
        )
        let encoded = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(STTBackendInfo.self, from: encoded)
        XCTAssertEqual(decoded, info)
    }

    /// Boundary test (drift catch): the engine app target and SDK
    /// each define `STTBackendInfo` independently. A literal JSON
    /// sample from the engine wire shape must decode on the SDK
    /// side or the picker breaks silently.
    func testSTTBackendsResponseDecoding() throws {
        let json = """
        {
            "backends": [
                {
                    "id": "apple_stt",
                    "displayName": "Apple Speech (On-device)",
                    "description": "On-device, no download",
                    "tier": "premium",
                    "sizeBytes": null,
                    "loadState": "loaded",
                    "isActive": true,
                    "supportsBatch": true,
                    "supportsStreaming": true,
                    "supportedLanguages": ["en", "es"]
                }
            ],
            "activeId": "apple_stt"
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(STTBackendsResponse.self, from: data)
        XCTAssertEqual(response.activeId, "apple_stt")
        XCTAssertEqual(response.backends.first?.tier, .premium)
        XCTAssertTrue(response.backends.first?.isActive ?? false)
    }

    /// `TranscriptionResult.tokens` was dropped during the SDK
    /// simplification; restored in #99 because whisper's
    /// hallucination filter (`ChunkProcessor`) and any
    /// timing-sensitive caller relies on it. Pin the field +
    /// `AlignedToken` shape so a future re-removal breaks the
    /// build, not whisper's runtime.
    func testTranscriptionResultDecodesTokens() throws {
        let json = """
        {
            "text": "hello world",
            "finalized": "hello world",
            "draft": "",
            "language": "en",
            "tokens": [
                { "text": "hello", "start": 0.0, "end": 0.42 },
                { "text": " world", "start": 0.42, "end": 0.95 }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertEqual(result.tokens?.count, 2)
        XCTAssertEqual(result.tokens?.first?.text, "hello")
        XCTAssertEqual(try XCTUnwrap(result.tokens?.last?.end), 0.95, accuracy: 1e-5)
    }

    /// Back-compat: a server that doesn't emit `tokens` decodes
    /// to `nil` (not a decode error). Older Yooz Engine builds in
    /// the wild emit no `tokens` field for non-aligned routes.
    func testTranscriptionResultWithoutTokensDecodesAsNil() throws {
        let json = """
        {"text":"hello","finalized":"hello","draft":""}
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(TranscriptionResult.self, from: data)
        XCTAssertNil(result.tokens)
    }

    func testSTTLanguageInfoDecoding() throws {
        let json = """
        {
            "code": "en",
            "name": "English",
            "implemented": true,
            "family": "parakeet-tdt"
        }
        """
        let data = json.data(using: .utf8)!
        let info = try JSONDecoder().decode(STTLanguageInfo.self, from: data)
        XCTAssertEqual(info.code, "en")
        XCTAssertEqual(info.name, "English")
        XCTAssertTrue(info.implemented)
        XCTAssertEqual(info.family, "parakeet-tdt")
    }

    func testBatchSTTRequestEncoding() throws {
        let request = BatchSTTRequest(samples: [0.1, 0.2, 0.3], language: "en", mode: "normal")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(BatchSTTRequest.self, from: data)
        XCTAssertEqual(decoded.samples, [0.1, 0.2, 0.3])
        XCTAssertEqual(decoded.language, "en")
        XCTAssertEqual(decoded.mode, "normal")
    }

    func testAudioModeEnum() {
        XCTAssertEqual(AudioMode.normal.rawValue, "normal")
        XCTAssertEqual(AudioMode.whispered.rawValue, "whispered")
    }

    func testStreamingSTTResultDecoding() throws {
        let json = """
        {
            "type": "partial",
            "text": "hello",
            "finalized": "hel",
            "draft": "lo"
        }
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(StreamingSTTResult.self, from: data)
        XCTAssertEqual(result.type, "partial")
        XCTAssertEqual(result.text, "hello")
        XCTAssertFalse(result.isFinal)

        let finalJson = """
        {"type":"final","text":"hello world","finalized":"hello world","draft":""}
        """
        let finalData = finalJson.data(using: .utf8)!
        let finalResult = try JSONDecoder().decode(StreamingSTTResult.self, from: finalData)
        XCTAssertTrue(finalResult.isFinal)
    }

    // MARK: - Contract Tests

    func testBatchSTTRequestEncodingWithEnumValues() throws {
        // Use enum raw values to catch drift between enum and string literals
        let request = BatchSTTRequest(
            samples: [0.1],
            language: STTLanguage.arabic.rawValue,
            mode: AudioMode.whispered.rawValue
        )
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["language"] as? String, "ar")
        XCTAssertEqual(json["mode"] as? String, "whispered")
    }

    func testSTTStreamConfigEncoding() throws {
        let config = STTStreamConfig(type: "config", language: "en", mode: "normal")
        let data = try JSONEncoder().encode(config)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["type"] as? String, "config")
        XCTAssertEqual(json["language"] as? String, "en")
        XCTAssertEqual(json["mode"] as? String, "normal")
    }

    func testSTTLoadRequestEncoding() throws {
        let request = STTLoadRequest(language: "ar")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(STTLoadRequest.self, from: data)
        XCTAssertEqual(decoded.language, "ar")
    }

    func testSTTLanguagesResponseDecoding() throws {
        let json = """
        {"languages": [{"code":"en","name":"English","implemented":true,"family":"parakeet-tdt"}]}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(STTLanguagesResponse.self, from: data)
        XCTAssertEqual(response.languages.count, 1)
        XCTAssertEqual(response.languages.first?.code, "en")
    }

    // MARK: - LLM Types

    func testLLMGenerateRequestEncoding() throws {
        let request = LLMGenerateRequest(
            prompt: "hello",
            model: "yooz-light-v2",
            systemPrompt: "Fix grammar"
        )
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "hello")
        XCTAssertEqual(decoded.model, "yooz-light-v2")
        XCTAssertEqual(decoded.systemPrompt, "Fix grammar")
    }

    func testLLMGenerateRequestMinimal() throws {
        let request = LLMGenerateRequest(prompt: "test")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "test")
        XCTAssertNil(decoded.model)
        XCTAssertNil(decoded.systemPrompt)
    }

    func testLLMGenerateResponseDecoding() throws {
        let json = """
        {
            "text": "Hello, world!",
            "model": "yooz-light-v2",
            "tokensGenerated": 5,
            "processingTimeMs": 120
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(response.text, "Hello, world!")
        XCTAssertEqual(response.model, "yooz-light-v2")
        XCTAssertEqual(response.tokensGenerated, 5)
        XCTAssertEqual(response.processingTimeMs, 120)
    }

    func testLLMGenerateResponseMinimal() throws {
        let json = """
        {"text": "result", "model": "yooz-quality-v2"}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(response.text, "result")
        XCTAssertNil(response.tokensGenerated)
        XCTAssertNil(response.processingTimeMs)
    }

    // MARK: - LLM Model Management Types

    func testLLMModelInfoFullRoundTrip() throws {
        // Full field set — exercises encode + decode of the SDK type the
        // whisper dropdown binds against. Equality on the Codable
        // Equatable conformance guards against silent field drift.
        let info = LLMModelInfo(
            id: "yooz-light-v2",
            displayName: "Yooz-Light",
            sizeBytes: 276 * 1024 * 1024,
            loaded: true,
            latencyHintMs: 200
        )
        let data = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(LLMModelInfo.self, from: data)
        XCTAssertEqual(decoded, info)
    }

    func testLLMModelInfoOptionalFieldsDecode() throws {
        // Wire shape emitted by `GET /v1/llm/models` for backends that
        // don't publish size or latency (e.g. Foundation Models). The
        // decoder must accept an absent `sizeBytes` + `latencyHintMs`
        // and return nil rather than throwing.
        let json = """
        {"id":"apple-intelligence","displayName":"Apple Intelligence","loaded":false}
        """
        let data = json.data(using: .utf8)!
        let info = try JSONDecoder().decode(LLMModelInfo.self, from: data)
        XCTAssertEqual(info.id, "apple-intelligence")
        XCTAssertEqual(info.displayName, "Apple Intelligence")
        XCTAssertFalse(info.loaded)
        XCTAssertNil(info.sizeBytes)
        XCTAssertNil(info.latencyHintMs)
    }

    func testLLMModelsResponseRoundTrip() throws {
        // Full server response shape: current id + catalogue. Used by
        // whisper's AI > Touch-up Model dropdown to populate options
        // and highlight the selected entry.
        let response = LLMModelsResponse(
            current: "yooz-light-v2",
            available: [
                LLMModelInfo(
                    id: "yooz-light-v2",
                    displayName: "Yooz-Light",
                    sizeBytes: 289_406_976,
                    loaded: true,
                    latencyHintMs: 200
                ),
                LLMModelInfo(
                    id: "yooz-quality-v2",
                    displayName: "Yooz-Quality",
                    sizeBytes: 1_087_963_136,
                    loaded: false,
                    latencyHintMs: 490
                )
            ]
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(LLMModelsResponse.self, from: data)
        XCTAssertEqual(decoded, response)
    }

    func testLLMModelsResponseDecodesServerJSON() throws {
        // Byte-for-byte shape emitted by the engine's `GET /v1/llm/models`
        // route. Locks the wire contract so a server-side rename is
        // caught by this test rather than by whisper in production.
        let json = """
        {
          "current": "yooz-light-v2",
          "available": [
            {"id":"yooz-light-v2","displayName":"Yooz-Light","sizeBytes":289406976,"loaded":true,"latencyHintMs":200},
            {"id":"yooz-quality-v2","displayName":"Yooz-Quality","sizeBytes":444596224,"loaded":false,"latencyHintMs":490}
          ]
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMModelsResponse.self, from: data)
        XCTAssertEqual(response.current, "yooz-light-v2")
        XCTAssertEqual(response.available.count, 2)
        XCTAssertEqual(response.available[0].id, "yooz-light-v2")
        XCTAssertTrue(response.available[0].loaded)
        XCTAssertEqual(response.available[1].id, "yooz-quality-v2")
        XCTAssertFalse(response.available[1].loaded)
    }

    func testLLMModelSelectionEncoding() throws {
        // `setModel` / `preloadModel` / `unloadModel` all share this
        // request body; verify the on-the-wire shape is exactly
        // `{"model": "..."}` so a server-side decoder change breaks
        // here instead of in whisper traffic.
        let selection = LLMModelSelection(model: "yooz-quality-v2")
        let data = try JSONEncoder().encode(selection)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["model"] as? String, "yooz-quality-v2")
        XCTAssertEqual(json.count, 1, "selection body must carry only the model key")
    }

    // MARK: - TouchUp Types

    func testTouchUpModeEnum() {
        XCTAssertEqual(TouchUpMode.off.rawValue, "off")
        XCTAssertEqual(TouchUpMode.light.rawValue, "light")
        XCTAssertEqual(TouchUpMode.standard.rawValue, "standard")
        XCTAssertEqual(TouchUpMode.full.rawValue, "full")
    }

    func testTouchUpResponseDecoding() throws {
        let json = """
        {
            "result": "Hello, world.",
            "mode": "standard",
            "processingTimeMs": 85
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TouchUpResponse.self, from: data)
        XCTAssertEqual(response.result, "Hello, world.")
        XCTAssertEqual(response.mode, .standard)
        XCTAssertEqual(response.processingTimeMs, 85)
        XCTAssertNil(response.modelUsed)
        XCTAssertNil(response.warnings)
    }

    func testTouchUpRequestWithLanguage() throws {
        let request = TouchUpRequest(text: "bonjour monde", mode: .full, language: "fr")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(TouchUpRequest.self, from: data)
        XCTAssertEqual(decoded.text, "bonjour monde")
        XCTAssertEqual(decoded.mode, .full)
        XCTAssertEqual(decoded.language, "fr")
    }

    func testTouchUpRequestWithoutLanguage() throws {
        let request = TouchUpRequest(text: "hello", mode: .light)
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(TouchUpRequest.self, from: data)
        XCTAssertNil(decoded.language)
    }

    // MARK: - Grammar Types

    func testGrammarCheckRequestEncoding() throws {
        let request = GrammarCheckRequest(text: "I are happy")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(GrammarCheckRequest.self, from: data)
        XCTAssertEqual(decoded.text, "I are happy")
        XCTAssertNil(decoded.categories)
    }

    func testGrammarCheckRequestWithCategories() throws {
        let request = GrammarCheckRequest(text: "test", categories: ["grammar", "basic"])
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(GrammarCheckRequest.self, from: data)
        XCTAssertEqual(decoded.categories, ["grammar", "basic"])
    }

    func testGrammarCheckResponseDecoding() throws {
        let json = """
        {"result": "I am happy", "correctionsApplied": 1}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(GrammarCheckResponse.self, from: data)
        XCTAssertEqual(response.result, "I am happy")
        XCTAssertEqual(response.correctionsApplied, 1)
        XCTAssertNil(response.ruleCount)
    }

    func testGrammarCheckResponseWithRuleCount() throws {
        let json = """
        {"result": "I am happy", "correctionsApplied": 1, "ruleCount": 1355}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(GrammarCheckResponse.self, from: data)
        XCTAssertEqual(response.ruleCount, 1355)
    }

    func testGrammarCheckRequestUsePOSEncoding() throws {
        let request = GrammarCheckRequest(text: "test", categories: ["basic"], usePOS: false)
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["usePOS"] as? Bool, false)
    }

    func testGrammarCheckRequestUsePOSNilOmitted() throws {
        let request = GrammarCheckRequest(text: "test")
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        // usePOS is nil, so it should either be absent or null
        let usePOS = json["usePOS"]
        XCTAssertTrue(usePOS == nil || usePOS is NSNull)
    }

    func testGrammarTierRawValues() {
        XCTAssertEqual(GrammarTier.free.rawValue, "free")
        XCTAssertEqual(GrammarTier.pro.rawValue, "pro")
        XCTAssertEqual(GrammarTier.premium.rawValue, "premium")
    }

    func testGrammarFreeCategoriesContent() {
        XCTAssertEqual(grammarFreeCategories.count, 4)
        XCTAssertTrue(grammarFreeCategories.contains("basic"))
        XCTAssertTrue(grammarFreeCategories.contains("grammar"))
        XCTAssertTrue(grammarFreeCategories.contains("articles"))
        XCTAssertTrue(grammarFreeCategories.contains("informal"))
    }

    func testGrammarAllCategoriesContent() {
        XCTAssertEqual(grammarAllCategories.count, 9)
        // All free categories should be in the full set
        for cat in grammarFreeCategories {
            XCTAssertTrue(grammarAllCategories.contains(cat), "Missing free category: \(cat)")
        }
        // Pro-only categories
        XCTAssertTrue(grammarAllCategories.contains("verbs"))
        XCTAssertTrue(grammarAllCategories.contains("numbers"))
        XCTAssertTrue(grammarAllCategories.contains("punctuation"))
        XCTAssertTrue(grammarAllCategories.contains("style"))
        XCTAssertTrue(grammarAllCategories.contains("advanced"))
    }

    // MARK: - VAD Types

    func testVADResponseDecoding() throws {
        let json = """
        {"segments": [{"startMs": 100, "endMs": 2500, "probability": 0.95}]}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(VADResponse.self, from: data)
        XCTAssertEqual(response.segments.count, 1)
        XCTAssertEqual(response.segments[0].startMs, 100)
        XCTAssertEqual(response.segments[0].endMs, 2500)
        XCTAssertEqual(response.segments[0].probability, 0.95, accuracy: 0.01)
    }

    func testVADRequestResetEncoding() throws {
        let request = VADRequest(samples: [0.1, 0.2], reset: false)
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["reset"] as? Bool, false)
    }

    func testVADRequestResetNilOmitted() throws {
        let request = VADRequest(samples: [0.1], reset: nil)
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let reset = json["reset"]
        XCTAssertTrue(reset == nil || reset is NSNull)
    }

    func testVADResponseEmptySegments() throws {
        let json = """
        {"segments": []}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(VADResponse.self, from: data)
        XCTAssertTrue(response.segments.isEmpty)
    }

    func testSpeechSegmentDecoding() throws {
        let json = """
        {"startMs": 0, "endMs": 1000, "probability": 0.87}
        """
        let data = json.data(using: .utf8)!
        let segment = try JSONDecoder().decode(SpeechSegment.self, from: data)
        XCTAssertEqual(segment.startMs, 0)
        XCTAssertEqual(segment.endMs, 1000)
        XCTAssertEqual(segment.probability, 0.87, accuracy: 0.01)
    }

    // MARK: - Audio Byte Serialization Round-Trip

    func testAudioSamplesByteRoundTrip() throws {
        // Verify the Float32 byte serialization contract used by
        // STTStream.sendAudio and the server's WebSocket binary handler
        let originalSamples: [Float] = [0.0, 1.0, -1.0, 0.5, -0.5, Float.leastNormalMagnitude]

        // Client-side: [Float] -> Data (matches STTStream.sendAudio)
        let data = originalSamples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }

        XCTAssertEqual(data.count, originalSamples.count * MemoryLayout<Float>.size)

        // Server-side: Data -> [Float] (matches APIServer WebSocket handler)
        let sampleCount = data.count / MemoryLayout<Float>.size
        let decoded: [Float] = data.withUnsafeBytes { ptr in
            [Float](unsafeUninitializedCapacity: sampleCount) { dest, initializedCount in
                _ = UnsafeMutableRawBufferPointer(dest).copyBytes(
                    from: UnsafeRawBufferPointer(ptr).prefix(sampleCount * MemoryLayout<Float>.size)
                )
                initializedCount = sampleCount
            }
        }

        XCTAssertEqual(decoded, originalSamples)
    }
}

/// Minimal in-memory `EngineTransport` for verifying `YoozEngineClient`
/// delegation. Not a mock of behavior — it records the path and returns a
/// canned health body so the test can assert the request actually reached the
/// injected transport.
private final class SpyTransport: EngineTransport, @unchecked Sendable {
    let baseURL = URL(string: "spy://test")!
    let port = 0
    private(set) var getPaths: [String] = []

    func connect() async throws {}
    func isReachable() async throws -> Bool { true }

    func get(_ path: String) async throws -> Data {
        getPaths.append(path)
        let health = HealthStatus(
            status: "ok",
            version: "spy",
            modules: ModuleStatus(
                stt: false, llm: false, touchup: false,
                grammar: false, vad: false, tts: false, infinite: nil
            )
        )
        return try JSONEncoder().encode(health)
    }

    func post(_ path: String, body: Data) async throws -> Data { Data() }
    func delete(_ path: String) async throws -> Data { Data() }

    @available(macOS 14.0, iOS 17.0, *)
    func webSocketURL(path: String) throws -> URL { baseURL }
}
