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

    /// SDK round-trip for the canonical picker shape (issue #97).
    /// Pinning the wire keys catches an accidental rename on either
    /// side that would cause silent picker breakage in apps.
    func testTouchUpModelInfoCodableRoundTrip() throws {
        let info = TouchUpModelInfo(
            id: "yooz-light-v3",
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
                    "id": "yooz-light-v3",
                    "displayName": "Yooz-Light",
                    "description": "Fast",
                    "tier": "light",
                    "sizeBytes": 289406976,
                    "loadState": "loaded",
                    "isActive": true
                },
                {
                    "id": "yooz-quality-v3",
                    "displayName": "Yooz-Quality",
                    "description": "High quality",
                    "tier": "quality",
                    "sizeBytes": 444596224,
                    "loadState": "available",
                    "isActive": false
                }
            ],
            "activeId": "yooz-light-v3"
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(TouchUpModelsResponse.self, from: data)
        XCTAssertEqual(response.models.count, 2)
        XCTAssertEqual(response.activeId, "yooz-light-v3")
        XCTAssertEqual(response.models.first?.id, "yooz-light-v3")
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
            model: "yooz-light-v3",
            systemPrompt: "Fix grammar"
        )
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "hello")
        XCTAssertEqual(decoded.model, "yooz-light-v3")
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
            "model": "yooz-light-v3",
            "tokensGenerated": 5,
            "processingTimeMs": 120
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(response.text, "Hello, world!")
        XCTAssertEqual(response.model, "yooz-light-v3")
        XCTAssertEqual(response.tokensGenerated, 5)
        XCTAssertEqual(response.processingTimeMs, 120)
    }

    func testLLMGenerateResponseMinimal() throws {
        let json = """
        {"text": "result", "model": "yooz-quality-v3"}
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(response.text, "result")
        XCTAssertNil(response.tokensGenerated)
        XCTAssertNil(response.processingTimeMs)
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
