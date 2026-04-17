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
