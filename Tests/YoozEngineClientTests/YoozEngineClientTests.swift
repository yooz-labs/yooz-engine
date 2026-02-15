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
            model: "yooz-light-v1"
        )
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "hello")
        XCTAssertEqual(decoded.model, "yooz-light-v1")
    }

    func testLLMGenerateRequestMinimal() throws {
        let request = LLMGenerateRequest(prompt: "test")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(LLMGenerateRequest.self, from: data)
        XCTAssertEqual(decoded.prompt, "test")
        XCTAssertNil(decoded.model)
    }

    func testLLMGenerateResponseDecoding() throws {
        let json = """
        {
            "text": "Hello, world!",
            "model": "yooz-light-v1",
            "tokensGenerated": 5,
            "processingTimeMs": 120
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(LLMGenerateResponse.self, from: data)
        XCTAssertEqual(response.text, "Hello, world!")
        XCTAssertEqual(response.model, "yooz-light-v1")
        XCTAssertEqual(response.tokensGenerated, 5)
        XCTAssertEqual(response.processingTimeMs, 120)
    }

    func testLLMGenerateResponseMinimal() throws {
        let json = """
        {"text": "result", "model": "yooz-quality-v1"}
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
