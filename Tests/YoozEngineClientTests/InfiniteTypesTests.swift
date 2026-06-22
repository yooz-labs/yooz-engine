// InfiniteTypesTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineClient

final class InfiniteTypesTests: XCTestCase {

    func testInfiniteModelInfoCodableRoundTrip() throws {
        let original = InfiniteModelInfo(
            id: "gemma4-e4b-1m",
            displayName: "Gemma4 E4B 1M",
            description: "Reduced-tier long-context model. Single-needle retrieval validated near 1M tokens; interactive tier ~256K (multi-hop degrades beyond).",
            tier: .light,
            sizeBytes: 3_221_225_472,
            loadState: .available,
            isActive: true,
            maxContextTokens: 1_000_000,
            nativeContextTokens: 131_072,
            ramTier: "reduced",
            backendKind: "paged-kv",
            adapterKind: "infinite-paged-kv-mlx-v1",
            huggingFaceID: "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
            revision: "b4966f32e71f9f4976a78f74bc8944b1d064bcbf",
            requiresAppleSilicon: true,
            evidenceRef: "infinite:research/18-gemma-support-matrix.md"
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(InfiniteModelInfo.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testInfiniteModelsResponseDecodesServerJSON() throws {
        let json = """
        {
          "activeId": "gemma4-e4b-1m",
          "models": [
            {
              "id": "gemma4-e4b-1m",
              "displayName": "Gemma4 E4B 1M",
              "description": "Reduced-tier long-context model. Single-needle retrieval validated near 1M tokens; interactive tier ~256K (multi-hop degrades beyond).",
              "tier": "light",
              "sizeBytes": 3221225472,
              "loadState": "available",
              "isActive": true,
              "maxContextTokens": 1000000,
              "nativeContextTokens": 131072,
              "ramTier": "reduced",
              "backendKind": "paged-kv",
              "adapterKind": "infinite-paged-kv-mlx-v1",
              "huggingFaceID": "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
              "revision": "b4966f32e71f9f4976a78f74bc8944b1d064bcbf",
              "requiresAppleSilicon": true,
              "evidenceRef": "infinite:research/18-gemma-support-matrix.md"
            }
          ]
        }
        """
        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(InfiniteModelsResponse.self, from: data)
        XCTAssertEqual(response.activeId, "gemma4-e4b-1m")
        XCTAssertEqual(response.models.count, 1)
        XCTAssertEqual(response.models[0].tier, .light)
        XCTAssertEqual(response.models[0].loadState, .available)
        XCTAssertEqual(response.models[0].maxContextTokens, 1_000_000)
        XCTAssertEqual(response.models[0].nativeContextTokens, 131_072)
        XCTAssertEqual(
            response.models[0].huggingFaceID,
            "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
        )
    }

    func testInfiniteSetModelRequestEncoding() throws {
        let request = InfiniteSetModelRequest(id: "gemma4-26b-a4b-1m", preload: false)
        let data = try JSONEncoder().encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(json["id"] as? String, "gemma4-26b-a4b-1m")
        XCTAssertEqual(json["preload"] as? Bool, false)
        XCTAssertEqual(json.count, 2)
    }

    func testInfiniteStatusDecoding() throws {
        let json = """
        {
          "loaded": false,
          "modelId": "gemma4-e4b-1m",
          "progress": null,
          "state": "idle",
          "activeSessions": 0,
          "maxContextTokens": 1000000,
          "ramTier": "reduced",
          "backendKind": "paged-kv",
          "cleanupPolicy": "explicit_delete_or_process_exit;max_active_sessions=16",
          "resources": {
            "physicalMemoryBytes": 68719476736,
            "wiredMemoryLimitBytes": 34359738368,
            "requiredRAMTier": "reduced",
            "peakMemoryBytes": null,
            "prefillTokensPerSecond": null,
            "decodeTokensPerSecond": null,
            "draftAcceptanceRate": null
          },
          "lastError": null
        }
        """
        let data = json.data(using: .utf8)!
        let status = try JSONDecoder().decode(InfiniteStatus.self, from: data)
        XCTAssertFalse(status.loaded)
        XCTAssertEqual(status.modelId, "gemma4-e4b-1m")
        XCTAssertNil(status.progress)
        XCTAssertEqual(status.state, "idle")
        XCTAssertEqual(status.activeSessions, 0)
        XCTAssertEqual(status.maxContextTokens, 1_000_000)
        XCTAssertEqual(status.cleanupPolicy, "explicit_delete_or_process_exit;max_active_sessions=16")
        XCTAssertEqual(status.resources?.wiredMemoryLimitBytes, 34_359_738_368)
    }

    func testInfiniteSessionLifecycleTypesRoundTrip() throws {
        let metrics = InfiniteResourceMetrics(
            physicalMemoryBytes: 68_719_476_736,
            wiredMemoryLimitBytes: 34_359_738_368,
            requiredRAMTier: "reduced",
            peakMemoryBytes: nil,
            prefillTokensPerSecond: nil,
            decodeTokensPerSecond: 12.5,
            draftAcceptanceRate: nil
        )
        let session = InfiniteSessionInfo(
            id: "session-1",
            modelId: "gemma4-e4b-1m",
            label: "work",
            state: "open",
            createdAt: "2026-06-20T04:00:00Z",
            updatedAt: "2026-06-20T04:01:00Z",
            contextWindowTokens: 1_000_000,
            inputCharacters: 128,
            estimatedInputTokens: 32,
            checkpointCount: 1,
            cleanupPolicy: "explicit_delete_or_process_exit;max_active_sessions=16",
            resources: metrics
        )
        let checkpoint = InfiniteSessionCheckpoint(
            id: "checkpoint-1",
            label: "after-load",
            createdAt: "2026-06-20T04:01:00Z",
            inputCharacters: 128,
            estimatedInputTokens: 32,
            resources: metrics
        )
        let response = InfiniteCheckpointSessionResponse(
            session: session,
            checkpoint: checkpoint
        )

        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(InfiniteCheckpointSessionResponse.self, from: data)
        XCTAssertEqual(decoded, response)
    }

    func testInfiniteSessionRequestEncoding() throws {
        let create = InfiniteCreateSessionRequest(modelId: "gemma4-e4b-1m", label: "doc")
        let createData = try JSONEncoder().encode(create)
        let createJSON = try JSONSerialization.jsonObject(with: createData) as! [String: Any]
        XCTAssertEqual(createJSON["modelId"] as? String, "gemma4-e4b-1m")
        XCTAssertEqual(createJSON["label"] as? String, "doc")

        let append = InfiniteAppendSessionRequest(text: "real context")
        let appendData = try JSONEncoder().encode(append)
        let appendJSON = try JSONSerialization.jsonObject(with: appendData) as! [String: Any]
        XCTAssertEqual(appendJSON["text"] as? String, "real context")

        let generate = InfiniteGenerateSessionRequest(prompt: "summarize", maxTokens: 16)
        let generateData = try JSONEncoder().encode(generate)
        let generateJSON = try JSONSerialization.jsonObject(with: generateData) as! [String: Any]
        XCTAssertEqual(generateJSON["prompt"] as? String, "summarize")
        XCTAssertEqual(generateJSON["maxTokens"] as? Int, 16)
    }
}
