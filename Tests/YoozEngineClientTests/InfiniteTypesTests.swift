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
            description: "Reduced-tier long-context model, proven at ~1M tokens.",
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
              "description": "Reduced-tier long-context model, proven at ~1M tokens.",
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
    }
}
