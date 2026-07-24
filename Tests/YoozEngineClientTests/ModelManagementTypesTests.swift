// ModelManagementTypesTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pin the wire contract for the model-management endpoints: decodes the
// exact JSON `APIServer` emits for `GET /v1/models`, `DELETE /v1/models/:id`,
// and `POST /v1/models/cleanup` through the shared `YoozEngineWire` types
// (#225; server and SDK previously carried separately-named duplicates —
// `ModelInfo`/`ModelsResponse`/`DeleteModelResponse`/`ModelCleanupResponse`
// server-side vs. `ManagedModelInfo`/`ManagedModelsResponse`/
// `DeleteModelResult`/`ModelCleanupResult` SDK-side; the SDK names are now
// canonical).

import XCTest
@testable import YoozEngineClient

final class ModelManagementTypesTests: XCTestCase {
    func testManagedModelsResponseDecodesServerJSON() throws {
        let json = """
        {
          "models": [
            {
              "id": "yooz-quality-v3",
              "module": "llm",
              "displayName": "Yooz-Quality",
              "sizeBytes": 1073741824,
              "cached": true,
              "loaded": false,
              "isActive": true,
              "deletable": false
            },
            {
              "id": "models--mlx-community--parakeet-tdt-0.6b-v3",
              "module": "stt",
              "displayName": "parakeet-tdt-0.6b-v3",
              "sizeBytes": 734003200,
              "cached": true,
              "loaded": false,
              "isActive": false,
              "deletable": true
            }
          ]
        }
        """
        let data = Data(json.utf8)
        let response = try JSONDecoder().decode(ManagedModelsResponse.self, from: data)
        XCTAssertEqual(response.models.count, 2)

        let llm = response.models[0]
        XCTAssertEqual(llm.id, "yooz-quality-v3")
        XCTAssertEqual(llm.module, "llm")
        XCTAssertEqual(llm.sizeBytes, 1_073_741_824)
        XCTAssertTrue(llm.cached)
        XCTAssertTrue(llm.isActive)
        XCTAssertFalse(llm.deletable)

        let stt = response.models[1]
        XCTAssertEqual(stt.id, "models--mlx-community--parakeet-tdt-0.6b-v3")
        XCTAssertTrue(stt.deletable)
    }

    func testDeleteModelResultRoundTrips() throws {
        let original = DeleteModelResult(id: "yooz-light-v3", reclaimedBytes: 289_406_976)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(DeleteModelResult.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testModelCleanupResultDecodesServerJSON() throws {
        let json = """
        {
          "totalReclaimedBytes": 5368709120,
          "perRepo": {
            "models--mlx-community--parakeet-tdt-0.6b-v3": 4294967296,
            "models--YoozLabs--Yooz-Quality-v3-Qwen3.5-4B": 1073741824
          }
        }
        """
        let data = Data(json.utf8)
        let result = try JSONDecoder().decode(ModelCleanupResult.self, from: data)
        XCTAssertEqual(result.totalReclaimedBytes, 5_368_709_120)
        XCTAssertEqual(result.perRepo.count, 2)
        XCTAssertEqual(
            result.perRepo["models--mlx-community--parakeet-tdt-0.6b-v3"],
            4_294_967_296
        )
    }
}
