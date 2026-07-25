// STTDownloadEndpoints.swift
// STTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Table endpoints for explicit STT model download / cancel (engine#291).
///
/// Mirrors `TouchUpEndpoints`' download pair so both pickers speak the same
/// contract: selecting a backend records a preference and never fetches;
/// downloading is an explicit user action that survives selection changes;
/// cancelling is available while a fetch is in flight.
///
/// Lives in STTModule (the module owning `STTDownloadCoordinator`) so one
/// handler body compiles into every product — the loopback app targets bind
/// it through `APIServer`'s table registration, and `YoozEngineInProcess`
/// binds it through `InProcessTransport`'s dispatch table.
public enum STTDownloadEndpoints {
    public static func endpoints() -> [Endpoint] {
        [
            // `POST /v1/stt/download` — start (or join) a fetch for a
            // backend without changing the active selection. Returns 202-ish
            // semantics: the body echoes the requested id; progress and the
            // terminal outcome arrive via `/v1/events` on the `stt` module.
            Endpoint(EndpointSpecs.sttDownload) { request in
                let (backend, language) = try decodeRequest(request.body)
                await STTDownloadCoordinator.shared.requestDownload(
                    backend, language: language
                )
                return try WireResponse.json(
                    STTDownloadResponse(id: backend.rawValue, downloading: true)
                )
            },
            // `POST /v1/stt/download/cancel` — abort an in-flight fetch.
            // No-op (200, `downloading: false`) when it isn't downloading;
            // never disturbs a settled or loaded backend.
            Endpoint(EndpointSpecs.sttCancelDownload) { request in
                let (backend, _) = try decodeRequest(request.body)
                await STTDownloadCoordinator.shared.cancelDownload(backend)
                return try WireResponse.json(
                    STTDownloadResponse(id: backend.rawValue, downloading: false)
                )
            },
        ]
    }

    /// Shared body validation: resolve the backend id and language, and
    /// reject ids that cannot be downloaded (Apple STT is OS-provided) with
    /// the same 400 `invalid_model` code the pickers already use.
    private static func decodeRequest(
        _ body: Data
    ) throws -> (STTBackendID, STTLanguage) {
        let request: STTDownloadRequest
        do {
            request = try JSONDecoder().decode(STTDownloadRequest.self, from: body)
        } catch {
            throw WireError.invalidRequest(error)
        }
        guard let backend = STTBackendID(rawValue: request.id),
              backend.requiresDownload
        else {
            let known = STTBackendID.allCases
                .filter(\.requiresDownload)
                .map(\.rawValue).joined(separator: ", ")
            throw WireError(
                status: 400,
                code: "invalid_model",
                message: "'\(request.id)' is not a downloadable STT backend. Known: \(known)."
            )
        }
        // Language selects which repo a multilingual backend fetches;
        // default to English so a consumer that only knows the backend id
        // still gets the common case right.
        let language = request.language
            .flatMap(STTLanguage.init(rawValue:)) ?? .english
        guard backend.supportedLanguages.contains(language) else {
            throw WireError(
                status: 400,
                code: "invalid_model",
                message: "\(backend.rawValue) does not support language '\(language.rawValue)'."
            )
        }
        return (backend, language)
    }
}
