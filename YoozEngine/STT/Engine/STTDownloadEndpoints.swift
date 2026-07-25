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
                // Backend only — NO language validation on this path (PR #294
                // review). Cancellation is keyed purely by backend in the
                // coordinator, and running the download path's
                // `supportedLanguages` check here made cancel unusable for
                // FastConformer: it supports [ar, fa, he], so a client
                // cancelling without a language (defaulting to English) got a
                // 400 and the multi-GB fetch kept running.
                let backend = try downloadableBackend(request.body)
                await STTDownloadCoordinator.shared.cancelDownload(backend)
                return try WireResponse.json(
                    STTDownloadResponse(id: backend.rawValue, downloading: false)
                )
            },
        ]
    }

    /// Backend-only validation, shared by both routes: decode the body and
    /// resolve a DOWNLOADABLE backend id. Language is deliberately not
    /// considered — cancellation doesn't need one, and the download path
    /// layers its own language check on top (PR #294 review).
    private static func downloadableBackend(_ body: Data) throws -> STTBackendID {
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
        return backend
    }

    /// Download-path validation: backend id plus the language whose repo the
    /// fetch should pull. Only the download route needs this — see the cancel
    /// handler for why cancellation must not enforce a language.
    private static func decodeRequest(
        _ body: Data
    ) throws -> (STTBackendID, STTLanguage) {
        let backend = try downloadableBackend(body)
        let request: STTDownloadRequest
        do {
            request = try JSONDecoder().decode(STTDownloadRequest.self, from: body)
        } catch {
            throw WireError.invalidRequest(error)
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
