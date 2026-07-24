// TouchUpEndpoints.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Table endpoints for the TouchUp picker family (engine#225 Phase B).
///
/// Lives in LLMModule — the module that owns `TouchUpEngine` — so the single
/// handler body compiles into every product that links the module: the
/// loopback app targets bind it through `APIServer`'s table registration and
/// `YoozEngineInProcess` binds it through `InProcessTransport`'s dispatch
/// table. Before the table, the two transports hand-mirrored this logic and
/// had already drifted: the in-process copy propagated raw `setActiveModel`
/// errors where the loopback mapped them onto the canonical picker wire
/// codes (AGENTS.md "Wire codes"). Both now speak the loopback's mapping.
public enum TouchUpEndpoints {
    public static func pickerEndpoints() -> [Endpoint] {
        [
            // `GET /v1/touchup/models` — canonical picker list. Exactly one
            // row has `isActive == true` (precondition'd in
            // `availableModels()`).
            Endpoint(EndpointSpecs.touchUpModels) { _ in
                let models = await TouchUpEngine.shared.availableModels()
                let activeId = await TouchUpEngine.shared.activeModel.rawValue
                return try WireResponse.json(
                    TouchUpModelsResponse(models: models, activeId: activeId)
                )
            },
            // `POST /v1/touchup/model` — set the active model. `preload`
            // defaults to true so a one-shot picker change warms the model
            // before the next `/v1/touchup` call. Returns the new active row
            // so clients don't need a follow-up GET.
            //
            // Non-blocking (engine#226): routes through
            // `setActiveModelAsync`, which records + persists the selection
            // and returns immediately — a multi-GB download never holds the
            // HTTP/in-process request open (the pre-#226 contract, which
            // timed out on slow links; engine#125 deferred fixing this for
            // the analogous STT picker). Progress and the eventual
            // loaded/failed outcome arrive via `/v1/events`, not this
            // response.
            Endpoint(EndpointSpecs.touchUpSetModel) { request in
                let body: TouchUpSetModelRequest
                do {
                    body = try JSONDecoder().decode(TouchUpSetModelRequest.self, from: request.body)
                } catch {
                    throw WireError.invalidRequest(error)
                }

                guard let selection = TouchUpModelSelection(rawValue: body.id) else {
                    let known = TouchUpModelSelection.allCases.map(\.rawValue)
                        .joined(separator: ", ")
                    throw WireError(
                        status: 400,
                        code: "invalid_model",
                        message: "Unknown TouchUp model id '\(body.id)'. Known: \(known)."
                    )
                }

                do {
                    let active = try await TouchUpEngine.shared.setActiveModelAsync(
                        selection,
                        preload: body.preload ?? true
                    )
                    return try WireResponse.json(active)
                } catch let error as LLMError {
                    // `notAvailable` is the FoundationModels-on-pre-26 case —
                    // surface as 501 so the picker UI can render "not
                    // supported on this Mac" cleanly. Any other load failure
                    // (network, OOM) is 500.
                    switch error {
                    case .notAvailable(let detail):
                        throw WireError(
                            status: 501, code: "model_unavailable", message: detail
                        )
                    default:
                        throw WireError(
                            status: 500,
                            code: "model_set_failed",
                            message: error.localizedDescription
                        )
                    }
                } catch {
                    throw WireError(
                        status: 500,
                        code: "model_set_failed",
                        message: error.localizedDescription
                    )
                }
            },
            // `POST /v1/touchup/download` — explicit download, decoupled
            // from the active selection (engine#288 slice 2). Returns the
            // tier's current row immediately; progress + terminal outcome
            // arrive via `/v1/events`, exactly like a preloading switch.
            Endpoint(EndpointSpecs.touchUpDownload) { request in
                let selection = try Self.downloadableSelection(from: request.body)
                guard let row = await TouchUpEngine.shared.requestDownload(selection) else {
                    throw WireError(
                        status: 500, code: "model_set_failed",
                        message: "no picker row for '\(selection.rawValue)'"
                    )
                }
                return try WireResponse.json(row)
            },
            // `POST /v1/touchup/download/cancel` — abort an in-flight
            // download. No-op (200 with the current row) when the tier
            // isn't downloading; never unloads a resident model.
            Endpoint(EndpointSpecs.touchUpCancelDownload) { request in
                let selection = try Self.downloadableSelection(from: request.body)
                guard let row = await TouchUpEngine.shared.cancelDownload(selection) else {
                    throw WireError(
                        status: 500, code: "model_set_failed",
                        message: "no picker row for '\(selection.rawValue)'"
                    )
                }
                return try WireResponse.json(row)
            },
        ]
    }

    /// Shared body validation for the download/cancel pair: decode
    /// `TouchUpDownloadRequest`, resolve the selection, and reject
    /// FoundationModels (OS-resident, nothing to download) with the same
    /// 400 wire code the picker uses for unknown ids.
    private static func downloadableSelection(
        from body: Data
    ) throws -> TouchUpModelSelection {
        let request: TouchUpDownloadRequest
        do {
            request = try JSONDecoder().decode(TouchUpDownloadRequest.self, from: body)
        } catch {
            throw WireError.invalidRequest(error)
        }
        guard let selection = TouchUpModelSelection(rawValue: request.id),
              selection != .foundationModels
        else {
            let known = TouchUpModelSelection.allCases
                .filter { $0 != .foundationModels }
                .map(\.rawValue).joined(separator: ", ")
            throw WireError(
                status: 400,
                code: "invalid_model",
                message: "'\(request.id)' is not a downloadable model id. Known: \(known)."
            )
        }
        return selection
    }
}
