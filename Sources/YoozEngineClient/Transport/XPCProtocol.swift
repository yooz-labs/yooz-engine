import Foundation

/// XPC service interface for the engine (epic #192 Phase 3).
///
/// Deliberately **byte-level** to mirror the `EngineTransport` seam: one method
/// carries `(method, path, body)` and replies with the response `Data`, so no
/// per-endpoint `NSSecureCoding` payload types are needed and the same wire DTOs
/// the SDK already decodes flow across the boundary unchanged. (Streaming STT,
/// which can't use a once-and-only-once reply block, lands in Phase 3b as a
/// bidirectional callback proxy.)
@objc public protocol YoozEngineXPCProtocol {
    /// - Parameters:
    ///   - method: `"GET"`, `"POST"`, or `"DELETE"`.
    ///   - path: the `/v1/...` path (may include a `?query`).
    ///   - body: JSON request body for `POST` (nil otherwise).
    ///   - reply: invoked once with either the response `Data` or an `Error`.
    func request(
        method: String,
        path: String,
        body: Data?,
        withReply reply: @escaping (Data?, Error?) -> Void
    )
}

/// Maps `YoozEngineError` to/from `NSError` so the SDK's typed errors survive the
/// XPC boundary (XPC carries `NSError`, not Swift enums). `userInfo` values are
/// `NSSecureCoding`-compatible (String/Int).
enum XPCErrorBridge {
    static let domain = "live.yooz.engine.xpc"

    static func toNSError(_ error: Error) -> NSError {
        guard let yooz = error as? YoozEngineError else {
            return error as NSError
        }
        let description = yooz.errorDescription ?? "engine error"
        switch yooz {
        case .serverError(let statusCode, let code, let message):
            return NSError(domain: domain, code: 1, userInfo: [
                NSLocalizedDescriptionKey: message,
                "kind": "serverError",
                "statusCode": statusCode,
                "code": code,
                "message": message,
            ])
        case .unsupportedOperation(let operation):
            return NSError(domain: domain, code: 2, userInfo: [
                NSLocalizedDescriptionKey: description,
                "kind": "unsupportedOperation",
                "operation": operation,
            ])
        case .httpError(let statusCode):
            return NSError(domain: domain, code: 3, userInfo: [
                NSLocalizedDescriptionKey: description,
                "kind": "httpError",
                "statusCode": statusCode,
            ])
        case .decodingError(let message):
            return NSError(domain: domain, code: 4, userInfo: [
                NSLocalizedDescriptionKey: description,
                "kind": "decodingError",
                "message": message,
            ])
        default:
            return NSError(domain: domain, code: 99, userInfo: [
                NSLocalizedDescriptionKey: description,
                "kind": "other",
            ])
        }
    }

    static func toYoozEngineError(_ error: Error) -> YoozEngineError {
        let nsError = error as NSError
        // Any non-bridged error (e.g. NSXPCConnection invalidation/interruption)
        // means the service is unreachable.
        guard nsError.domain == domain else {
            return .engineNotReachable
        }
        switch nsError.userInfo["kind"] as? String {
        case "serverError":
            return .serverError(
                statusCode: nsError.userInfo["statusCode"] as? Int ?? 0,
                code: nsError.userInfo["code"] as? String ?? "",
                message: nsError.userInfo["message"] as? String ?? nsError.localizedDescription
            )
        case "unsupportedOperation":
            return .unsupportedOperation(operation: nsError.userInfo["operation"] as? String ?? "")
        case "httpError":
            return .httpError(statusCode: nsError.userInfo["statusCode"] as? Int ?? 0)
        case "decodingError":
            return .decodingError(nsError.userInfo["message"] as? String ?? nsError.localizedDescription)
        default:
            return .invalidResponse
        }
    }
}
