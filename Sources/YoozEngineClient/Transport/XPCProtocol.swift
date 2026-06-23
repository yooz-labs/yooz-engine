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
        // Every case carries a `kind` discriminator plus its associated payload,
        // so the typed error reconstructs losslessly on the other side (no case
        // is flattened to a generic `.invalidResponse`).
        var info: [String: Any] = [NSLocalizedDescriptionKey: yooz.errorDescription ?? "engine error"]
        switch yooz {
        case .engineNotInstalled:
            info["kind"] = "engineNotInstalled"
        case .engineNotReachable:
            info["kind"] = "engineNotReachable"
        case .engineLaunchFailed(let reason):
            info["kind"] = "engineLaunchFailed"
            info["reason"] = reason
        case .portHeldByStaleEngine(let port):
            info["kind"] = "portHeldByStaleEngine"
            info["port"] = port
        case .invalidResponse:
            info["kind"] = "invalidResponse"
        case .httpError(let statusCode):
            info["kind"] = "httpError"
            info["statusCode"] = statusCode
        case .serverError(let statusCode, let code, let message):
            info["kind"] = "serverError"
            info["statusCode"] = statusCode
            info["code"] = code
            info["message"] = message
        case .decodingError(let message):
            info["kind"] = "decodingError"
            info["message"] = message
        case .webSocketError(let message):
            info["kind"] = "webSocketError"
            info["message"] = message
        case .unsupportedOperation(let operation):
            info["kind"] = "unsupportedOperation"
            info["operation"] = operation
        }
        return NSError(domain: domain, code: 1, userInfo: info)
    }

    static func toYoozEngineError(_ error: Error) -> YoozEngineError {
        let nsError = error as NSError
        // A non-bridged error (e.g. NSXPCConnection invalidation/interruption)
        // means the service is unreachable.
        guard nsError.domain == domain else {
            return .engineNotReachable
        }
        let info = nsError.userInfo
        let message = info["message"] as? String ?? nsError.localizedDescription
        switch info["kind"] as? String {
        case "engineNotInstalled":     return .engineNotInstalled
        case "engineNotReachable":     return .engineNotReachable
        case "engineLaunchFailed":     return .engineLaunchFailed(info["reason"] as? String ?? message)
        case "portHeldByStaleEngine":  return .portHeldByStaleEngine(port: info["port"] as? Int ?? 0)
        case "invalidResponse":        return .invalidResponse
        case "httpError":              return .httpError(statusCode: info["statusCode"] as? Int ?? 0)
        case "serverError":
            return .serverError(
                statusCode: info["statusCode"] as? Int ?? 0,
                code: info["code"] as? String ?? "",
                message: message
            )
        case "decodingError":          return .decodingError(message)
        case "webSocketError":         return .webSocketError(message)
        case "unsupportedOperation":   return .unsupportedOperation(operation: info["operation"] as? String ?? "")
        default:                       return .invalidResponse
        }
    }
}
