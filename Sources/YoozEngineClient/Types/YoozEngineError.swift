import Foundation

public enum YoozEngineError: LocalizedError, Sendable, Equatable {
    case engineNotInstalled
    case engineNotReachable
    case engineLaunchFailed(String)
    /// Port :19920 is being held by a process that is not responding to
    /// `/v1/health`. Likely a stale/crashed engine. The user should
    /// terminate the holder or set `YOOZ_ENGINE_AUTO_RECOVER=1` to let
    /// the SDK kill it and relaunch.
    case portHeldByStaleEngine(port: Int)
    case invalidResponse
    case httpError(statusCode: Int)
    /// A non-2xx response that carried the engine's structured error body
    /// (`{"error": ..., "code": ...}`). `code` is the stable machine-readable
    /// identifier (e.g. `generation_unavailable`, `session_not_found`,
    /// `session_limit_exceeded`, `module_not_bundled`) callers should branch on;
    /// `message` is the human-readable text. Falls back to `httpError` when the
    /// body is absent or not the structured shape.
    case serverError(statusCode: Int, code: String, message: String)
    case decodingError(String)
    case webSocketError(String)

    public var errorDescription: String? {
        switch self {
        case .engineNotInstalled:
            return "Yooz Engine is not installed. Please install it from yooz.live"
        case .engineNotReachable:
            return "Yooz Engine is not reachable. Please ensure it is running."
        case .engineLaunchFailed(let reason):
            return "Failed to launch Yooz Engine: \(reason)"
        case .portHeldByStaleEngine(let port):
            return "Port \(port) is held by a process that is not responding "
                + "to Yooz Engine health checks. Likely a stale engine "
                + "instance. Terminate it manually, or set "
                + "YOOZ_ENGINE_AUTO_RECOVER=1 to let the SDK recover automatically."
        case .invalidResponse:
            return "Invalid response from Yooz Engine"
        case .httpError(let code):
            return "HTTP error \(code) from Yooz Engine"
        case .serverError(let code, let errorCode, let message):
            return "Yooz Engine error \(code) [\(errorCode)]: \(message)"
        case .decodingError(let message):
            return "Failed to decode response: \(message)"
        case .webSocketError(let message):
            return "WebSocket error: \(message)"
        }
    }
}
