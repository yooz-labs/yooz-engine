import Foundation

public enum YoozEngineError: LocalizedError {
    case engineNotInstalled
    case engineNotReachable
    case invalidResponse
    case httpError(statusCode: Int)
    case decodingError(String)
    case webSocketError(String)

    public var errorDescription: String? {
        switch self {
        case .engineNotInstalled:
            return "Yooz Engine is not installed. Please install it from yooz.live"
        case .engineNotReachable:
            return "Yooz Engine is not reachable. Please ensure it is running."
        case .invalidResponse:
            return "Invalid response from Yooz Engine"
        case .httpError(let code):
            return "HTTP error \(code) from Yooz Engine"
        case .decodingError(let message):
            return "Failed to decode response: \(message)"
        case .webSocketError(let message):
            return "WebSocket error: \(message)"
        }
    }
}
