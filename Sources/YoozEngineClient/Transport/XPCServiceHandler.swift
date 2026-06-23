import Foundation

/// Service-side XPC handler (epic #192 Phase 3): implements
/// `YoozEngineXPCProtocol` by delegating to an injected `EngineTransport` — the
/// consumer wires it with `InProcessTransport()` so the XPC service runs the
/// engine modules in its own sandboxed process.
///
/// Reusable across the real `.xpc` bundle and the anonymous-listener tests.
public final class XPCServiceHandler: NSObject, YoozEngineXPCProtocol, @unchecked Sendable {
    private let transport: any EngineTransport

    public init(transport: any EngineTransport) {
        self.transport = transport
        super.init()
    }

    public func request(
        method: String,
        path: String,
        body: Data?,
        withReply reply: @escaping (Data?, Error?) -> Void
    ) {
        let transport = self.transport
        Task {
            do {
                let data: Data
                switch method {
                case "GET":
                    data = try await transport.get(path)
                case "POST":
                    data = try await transport.post(path, body: body ?? Data())
                case "DELETE":
                    data = try await transport.delete(path)
                default:
                    throw YoozEngineError.serverError(
                        statusCode: 405, code: "method_not_allowed",
                        message: "Unsupported method '\(method)'"
                    )
                }
                reply(data, nil)
            } catch {
                reply(nil, XPCErrorBridge.toNSError(error))
            }
        }
    }
}

/// `NSXPCListenerDelegate` that exports a fresh `XPCServiceHandler` per
/// connection. The `.xpc` bundle's `main` is a few lines around it:
///
/// ```swift
/// import YoozEngineClient
/// import YoozEngineInProcess
///
/// let delegate = XPCServiceListenerDelegate {
///     XPCServiceHandler(transport: InProcessTransport())
/// }
/// let listener = NSXPCListener.service()   // system-launched service
/// listener.delegate = delegate
/// listener.resume()
/// ```
///
/// (Building the `.xpc` bundle itself — Info.plist, entitlements, the
/// team-ID-prefixed app group for the shared model cache — is app/xcodegen work;
/// see `docs/engine-app-packaging.md`.)
public final class XPCServiceListenerDelegate: NSObject, NSXPCListenerDelegate, @unchecked Sendable {
    private let makeHandler: @Sendable () -> XPCServiceHandler

    public init(makeHandler: @escaping @Sendable () -> XPCServiceHandler) {
        self.makeHandler = makeHandler
        super.init()
    }

    public func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: YoozEngineXPCProtocol.self)
        newConnection.exportedObject = makeHandler()
        newConnection.resume()
        return true
    }
}
