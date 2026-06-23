import AppleSTTModule
import EngineCore
import GrammarModule
import LLMModule
import STTModule
import VADModule

/// Bootstraps the in-process engine: registers the linked module actors into
/// `ModuleRegistry.shared`, mirroring `EngineAppDelegate.registerModules()`
/// (which is app-layer / AppKit and therefore not reusable by a library).
///
/// Model weights load lazily on first use — the per-endpoint handlers in
/// `InProcessTransport` call `start()` / `load()` as needed, exactly as the
/// loopback server does. There is no eager-load step here; `ModuleEagerLoader`
/// lives in the Xcode-only `YoozEngine/Core` tree and is not part of any SPM
/// module target.
///
/// Infinite is intentionally absent: its consumer is the loopback super-yooz
/// host, and the in-process transport reports `unsupportedInProcess` for that
/// API (epic #192).
public actor EngineInProcessHost {
    public static let shared = EngineInProcessHost()

    private var didBootstrap = false

    public init() {}

    /// Register the linked engine modules. Idempotent; safe to call on every
    /// `connect()`.
    public func bootstrap() async {
        guard !didBootstrap else { return }
        didBootstrap = true

        await ModuleRegistry.shared.register(GrammarEngine.shared)
        await ModuleRegistry.shared.register(TouchUpEngine.shared)
        await ModuleRegistry.shared.register(YoozSTTEngine.shared)
        await ModuleRegistry.shared.register(AppleSTTEngine.shared)
        await ModuleRegistry.shared.register(VADEngine.shared)
    }
}
