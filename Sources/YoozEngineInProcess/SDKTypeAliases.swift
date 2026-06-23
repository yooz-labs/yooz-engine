import YoozEngineClient

// SDK wire-type aliases.
//
// `InProcessTransport.swift` imports the engine modules (STTModule, EngineCore,
// …) alongside the SDK, where several type names collide: `AlignedToken` and
// `TranscriptionResult` (SDK vs STTModule), `ModulesResponse` / `ModuleManifest`
// (SDK vs EngineCore). The usual fix — module-qualifying as
// `YoozEngineClient.AlignedToken` — does NOT work because the module
// `YoozEngineClient` shares its name with the SDK's `YoozEngineClient` class, so
// the dotted form resolves to a (non-existent) class member.
//
// This file imports ONLY `YoozEngineClient`, so the names resolve unambiguously
// here. The `SDK`-prefixed aliases are then used in the transport, free of any
// collision.
typealias SDKHealthStatus = HealthStatus
typealias SDKModuleStatus = ModuleStatus
typealias SDKModulesResponse = ModulesResponse
typealias SDKModuleManifest = ModuleManifest
typealias SDKGrammarCheckResponse = GrammarCheckResponse
typealias SDKVADResponse = VADResponse
typealias SDKSpeechSegment = SpeechSegment
typealias SDKTranscriptionResult = TranscriptionResult
typealias SDKAlignedToken = AlignedToken
typealias SDKLLMGenerateResponse = LLMGenerateResponse

// Status + picker DTOs (Phase 2b). Several collide with engine module types
// (`ModelTier`, `ModelLoadState`, `TouchUpModelInfo`, `TouchUpMode`).
typealias SDKSTTStatus = STTStatus
typealias SDKSTTLanguagesResponse = STTLanguagesResponse
typealias SDKSTTLanguageInfo = STTLanguageInfo
typealias SDKSTTBackendInfo = STTBackendInfo
typealias SDKSTTBackendsResponse = STTBackendsResponse
typealias SDKModelTier = ModelTier
typealias SDKModelLoadState = ModelLoadState
typealias SDKLLMStatus = LLMStatus
typealias SDKLLMModelInfo = LLMModelInfo
typealias SDKLLMModelsResponse = LLMModelsResponse
typealias SDKTouchUpModelInfo = TouchUpModelInfo
typealias SDKTouchUpModelsResponse = TouchUpModelsResponse
typealias SDKTouchUpResponse = TouchUpResponse
typealias SDKTouchUpMode = TouchUpMode
