import YoozEngineClient

// SDK wire-type aliases.
//
// `InProcessTransport.swift` imports the engine modules (STTModule, EngineCore,
// …) alongside the SDK. Before #225, several wire DTOs (`ModulesResponse`,
// `TouchUpModelInfo`, `LLMStatus`, ...) had independent SDK-side and
// EngineCore-side declarations, so referencing the bare name was ambiguous —
// this file's `SDK`-prefixed aliases picked one arm to disambiguate. #225
// moved every one of those types into `YoozEngineWire`, which both
// `EngineCore` and `YoozEngineClient` now re-export; there is only one
// declaration left, so the bare name resolves without ambiguity and those
// aliases were deleted.
//
// What's left below are genuine, permanent collisions, not #225 artifacts:
// `STTModule` has its own internal `AlignedToken` (`ParakeetConfiguration.swift`)
// and `TranscriptionResult` (`ParakeetModel.swift`) domain types — the
// batch-transcription backend's native result shape (e.g. `start` +
// `duration` rather than the wire's `start` + `end`) — and `LLMModule` has
// its own internal `LLMModelInfo` (`TouchUp/TouchUpEngine.swift`, fields
// `type` / `isLoaded` / `isCached`) — all deliberately distinct from the
// `YoozEngineWire` wire types of the same name. The usual fix —
// module-qualifying as `YoozEngineClient.AlignedToken` — does NOT work
// because the module `YoozEngineClient` shares its name with the SDK's
// `YoozEngineClient` class, so the dotted form resolves to a (non-existent)
// class member. This file imports ONLY `YoozEngineClient`, so the names
// resolve unambiguously here; the `SDK`-prefixed aliases are then used in the
// transport, free of any collision.
typealias SDKAlignedToken = AlignedToken
typealias SDKTranscriptionResult = TranscriptionResult
typealias SDKLLMModelInfo = LLMModelInfo
