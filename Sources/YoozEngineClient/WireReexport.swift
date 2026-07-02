// WireReexport.swift
// YoozEngineClient
//
// Copyright 2026 Yooz Labs. All rights reserved.

/// Re-exports every wire DTO from `YoozEngineWire` (#225) so existing
/// `import YoozEngineClient` call sites keep resolving `LLMGenerateResponse`,
/// `TouchUpModelInfo`, `ModulesResponse`, etc. unqualified, exactly as when
/// those types were declared directly in `Sources/YoozEngineClient/Types/`.
@_exported import YoozEngineWire

/// Back-compat typealias for the TouchUp-prefixed name shipped in #97.
/// Remove once consumer apps (whisper, notes, voice) are known to be on the
/// new name (target: v0.7.x of the engine SDK).
public typealias TouchUpModelTier = ModelTier

/// Back-compat typealias. See `TouchUpModelTier` for removal plan.
public typealias TouchUpModelLoadState = ModelLoadState
