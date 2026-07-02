// WireReexport.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

/// Re-exports every wire DTO from `YoozEngineWire` (#225) so existing
/// `import EngineCore` call sites (`LLMModule`, `STTModule`, `GrammarModule`,
/// `AppleSTTModule`, `VADModule`, the app target's `APIServer`/`APITypes`)
/// keep resolving `ModelTier`, `LoadState`, `ModulesResponse`, etc.
/// unqualified, exactly as when those types were declared directly in
/// `Sources/EngineCore/`.
@_exported import YoozEngineWire
