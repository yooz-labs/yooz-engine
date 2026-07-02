// GPUAdmissionWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// A workload's latency sensitivity for MLX/GPU submission (engine#228).
///
/// Crosses the wire as the optional, additive `workloadClass` field on
/// `TouchUpRequest` / `LLMGenerateRequest`: omitting it preserves the
/// pre-#228 behavior (the engine defaults both routes to `.background`).
/// Set `.interactive` only for a genuinely latency-sensitive one-off call
/// that should never queue behind another interactive workload.
///
/// Single definition shared by the engine (where `EngineCore` aliases it as
/// `MLXWorkloadClass`, the name `MLXAdmissionGate`'s scheduling API uses)
/// and the SDK — #228 originally shipped an SDK-side mirror
/// (`GPUWorkloadClass` in `Types/GPUAdmissionTypes.swift`) held in sync by
/// rawValue; the #225 wire consolidation collapsed the two into this one
/// declaration.
///
/// Deliberately strict (no tolerant-decode fallback like `ModelTier`): a
/// declared scheduling class is explicit caller intent, and silently
/// downgrading a mistyped `.interactive` to `.background` would make the
/// request queue behind other work with no trace of why. Unknown values
/// fail the body decode; both transports surface that as a rejected
/// request (HTTP 400 `invalid_request` on loopback).
public enum GPUWorkloadClass: String, Codable, Sendable {
    /// Latency-sensitive, small — a live streaming STT session, a short
    /// grammar call. Never queues at the engine's admission gate; always
    /// admitted immediately. While active, it signals `background`
    /// submissions to queue or yield.
    case interactive
    /// Throughput work that can tolerate queuing behind an active
    /// interactive workload — batch transcription, TouchUp/LLM generation,
    /// Infinite append/generate (the default for touch-up / raw
    /// generation).
    case background
}
