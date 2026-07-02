// EngineEventBus.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Fan-out publisher backing `/v1/events` (engine#226): every module actor
/// that mutates model-selection state publishes here; every transport
/// (loopback WS, in-process `AsyncStream`) reads from here. One channel for
/// every event kind — coordinates with, and subsumes, the SSE
/// download-progress idea tracked on #127 (that issue stays open; this bus
/// is the "one channel, not two" resolution the engine#226 issue calls for).
///
/// Subscribers get every event published AFTER they subscribe — there is no
/// replay/backlog. The pre-first-event state lives in `GET /v1/state`
/// (`EngineStateSnapshot`), which a client fetches once before opening the
/// event stream (see `EngineStateStore.start()` in the SDK).
///
/// An actor (not a lock-guarded class) so `publish`/`subscribe` serialize
/// without a separate synchronization primitive, matching the rest of this
/// module's actor-first style (`ModelSelectionStore`, `SessionCoordinator`).
public actor EngineEventBus {
    public static let shared = EngineEventBus()

    private var subscribers: [UUID: AsyncStream<EngineEvent>.Continuation] = [:]

    public init() {}

    /// Subscribe to the live event feed. The stream ends when the consuming
    /// `Task` is cancelled or its iteration is otherwise torn down —
    /// `AsyncStream`'s `onTermination` callback removes the subscriber from
    /// the bus either way, so there is no separate `unsubscribe` call.
    public func subscribe() -> AsyncStream<EngineEvent> {
        let id = UUID()
        let (stream, continuation) = AsyncStream<EngineEvent>.makeStream()
        subscribers[id] = continuation
        continuation.onTermination = { @Sendable [weak self] _ in
            guard let self else { return }
            Task { await self.remove(id) }
        }
        return stream
    }

    /// Publish one event to every current subscriber. Never blocks on a
    /// slow consumer: `AsyncStream.Continuation.yield` buffers (unbounded
    /// by default), so a stalled WebSocket write cannot back-pressure the
    /// publisher — typically a model-load `Task` that must not stall on a
    /// UI's event-loop hiccup.
    public func publish(_ event: EngineEvent) {
        for continuation in subscribers.values {
            continuation.yield(event)
        }
    }

    /// Current subscriber count. Test-observability only (no production
    /// caller needs this); not part of the wire contract.
    public var subscriberCount: Int { subscribers.count }

    private func remove(_ id: UUID) {
        subscribers[id] = nil
    }
}
