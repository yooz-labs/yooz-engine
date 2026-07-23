// EngineEventBusTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Pins the `/v1/events` fan-out contract (engine#226): every subscriber
/// receives every event published after it subscribes, subscribers don't
/// interfere with each other, and cancelling a subscriber's consuming Task
/// removes it from the bus.
final class EngineEventBusTests: XCTestCase {
    func testSubscriberReceivesPublishedEvent() async {
        let bus = EngineEventBus()
        let stream = await bus.subscribe()
        var iterator = stream.makeAsyncIterator()

        await bus.publish(EngineEvent(kind: .modelChanged, module: "touchup", modelId: "yooz-light-v3"))

        let received = await iterator.next()
        XCTAssertEqual(received?.kind, .modelChanged)
        XCTAssertEqual(received?.module, "touchup")
        XCTAssertEqual(received?.modelId, "yooz-light-v3")
    }

    func testEventsArriveInPublishOrder() async {
        let bus = EngineEventBus()
        let stream = await bus.subscribe()
        var iterator = stream.makeAsyncIterator()

        await bus.publish(EngineEvent(kind: .downloadProgress, module: "touchup", progress: 0.1))
        await bus.publish(EngineEvent(kind: .downloadProgress, module: "touchup", progress: 0.5))
        await bus.publish(EngineEvent(kind: .loadStateChanged, module: "touchup", loadState: .loaded))

        let first = await iterator.next()
        let second = await iterator.next()
        let third = await iterator.next()
        XCTAssertEqual(first?.progress, 0.1)
        XCTAssertEqual(second?.progress, 0.5)
        XCTAssertEqual(third?.loadState, .loaded)
    }

    func testMultipleSubscribersEachGetEveryEvent() async {
        let bus = EngineEventBus()
        let streamA = await bus.subscribe()
        let streamB = await bus.subscribe()
        var iteratorA = streamA.makeAsyncIterator()
        var iteratorB = streamB.makeAsyncIterator()

        await bus.publish(EngineEvent(kind: .residencyChanged, module: "touchup", modelId: "yooz-quality-v3"))

        let a = await iteratorA.next()
        let b = await iteratorB.next()
        XCTAssertEqual(a?.modelId, "yooz-quality-v3")
        XCTAssertEqual(b?.modelId, "yooz-quality-v3")
    }

    /// A subscriber that never subscribed must not see events published
    /// before it joined — no replay/backlog (the pre-first-event state is
    /// `GET /v1/state`, not this bus).
    func testLateSubscriberDoesNotSeePriorEvents() async {
        let bus = EngineEventBus()
        await bus.publish(EngineEvent(kind: .modelChanged, module: "touchup", modelId: "yooz-light-v3"))

        let stream = await bus.subscribe()
        await bus.publish(EngineEvent(kind: .modelChanged, module: "touchup", modelId: "yooz-quality-v3"))

        var iterator = stream.makeAsyncIterator()
        let received = await iterator.next()
        XCTAssertEqual(received?.modelId, "yooz-quality-v3")
    }

    func testCancellingConsumerRemovesSubscriber() async {
        let bus = EngineEventBus()
        let stream = await bus.subscribe()
        let countBefore = await bus.subscriberCount
        XCTAssertEqual(countBefore, 1)

        let task = Task {
            for await _ in stream { /* drain until cancelled */ }
        }
        task.cancel()
        _ = await task.value

        // `onTermination` fires asynchronously off the cancellation; poll
        // briefly rather than asserting immediately.
        var countAfter = await bus.subscriberCount
        for _ in 0..<20 where countAfter != 0 {
            try? await Task.sleep(for: .milliseconds(10))
            countAfter = await bus.subscriberCount
        }
        XCTAssertEqual(countAfter, 0)
    }

    func testEngineEventDefaultsTsToNonEmptyISO8601String() {
        let event = EngineEvent(kind: .modelChanged, module: "touchup", modelId: "yooz-light-v3")
        XCTAssertFalse(event.ts.isEmpty)
        XCTAssertNotNil(ISO8601DateFormatter().date(from: event.ts))
    }

    /// Concurrent publishers racing a live consumer (PR #239 review): the
    /// actor serializes `publish`, so every event from every publisher must
    /// arrive exactly once — no drops, no duplicates — even when many tasks
    /// publish simultaneously while the subscriber drains.
    func testConcurrentPublishersDeliverEveryEventExactlyOnce() async {
        let bus = EngineEventBus()
        let stream = await bus.subscribe()

        let publisherCount = 8
        let eventsPerPublisher = 20
        let total = publisherCount * eventsPerPublisher

        let consumer = Task { () -> [String] in
            var received: [String] = []
            for await event in stream {
                if let id = event.modelId { received.append(id) }
                if received.count == total { break }
            }
            return received
        }

        await withTaskGroup(of: Void.self) { group in
            for publisher in 0..<publisherCount {
                group.addTask {
                    for n in 0..<eventsPerPublisher {
                        await bus.publish(EngineEvent(
                            kind: .downloadProgress, module: "touchup",
                            modelId: "p\(publisher)-e\(n)", progress: 0.5
                        ))
                    }
                }
            }
        }

        let received = await consumer.value
        XCTAssertEqual(received.count, total)
        XCTAssertEqual(
            Set(received).count, total,
            "every published event must arrive exactly once (no duplicates)"
        )
    }

    /// The per-subscriber buffer is bounded (PR #239 review): a subscriber
    /// that never consumes cannot grow engine memory without limit — once
    /// the buffer is full, older events drop and only the newest
    /// `subscriberBufferLimit` remain.
    func testStalledSubscriberBufferIsBoundedToNewest() async {
        let bus = EngineEventBus()
        let stream = await bus.subscribe()

        let overfill = EngineEventBus.subscriberBufferLimit + 50
        for n in 0..<overfill {
            await bus.publish(EngineEvent(
                kind: .downloadProgress, module: "touchup",
                modelId: "e\(n)", progress: 0.5
            ))
        }

        // Drain whatever was buffered; the stream stays open, so stop
        // reading once we've seen the newest event.
        var received: [String] = []
        for await event in stream {
            if let id = event.modelId { received.append(id) }
            if event.modelId == "e\(overfill - 1)" { break }
        }

        XCTAssertEqual(received.count, EngineEventBus.subscriberBufferLimit)
        XCTAssertEqual(
            received.first, "e\(overfill - EngineEventBus.subscriberBufferLimit)",
            "bufferingNewest must drop the OLDEST events when full"
        )
        XCTAssertEqual(received.last, "e\(overfill - 1)")
    }
}
