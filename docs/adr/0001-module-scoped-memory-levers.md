# ADR 0001 — Memory levers are module-scoped and resource-specific

**Status:** Accepted
**Date:** 2026-07-25
**Issues:** engine#299 (LLM prompt-cache drop), engine#291 (STT download/cancel), engine#292/#293 (per-row download progress)
**Consumers affected:** yooz-whisper, yooz-notes, yooz-crisp, remi, super-yooz

## Context

The engine holds multi-GB model weights on behalf of consumer apps that do not
control each other. Today that is whisper and remi; the stated direction is
super-yooz, where several apps' modules share **one** engine process precisely
so the weights are not duplicated.

That makes memory a shared resource with no single owner, and the API is what
arbitrates it. Until now the engine offered consumers exactly two memory
levers:

- `POST /v1/llm/unload` — frees a model's weights. Effective, but the next
  request pays a multi-second cold reload.
- `POST /v1/session/begin` — fans `resetForNewSession()` out to **every**
  `SessionResettable` module. It exists to mean "a new recording session
  started", and for the LLM it happens to reach
  `MLXLLMBackend.clearSession()`, which drops the cached prompt KV state.

So a consumer wanting to reclaim cache memory had to choose between unloading
the model or resetting every module in the engine — including modules
belonging to a different app. With one consumer that is untidy. With several
sharing a host it is incorrect: one app reclaiming memory would silently clear
another app's session state.

Measured, on an M4 Pro serving `YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx` for
remi's auto-approve workload (~1K-token prompts, 38 sequential evaluations,
reasoning disabled):

| | physical footprint |
|---|---|
| after model load | 2.9 GB |
| after 38 evaluations | 4.4 GB (plateaus) |

Two resources with materially different economics are hiding behind one lever:

- **Weights** — ~2.9 GB, expensive to restore (seconds), worth keeping resident
  through short idle periods.
- **Prompt KV cache** — ~1.5 GB, cheap to recompute (sub-second), and a
  genuine win while active. remi sends an identical system prompt on every
  evaluation, so the cache hit is a large part of why p50 lands near 1.0 s.

Collapsing them forces every consumer into a false choice: hold 4.4 GB, or
unload and pay a cold load. On a 16 GB laptop running an app suite, neither is
the right answer for an idle module.

## Decision

**Lifecycle and memory operations are scoped to the module and to the specific
resource they affect. Engine-global fan-outs are reserved for genuinely global
events.**

Concretely:

1. Each module exposes its own memory levers under its own path. engine#299
   adds `POST /v1/llm/clear-cache`, which drops the LLM prompt KV cache while
   leaving weights resident — the middle lever between "keep everything" and
   "unload".
2. `/v1/session/begin` keeps its meaning — "a new recording session started" —
   and is not the mechanism by which consumers manage memory. Its global
   fan-out is correct for what it means and wrong for what it was being
   borrowed for.
3. Distinct resources get distinct verbs. "Drop the cache" and "unload the
   weights" are different operations with different costs; a consumer must be
   able to ask for either.
4. New stateful modules ship their own scoped levers rather than relying on
   the global reset to reach them.

## Consequences

**Consumers can implement graduated idle policies.** The shape remi is
building (drop the cache at a short idle, unload the weights only after a long
one) becomes available to every app, not just remi. Whisper's touch-up tier
has the same profile — a large resident model, bursty use, long idle gaps — and
gets the same benefit with no engine change beyond this one.

**super-yooz gets something it cannot function without.** A host arbitrating
memory across several apps' modules needs to reclaim from an idle module
without disturbing an active one. Global resets make that impossible by
construction; module-scoped levers make it a policy decision the host can
actually make.

**A cost we accept: more API surface.** Each module gains verbs, and each verb
is a compatibility commitment. The alternative — one general "reclaim memory"
call — cannot express which module or which resource, which is exactly the
problem being fixed.

**Consumers must degrade when a lever is absent.** The engine and the apps ship
on separate cycles, so a consumer will routinely run against an engine older
than the lever it wants. Absence must be a silent, logged-once fallback to the
coarser lever, never a failure — remi's implementation treats a 404 from
`clear-cache` as "fall back to unload-only".

**This generalizes beyond memory.** The same reasoning already applies to
downloads (engine#291 extended explicit download/cancel to STT rather than
overloading a selection change) and to observability (engine#293 added a
per-row `downloadProgress` because a global event stream could not answer "how
is THIS model doing"). The recurring lesson: as soon as a second consumer
exists, engine-global operations become other people's side effects.

## Alternatives considered

**Keep using `/v1/session/begin`.** Zero new API. Rejected: it means something
else, and using it for memory reclaim makes one consumer's housekeeping into
every other module's state loss. That is a latent bug that only appears once
two apps share a host — the exact configuration super-yooz is built for.

**A flag on unload (`{keepWeights: true}`).** Smaller surface than a new route
and genuinely reasonable. Rejected mainly on clarity: "unload but do not
unload" reads as a contradiction at the call site, and the two operations have
different failure and idempotency semantics. Worth revisiting if the verb count
per module becomes a real burden.

**Let the engine decide on its own (internal idle eviction).** Attractive
because it needs no consumer cooperation. Rejected for now: the engine cannot
see a consumer's intent — remi's daemon may sit idle for an hour and then need
a verdict in under a second, while a whisper user may be mid-dictation with
long pauses. Consumers know their own latency budgets; the engine should give
them the levers rather than guess. Revisit if consumers converge on the same
policy anyway, at which point a default engine policy becomes the simpler
answer.
