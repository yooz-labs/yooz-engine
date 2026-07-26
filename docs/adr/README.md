# Architecture Decision Records

Decisions that shape the engine's contract with its consumers, and that would
otherwise be re-litigated every time a new app integrates.

Kept in `docs/` rather than `.context/` deliberately: `.context/` is gitignored
and local to one machine, and these records exist to be read by the people
building against this engine.

An ADR belongs here when the decision is **hard to reverse** or **binds
consumers** — API shape, ownership boundaries, compatibility rules. Routine
implementation choices belong in the issue or the code comment where they were
made, not here.

Format: Context (with evidence, ideally measured), Decision, Consequences
including the costs accepted, and Alternatives considered with the reason each
was rejected. A future reader must be able to tell whether the reasoning still
holds when the facts change.

| # | Title | Status |
|---|---|---|
| [0001](0001-module-scoped-memory-levers.md) | Memory levers are module-scoped and resource-specific | Accepted |
