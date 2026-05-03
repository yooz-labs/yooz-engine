# Contributing to Yooz Engine

Thanks for considering a contribution. The engine is the heart of the Yooz ecosystem; PRs that improve correctness, performance, or developer ergonomics are welcome.

## Before you start

- **License agreement**: this repository is licensed under [PolyForm Shield 1.0.0](LICENSE.md). By contributing, you agree your contribution is provided under the same license. The strategic rationale lives in [`LICENSING.md`](LICENSING.md).
- **DCO sign-off** (required): every commit must carry a `Signed-off-by:` trailer. This is a lightweight statement that you have the right to submit the contribution under the project's license. No CLA required.

  ```bash
  git commit -s -m "feat: add new module health endpoint"
  ```

  The `-s` flag adds a line like `Signed-off-by: Your Name <you@example.com>` derived from `git config user.name` and `user.email`.

- **Discuss first** for non-trivial changes. Open an issue describing the problem and the proposed approach before writing a large patch — saves everyone time. Small fixes (typos, obvious bugs, perf wins on a single function) are fine to send directly.

## Workflow

1. **Open an issue** describing the bug or feature (skip for trivial fixes).
2. **Branch from `main`** (or the relevant epic branch if one is in flight): `git checkout -b feature/issue-N-short-description`.
3. **Make atomic commits** with concise messages (under 50 chars on the subject line, no AI attribution).
4. **Write tests**. We use XCTest. Run the full suite before pushing:

   ```bash
   xcodegen generate
   xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
     -configuration Debug -skipMacroValidation \
     -destination 'platform=macOS' test
   ```

   `-skipMacroValidation` is required so the `MLXHuggingFaceMacros` trust prompt doesn't hang the headless invocation.
5. **Run lint**: SwiftLint locally (`swiftlint lint`) and Spell Check (`typos` if you have it). CI runs both on every PR.
6. **Open a PR** against `main` (or the epic branch). Describe what changed, why, and how to test it. Reference the issue number with `Closes #N`.
7. **Address review findings**. Maintainers run an automated multi-agent review on every PR before merge (you don't need to run it yourself); plus human review. Address every finding that isn't a genuine false positive; for skips, explain why in a PR comment.
8. **Merge after CI green**. Don't merge with red CI. Maintainers handle the merge.

## Commit style

```
fix(#43): variant-aware eager-load for module readiness

- New EngineVariant enum + compile-time gate.
- ModuleEagerLoader actor primes modules after server starts.
- /v1/health adds a `loading` state for in-flight loads.

Signed-off-by: Your Name <you@example.com>
```

- Subject: imperative, present tense, under 50 chars, optional `type(#issue):` prefix.
- Body: what + why, not how. The diff shows how.
- No emojis, no "Built with Claude", no AI attribution.

## What not to commit

- Secrets (`.env`, API keys, signed builds).
- Large binaries — use Git LFS or HuggingFace if you must version a model artifact.
- Auto-generated files unless they're explicitly tracked (`xcodegen generate` regenerates `YoozEngine.xcodeproj`).
- Personal IDE config that doesn't fit the team setup.

## Tests

- **Unit tests**: XCTest under `Tests/`. Run before pushing.
- **No mocks of internal modules**. Test against the real APIs where possible. Mock only at system boundaries (network, file system if needed).
- **Coverage goal**: every public method exposed via `/v1/*` has at least a wire-format and a happy-path test.

## Code style

- Swift 6 concurrency: use `@MainActor` on UI/state-holder classes.
- Singletons (Grammar, VAD): actor singletons; require `load()` before use where applicable.
- Don't add error handling for impossible scenarios. Trust your function contracts.
- Don't add comments that explain WHAT the code does (the names should). Comment only the WHY when it's non-obvious.

## Security

Found a vulnerability? See [`SECURITY.md`](SECURITY.md) — please don't open a public issue.

## Questions

Open an issue, or email **dev@yooz.info**. Welcome aboard.
