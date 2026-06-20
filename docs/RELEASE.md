# Yooz Engine Release Runbook

Local-only release pipeline for the signed `.app` bundles. GitHub Actions does
not have macOS runners for this repository (see
[engine#23](https://github.com/yooz-labs/yooz-engine/issues/23)); the Release
Notes workflow only validates the tag and creates a draft GitHub release,
then a maintainer uploads artifacts built on their own machine.

Scope: full engine (`YoozEngine.app`), lite variant (`YoozEngineLite.app`),
and the whisper helper (`YoozEngineWhisper.app`). The full engine includes
Infinite long-context endpoints; Lite and Whisper intentionally do not.
Notarization is explicitly
deferred in Phase 5 (see `.context/phase5_epic.md`); Gatekeeper will warn on
first launch for any non-notarized artifact.

## Prerequisites

- macOS with Xcode 16 or newer
- `xcodegen` (`brew install xcodegen`)
- `gh` CLI authenticated against `yooz-labs`
- Metal toolchain installed (matches Xcode; required for MLX)
- A `Developer ID Application` cert in the login keychain
  - If absent, scripts fall back to ad-hoc signing with a loud warning and
    Gatekeeper will reject the bundle outside of developer mode

Optional environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `YOOZ_ENGINE_CONFIG` | Build configuration | `Debug` (see #38) |
| `YOOZ_SIGNING_IDENTITY` | Pin a specific cert hash | auto-detect |
| `YOOZ_HELPER_CONFIG` | Whisper-helper build config | `Debug` |

## One-shot release

```bash
# 1. Bump the version in Sources/EngineCore/EngineConfig.swift
#    and YoozEngine/Info.plist (CFBundleShortVersionString) to match.
#    Commit the bump.
git checkout -b release/vX.Y.Z
# edit Sources/EngineCore/EngineConfig.swift (`public static let version = "X.Y.Z"`)
# edit YoozEngine/Info.plist (CFBundleShortVersionString)
git commit -am "chore: bump version to X.Y.Z"

# 2. Build + sign the three artifacts locally.
bash scripts/release-engine.sh

# 3. Smoke-test. All three .apps must print ALIVE.
bash scripts/smoke-test-release.sh

# 4. Tag + push. The release-notes.yml workflow creates a draft release.
git tag vX.Y.Z
git push origin release/vX.Y.Z vX.Y.Z

# 5. Upload the artifacts + manifest to the draft.
gh release upload vX.Y.Z \
    dist/YoozEngine.app.zip \
    dist/YoozEngineLite.app.zip \
    dist/YoozEngineWhisper.app.zip \
    dist/RELEASE.md

# 6. Edit the draft release body: paste dist/RELEASE.md contents and publish.
gh release edit vX.Y.Z --notes-file dist/RELEASE.md
gh release edit vX.Y.Z --draft=false
```

## What each script does

- `scripts/build-engine-release.sh` — builds `dist/YoozEngine.app` (full
  variant: STT + LLM + VAD + Grammar + AppleSTT + Infinite), signs it, runs
  `codesign --verify --deep --strict`.
- `scripts/build-engine-lite.sh` — same for `dist/YoozEngineLite.app`
  (Apple STT only; sub-GB bundle, no MLX).
- `scripts/build-whisper-helper.sh` — A5 script; builds
  `dist/YoozEngineWhisper.app` for embedding into `Yooz Whisper.app`.
- `scripts/release-engine.sh` — orchestrator; runs all three builds,
  zips each bundle with `ditto` (preserves codesign), computes SHA256 of
  binary and zip for every variant, and writes `dist/RELEASE.md`.
- `scripts/smoke-test-release.sh` — launches each built `.app` serially
  (port 19920 is shared), probes `/v1/health`, kills and waits for the
  port to free before the next launch. Exits non-zero on any failure.

## Verifying a release artifact

Anyone who downloaded a `.app.zip` from a GitHub release can confirm it:

```bash
unzip -q YoozEngine.app.zip
codesign --verify --deep --strict --verbose=2 YoozEngine.app
shasum -a 256 "YoozEngine.app/Contents/MacOS/Yooz Engine"
# Compare the SHA256 against the value listed in RELEASE.md.
```

For the whisper helper the command is the same with bundle name
`YoozEngineWhisper.app` and binary `Yooz Engine (Whisper)`.

For Infinite contract coverage before cutting a full-engine release, run:

```bash
swift test --filter InfiniteTypesTests
xcodegen generate
xcodebuild -project YoozEngine.xcodeproj -scheme YoozEngine \
  -configuration Debug -skipMacroValidation \
  -derivedDataPath .build/DerivedData \
  build-for-testing
scripts/run-integration.sh
```

The integration suite drives the served app through `YoozEngineClient` and
checks `/v1/modules`, `/v1/infinite/models`, `/v1/infinite/status`, session
create/append/fetch/checkpoint/delete, expected `501 generation_unavailable`,
and deleted-session `404`.

## Known limitations (A6)

- **No notarization.** First launch shows "Apple could not verify this app
  is free of malware." Right-click + Open to bypass during testing.
  Notarization will land in a later phase.
- **No macOS CI build.** Artifacts are maintainer-built locally. The
  `release-notes.yml` workflow only creates a draft release and reminds
  the maintainer to upload.
- **Debug default.** MLX transitive embeds fail in Release (see #38);
  `YOOZ_ENGINE_CONFIG=Release` can be tried once that lands, until then
  the scripts warn loudly and default to Debug.
- **Shared port.** Smoke test launches are serial — the three variants
  all bind port 19920, so parallel smoke-testing is not possible.
- **Infinite full-variant only.** `/v1/infinite/*` is available in
  `YoozEngine.app`; Lite and Whisper return module-not-bundled `501` by
  design. Full-tier Infinite rows require 64 GiB+ Apple Silicon.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `xcodegen not on PATH` | `brew install xcodegen` |
| `codesign verify failed` | Check `.build/<variant>-build.log`; common cause is a stale `dist/<variant>.app` from a previous run — re-run the build |
| `smoke-test: port 19920 held at startup` | Another engine is running; `pkill -f "Yooz Engine"` and re-run |
| `ad-hoc signing` warning | Install a Developer ID Application cert into the login keychain |
| `ditto zip failed` | Ensure no Finder / Xcode process has a handle on the bundle |

## Cross-references

- Epic tracker: [engine#24](https://github.com/yooz-labs/yooz-engine/issues/24)
- Infinite epic: [engine#160](https://github.com/yooz-labs/yooz-engine/issues/160)
- Infinite API: [INFINITE_MODULE.md](INFINITE_MODULE.md)
- A5 (whisper helper): [engine#29](https://github.com/yooz-labs/yooz-engine/issues/29)
- A6 (this pipeline): [engine#30](https://github.com/yooz-labs/yooz-engine/issues/30)
- CI macOS constraint: [engine#23](https://github.com/yooz-labs/yooz-engine/issues/23)
- MLX Release embed bug: [engine#38](https://github.com/yooz-labs/yooz-engine/issues/38)
