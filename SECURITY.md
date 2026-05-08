# Security Policy

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email **dev@yooz.info** with:

- A clear description of the issue.
- A minimal reproduction (steps, code, network requests, etc.) if you have one.
- Affected version(s) — engine version from `/v1/health`, your macOS version, and any consumer-app version (Whisper, Notes, etc.) if relevant.
- Your name and contact for follow-up. We're happy to credit you in the fix announcement if you'd like.

We aim to acknowledge within **2 business days** and provide a triage decision (accepted / needs more info / not a security issue) within **5 business days**.

## Scope

In scope:

- The engine HTTP/WebSocket server (`localhost:19920`): authentication bypass, request smuggling, path traversal, unintended data exposure across modules.
- Model loading paths: arbitrary file read/write via crafted model paths or HuggingFace URLs, deserialization issues in safetensors / MLX checkpoint parsing.
- The `YoozEngineClient` Swift Package: authentication / session-handling bugs that could leak a user's local engine port to a remote.
- `text-cleanup` Rust grammar engine: memory safety, RCE via crafted XML rule files.
- The auto-launch mechanism for the menu-bar app: privilege escalation, arbitrary code execution at login.

Out of scope (please don't report these):

- Vulnerabilities in third-party dependencies (`mlx-swift`, `Hummingbird`, etc.) where the upstream project is the right place to report. We track upstream security advisories and update.
- Issues that require physical access to the user's unlocked Mac.
- DoS via local processes that already have user-level privileges (the user can already kill the engine).
- Self-XSS or social-engineering scenarios that require the user to actively cooperate with the attacker.
- Findings on test fixtures, sample data, or development-only code paths.

## Disclosure timeline

We follow **coordinated disclosure**:

1. You report → we acknowledge within 2 business days.
2. We triage and confirm within 5 business days.
3. We develop + test a fix. Standard fix window: **30 days** for critical / high, **60 days** for medium, **90 days** for low.
4. We coordinate the disclosure date with you. Default: full public disclosure with credit to the reporter when the fix ships.
5. We may request you withhold public disclosure until the fix is shipped. We will not silently delay; we'll communicate the timeline.

## Supported versions

Yooz Engine is in active development; we support the **latest minor release** with security fixes.

| Version | Supported |
|---|---|
| 0.6.x (Phase 5: modular engine + thin client) | yes |
| 0.5.x (Phase 4.5: engine sync) | best effort |
| < 0.5 | no |

If you need security backports for older versions in production, contact **dev@yooz.info** to discuss.

## Hall of Fame

We'll list reporters with their permission once we've shipped fixes.

---

For non-security questions or general issues, please use **GitHub Issues**.
