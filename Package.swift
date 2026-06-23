// swift-tools-version: 5.9
// swiftlint:disable line_length

import PackageDescription

// Root SwiftPM package for the engine repository.
//
// Three products:
//
//   - `YoozEngineClient`: the thin-client SDK consumed by Yooz apps
//     (Whisper, Notes, etc.) that talk to the engine over HTTP/WS.
//
//   - `Qwen3ASRMelFrontend`: the Phase 2 native Swift mel-spectrogram
//     frontend. Lives under
//     `YoozEngine/STT/Models/Qwen3ASR/MelFrontend/`. Kept as its own
//     SwiftPM target so the Phase 2 mel parity tests keep running
//     unchanged after Phase 3 lands the broader `Qwen3ASR` umbrella.
//
//   - `Qwen3ASR`: the Phase 3 Qwen3-ASR audio encoder library. Lives
//     under `YoozEngine/STT/Models/Qwen3ASR/` (the parent of the mel
//     frontend) and re-exports `Qwen3ASRMelFrontend`. The path
//     excludes `MelFrontend/` so the two SwiftPM targets do not
//     overlap on disk; symbolically the encoder reaches the mel
//     frontend through the SwiftPM target dependency.
//
// Both `Qwen3ASR*` libraries compile from the exact same source
// files that the Xcode `YoozEngine` app target ingests via xcodegen
// (`project.yml` `sources:` glob), so SwiftPM and xcodegen share a
// single source of truth.
//
// Why expose `Qwen3ASR` via SwiftPM at all? The heavy parity tests
// (606 MB safetensors load) deadlock under `xcodebuild test`: the
// Xcode test host is a GUI app subject to TCC / Launch Services
// gating that never resolves headlessly. Running the same tests via
// `swift test` produces a plain CLI binary that bypasses the GUI
// gating. The xcodebuild flow still runs the structural / unit
// tests; the parity suite runs from SwiftPM.
let package = Package(
    name: "YoozEngineRepo",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "YoozEngineClient", targets: ["YoozEngineClient"]),
        .library(
            name: "Qwen3ASRMelFrontend",
            targets: ["Qwen3ASRMelFrontend"]
        ),
        .library(name: "Qwen3ASR", targets: ["Qwen3ASR"]),

        // Engine modules exposed for in-process linking by standalone App Store
        // apps (epic #192). Same source dirs the xcodegen module-framework targets
        // (STTModule, LLMModule, …) ingest via project.yml — the app targets
        // (YoozEngine*) exclude those dirs and just link the frameworks — so
        // SwiftPM and xcodegen share one source of truth.
        .library(name: "EngineCore", targets: ["EngineCore"]),
        .library(name: "AppleSTTModule", targets: ["AppleSTTModule"]),
        .library(name: "VADModule", targets: ["VADModule"]),
        .library(name: "LLMModule", targets: ["LLMModule"]),
        .library(name: "GrammarModule", targets: ["GrammarModule"]),
        .library(name: "STTModule", targets: ["STTModule"]),
    ],
    dependencies: [
        // mlx-swift 0.31.4 release (what mlx-swift-lm main declares). Pin the
        // release, not mlx-swift main, which is on the unreleased swift-tools 6.3
        // toolchain. Kept in lockstep with project.yml.
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            exact: "0.31.4"
        ),
        // Phase 4: Qwen3 text decoder + tokenizer-aware chat template
        // are reused from mlx-swift-lm + swift-transformers. The
        // engine Xcode project already pulls these in via project.yml;
        // SwiftPM needs them too so the headless parity tests can
        // exercise the full bridge. Kept in lockstep with project.yml: our clean
        // ml-explore fork (yooz-labs/mlx-swift-lm) main = ml-explore main + the
        // gemma4_unified `vision_embedder` fix (PR #363) + the MLXLLM Gemma4 MoE
        // port (PR #364); E-series KV-sharing rides upstream #342.
        .package(
            url: "https://github.com/yooz-labs/mlx-swift-lm",
            revision: "0f1865ad8d44824c5ad36ba1e9e4203d74838eb6"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            "1.2.0" ..< "1.3.0"
        ),
        // swift-huggingface (HubClient) — used by the #huggingFaceTokenizerLoader
        // macro that LLM/Infinite expand. Matches project.yml.
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            "0.9.0" ..< "0.10.0"
        ),
    ],
    targets: [
        .target(
            name: "YoozEngineClient",
            path: "Sources/YoozEngineClient"
        ),

        // MARK: - Engine modules (in-process, epic #192)
        .target(
            name: "EngineCore",
            path: "Sources/EngineCore"
        ),
        .target(
            name: "AppleSTTModule",
            dependencies: ["EngineCore"],
            path: "YoozEngine/AppleSTT"
        ),
        .target(
            name: "VADModule",
            dependencies: ["EngineCore"],
            path: "YoozEngine/VAD"
        ),
        // LLM + TouchUp are mutually referential (one module in xcodegen), so a
        // single SPM target spans both dirs. The #huggingFaceTokenizerLoader
        // macro builds natively under SwiftPM (the xcodegen-only
        // -skipMacroValidation flag is an xcodebuild flag, not a swiftc one, and
        // isn't needed here), so this stays a usable versioned dependency.
        .target(
            name: "LLMModule",
            dependencies: [
                "EngineCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "YoozEngine",
            sources: ["LLM", "TouchUp"]
        ),
        // Grammar depends on the vendored Rust text-cleanup xcframework. The whole
        // Vendor/ tree is gitignored (build output, regenerated from the Rust
        // source), so on a clean checkout this path is absent and `swift build`
        // fails until the xcframework is built. For external consumers it must
        // become a remote .binaryTarget(url:checksum:) (publish like the helper
        // release) — tracked as a follow-up. Local path proves the in-repo integration.
        .target(
            name: "GrammarModule",
            dependencies: ["EngineCore", "YoozTextCleanup"],
            path: "YoozEngine/Grammar"
        ),
        .binaryTarget(
            name: "YoozTextCleanup",
            path: "Vendor/YoozTextCleanup/YoozTextCleanup.xcframework"
        ),
        // STTModule is the whole YoozEngine/STT EXCEPT Models/Qwen3ASR, which
        // stays its own SPM target so the 606 MB parity tests can run headlessly
        // via `swift test` (see the file-header comment for why xcodebuild
        // deadlocks). The STT engine references Qwen3ASR types (public,
        // one-directional), bridged by a `#if canImport(Qwen3ASR)` import in the
        // few referencing files.
        .target(
            name: "STTModule",
            dependencies: [
                "EngineCore",
                "Qwen3ASR",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "YoozEngine/STT",
            exclude: ["Models/Qwen3ASR"]
        ),
        .target(
            name: "Qwen3ASRMelFrontend",
            path: "YoozEngine/STT/Models/Qwen3ASR/MelFrontend"
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "Qwen3ASRMelFrontend",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "YoozEngine/STT/Models/Qwen3ASR",
            exclude: ["MelFrontend"]
        ),
        .testTarget(
            name: "YoozEngineClientTests",
            dependencies: ["YoozEngineClient"],
            path: "Tests/YoozEngineClientTests"
        ),
        .testTarget(
            name: "Qwen3ASRMelFrontendTests",
            dependencies: ["Qwen3ASRMelFrontend"],
            path: "Tests/Qwen3ASRMelFrontendTests"
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: [
                "Qwen3ASR",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Tests/YoozEngineTests/Qwen3ASR"
        ),
    ]
)
