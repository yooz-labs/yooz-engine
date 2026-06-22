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
    ],
    dependencies: [
        // mlx-swift 0.31.4 release (what mlx-swift-lm main declares). Pin the
        // release, not mlx-swift main, which is on the unreleased swift-tools 6.3
        // toolchain. Kept in lockstep with project.yml.
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.31.4")
        ),
        // Phase 4: Qwen3 text decoder + tokenizer-aware chat template
        // are reused from mlx-swift-lm + swift-transformers. The
        // engine Xcode project already pulls these in via project.yml;
        // SwiftPM needs them too so the headless parity tests can
        // exercise the full bridge. Kept in lockstep with project.yml: our clean
        // ml-explore fork (yooz-labs/mlx-swift-lm) main, = ml-explore main + the
        // gemma4_unified `vision_embedder` fix + the MLXLLM Gemma4 MoE/KV-sharing
        // port, both with upstream PRs in flight (yooz-engine#187).
        .package(
            url: "https://github.com/yooz-labs/mlx-swift-lm",
            revision: "f4fd39e7037574169bbb521f3724b0f71512b498"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.1.6"
        ),
    ],
    targets: [
        .target(
            name: "YoozEngineClient",
            path: "Sources/YoozEngineClient"
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
