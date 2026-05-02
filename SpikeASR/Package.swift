// swift-tools-version: 5.9

import PackageDescription

// SpikeASR — Phase 1 spike for issue #47.
// Standalone SwiftPM package proving an MLX-Swift-only port of the
// Qwen3-ASR-1.7B audio encoder is feasible end-to-end. Pinned to the
// same mlx-swift release the YoozEngine app uses (project.yml:
// mlx-swift from 0.21.2).
//
// Structure:
//   - SpikeASR        : library; encoder model + safetensors loader.
//   - SpikeASRTool    : CLI for running the loader against the
//                       checkpoint on /Volumes/S1; not a ship target.
//   - SpikeASRTests   : unit + parity tests. Parity tests xfail-skip
//                       gracefully if the S1 artifacts are absent
//                       (CI-friendly).
let package = Package(
    name: "SpikeASR",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SpikeASR", targets: ["SpikeASR"]),
        .executable(name: "spike-asr", targets: ["SpikeASRTool"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.2"),
    ],
    targets: [
        .target(
            name: "SpikeASR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/SpikeASR"
        ),
        .executableTarget(
            name: "SpikeASRTool",
            dependencies: [
                "SpikeASR",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SpikeASRTool"
        ),
        .testTarget(
            name: "SpikeASRTests",
            dependencies: [
                "SpikeASR",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/SpikeASRTests"
        ),
    ]
)
