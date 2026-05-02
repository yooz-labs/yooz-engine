// swift-tools-version: 5.9

import PackageDescription

// Two products live in this root package:
//
//  - `YoozEngineClient`: the thin-client SDK consumed by Yooz apps
//    (Whisper, Notes, etc.) that talk to the engine over HTTP/WS.
//
//  - `Qwen3ASRMelFrontend`: the Phase 2 native Swift mel-spectrogram
//    frontend for Qwen3-ASR. The library lives under
//    `YoozEngine/STT/Models/Qwen3ASR/MelFrontend/` so the Xcode
//    `YoozEngine` app target compiles the exact same source files as
//    part of the production build (the `project.yml` `sources:` glob
//    already picks them up). Re-exposing it here as a SwiftPM target
//    lets us run the parity gate via `swift test` without dragging
//    the Hummingbird/MLX-heavy macOS app into the test fixture (the
//    host-app TEST_HOST path is incompatible with the menu-bar
//    `LSUIElement` setting that keeps the engine alive in
//    production).
let package = Package(
    name: "YoozEnginePackages",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "YoozEngineClient", targets: ["YoozEngineClient"]),
        .library(
            name: "Qwen3ASRMelFrontend",
            targets: ["Qwen3ASRMelFrontend"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "YoozEngineClient",
            path: "Sources/YoozEngineClient"
        ),
        .target(
            name: "Qwen3ASRMelFrontend",
            path: "YoozEngine/STT/Models/Qwen3ASR/MelFrontend"
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
    ]
)
