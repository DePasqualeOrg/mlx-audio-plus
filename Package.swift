// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "MLXAudio",
            targets: ["MLXAudio"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.0")),
        .package(url: "https://github.com/espeak-ng/espeak-ng-spm.git", branch: "master"),
    ],
    targets: [
        .target(
            name: "MLXAudio",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "libespeak-ng", package: "espeak-ng-spm"),
                .product(name: "espeak-ng-data", package: "espeak-ng-spm"),
            ],
            path: "mlx_audio_swift/tts/MLXAudio",
            exclude: ["TTS/Kokoro/Preview Content"],
            resources: [
                .process("TTS/Kokoro/Resources")  // Kokoro voices
            ]
        ),
        .testTarget(
            name: "MLXAudioTests",
            dependencies: ["MLXAudio"],
            path: "mlx_audio_swift/tts/Tests"
        ),
    ]
)
