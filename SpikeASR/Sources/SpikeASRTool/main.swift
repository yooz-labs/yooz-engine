import Foundation
import MLX
import SpikeASR

// Phase 1 spike CLI — loads the audio_tower from the cached
// Qwen3-ASR-1.7B-8bit checkpoint, runs a forward pass on the
// reference parity inputs, and writes the encoder hidden states
// to a safetensors file under the spike artifacts directory so
// the parity test (and a human) can compare against the Python
// reference.
//
// Usage:
//   spike-asr smoke
//   spike-asr forward
//
// Both modes require the Phase 1 artifacts on /Volumes/S1.

let weightsURL = URL(
    fileURLWithPath:
        "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
        + "audio_tower_bf16.safetensors")
let parityInputsURL = URL(
    fileURLWithPath:
        "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
        + "parity_inputs.safetensors")
let parityOutputsURL = URL(
    fileURLWithPath:
        "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
        + "swift_encoder_outputs.safetensors")

func runSmoke() throws {
    let cfg = AudioEncoderConfig()
    let encoder = AudioEncoder(cfg)
    try SpikeLoader.loadAudioTower(from: weightsURL, into: encoder)
    let synthetic = MLXArray.zeros([1, cfg.numMelBins, 100], dtype: .float32)
    let mask = MLXArray(
        Array(repeating: Int32(1), count: 100)
    ).reshaped(1, 100)
    let out = encoder(
        inputFeatures: synthetic, featureAttentionMask: mask
    )
    eval(out)
    print(
        "[smoke] output shape =", out.shape,
        "dtype =", out.dtype
    )
}

func runForward() throws {
    let cfg = AudioEncoderConfig()
    let encoder = AudioEncoder(cfg)
    try SpikeLoader.loadAudioTower(from: weightsURL, into: encoder)
    let inputs = try MLX.loadArrays(url: parityInputsURL)
    guard let features = inputs["input_features"] else {
        FileHandle.standardError.write(
            Data("missing input_features\n".utf8)
        )
        exit(2)
    }
    let mask = inputs["feature_attention_mask"]
    let output = encoder(inputFeatures: features, featureAttentionMask: mask)
    eval(output)
    try MLX.save(arrays: ["encoder_hidden_states": output], url: parityOutputsURL)
    print(
        "[forward] output shape =", output.shape,
        "dtype =", output.dtype,
        "wrote", parityOutputsURL.path
    )
}

let argv = CommandLine.arguments.dropFirst()
let mode = argv.first ?? "forward"
do {
    switch mode {
    case "smoke": try runSmoke()
    case "forward": try runForward()
    default:
        FileHandle.standardError.write(
            Data("unknown mode: \(mode)\n".utf8)
        )
        exit(64)
    }
} catch {
    FileHandle.standardError.write(
        Data("error: \(error)\n".utf8)
    )
    exit(1)
}
