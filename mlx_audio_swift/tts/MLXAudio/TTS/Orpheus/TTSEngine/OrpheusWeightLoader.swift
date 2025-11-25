//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import Hub

// Utility class for loading and preprocessing the weights for the model
public class OrpheusWeightLoader {
    private init() {}

    static public let defaultRepoId = "mlx-community/orpheus-3b-0.1-ft-4bit"
    static let defaultWeightsFilename = "model.safetensors"

    static func loadWeights(
        repoId: String = defaultRepoId,
        filename: String = defaultWeightsFilename,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> [String: MLXArray] {
        let modelDirectoryURL = try await Hub.snapshot(
            from: repoId,
            matching: [filename],
            progressHandler: progressHandler
        )
        let weightFileURL = modelDirectoryURL.appending(path: filename)
        return try loadWeights(from: weightFileURL)
    }

    static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        let weights = try MLX.loadArrays(url: url)
        var processedWeights: [String: MLXArray] = [:]

        let groupSize = 64
        for (key, value) in weights {
            if key.hasSuffix(".weight") {
                // Detect quantized weight by dtype uint32
                if value.dtype == .uint32 {
                    // Look for associated scales and biases
                    let scaleKey = key.replacingOccurrences(of: ".weight", with: ".scales")
                    let biasKey = key.replacingOccurrences(of: ".weight", with: ".biases")
                    if let scales = weights[scaleKey], let biases = weights[biasKey] {
                        let deq = Dequantizer.dequantize(value, scales: scales, biases: biases, groupSize: groupSize, bits: 4)
                        processedWeights[key] = deq

                    } else {
                        Log.model.warning("Missing scales/biases for quantized weight \(key). Loading raw.")
                        processedWeights[key] = value
                    }
                } else {
                    processedWeights[key] = value
                }
            } else {
                // Non-weight tensors keep original
                processedWeights[key] = value
            }
        }

        return processedWeights
    }
}
