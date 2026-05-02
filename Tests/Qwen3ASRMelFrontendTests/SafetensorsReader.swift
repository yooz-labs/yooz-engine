import Foundation

/// Minimal read-only safetensors loader for the parity fixtures.
///
/// The Phase 2 parity dumps (`dump_mel.py`) use `safetensors.numpy.save_file`
/// with a small set of plain tensor types: `F32` for audio + features +
/// mel filter weights, `I32` for the attention mask. A full safetensors
/// reader would pull in `mlx-swift`, but that package's runtime
/// requires a Metal `.metallib` resource that `swift test` (outside an
/// `.app` bundle) cannot resolve. Reimplementing the trivial subset
/// of the format we actually need keeps the test target Apple-toolchain-only.
///
/// Format reference: https://github.com/huggingface/safetensors
enum SafetensorsReader {

    enum Error: Swift.Error, CustomStringConvertible {
        case fileNotFound(String)
        case unreadableHeader(String)
        case unsupportedDType(String)
        case missingTensor(String)
        case shapeMismatch(name: String, expected: [Int], actual: [Int])
        case truncatedTensorData(name: String)

        var description: String {
            switch self {
            case .fileNotFound(let p):
                return "safetensors fixture not found at \(p)"
            case .unreadableHeader(let m):
                return "safetensors header is malformed: \(m)"
            case .unsupportedDType(let dt):
                return "unsupported safetensors dtype: \(dt)"
            case .missingTensor(let n):
                return "safetensors tensor '\(n)' missing in fixture"
            case .shapeMismatch(let n, let e, let a):
                return
                    "safetensors tensor '\(n)' shape mismatch:"
                    + " expected \(e), got \(a)"
            case .truncatedTensorData(let n):
                return "safetensors tensor '\(n)' data is truncated"
            }
        }
    }

    /// Per-tensor metadata pulled from the JSON header.
    struct TensorEntry {
        let dtype: String
        let shape: [Int]
        let dataOffset: Int  // absolute offset in file
        let dataLength: Int
    }

    /// Loaded safetensors blob: header table + raw payload (so callers
    /// can slice tensors out without re-reading the file).
    struct Bundle {
        let entries: [String: TensorEntry]
        let payload: Data

        /// Read a `Float32` tensor by name, validating shape if given.
        func float32(_ name: String, expectedShape: [Int]? = nil) throws
            -> (values: [Float], shape: [Int])
        {
            guard let entry = entries[name] else {
                throw Error.missingTensor(name)
            }
            guard entry.dtype == "F32" else {
                throw Error.unsupportedDType("\(entry.dtype) for tensor \(name)")
            }
            if let exp = expectedShape, exp != entry.shape {
                throw Error.shapeMismatch(
                    name: name, expected: exp, actual: entry.shape
                )
            }
            let count = entry.shape.reduce(1, *)
            guard entry.dataLength == count * MemoryLayout<Float>.size else {
                throw Error.truncatedTensorData(name: name)
            }
            // Slice into the payload. Header offsets are relative to
            // the start of the data segment, so subtract the header's
            // size and the 8-byte length prefix that we already
            // skipped when slicing `payload`.
            let start = entry.dataOffset
            let end = start + entry.dataLength
            guard end <= payload.count else {
                throw Error.truncatedTensorData(name: name)
            }
            let slice = payload[start..<end]
            var out = [Float](repeating: 0, count: count)
            out.withUnsafeMutableBufferPointer { dst in
                _ = slice.copyBytes(to: UnsafeMutableRawBufferPointer(dst))
            }
            return (out, entry.shape)
        }

        /// Read an `Int32` tensor by name.
        func int32(_ name: String, expectedShape: [Int]? = nil) throws
            -> (values: [Int32], shape: [Int])
        {
            guard let entry = entries[name] else {
                throw Error.missingTensor(name)
            }
            guard entry.dtype == "I32" else {
                throw Error.unsupportedDType("\(entry.dtype) for tensor \(name)")
            }
            if let exp = expectedShape, exp != entry.shape {
                throw Error.shapeMismatch(
                    name: name, expected: exp, actual: entry.shape
                )
            }
            let count = entry.shape.reduce(1, *)
            guard entry.dataLength == count * MemoryLayout<Int32>.size else {
                throw Error.truncatedTensorData(name: name)
            }
            let start = entry.dataOffset
            let end = start + entry.dataLength
            guard end <= payload.count else {
                throw Error.truncatedTensorData(name: name)
            }
            let slice = payload[start..<end]
            var out = [Int32](repeating: 0, count: count)
            out.withUnsafeMutableBufferPointer { dst in
                _ = slice.copyBytes(to: UnsafeMutableRawBufferPointer(dst))
            }
            return (out, entry.shape)
        }
    }

    /// Load a safetensors file from disk.
    static func load(url: URL) throws -> Bundle {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Error.fileNotFound(url.path)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)

        // 8-byte LE uint64 = JSON header length
        guard data.count >= 8 else {
            throw Error.unreadableHeader("file shorter than 8 bytes")
        }
        let headerLength: UInt64 = data.withUnsafeBytes { raw in
            raw.load(fromByteOffset: 0, as: UInt64.self)
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLength)
        guard headerEnd <= data.count else {
            throw Error.unreadableHeader("declared header length exceeds file")
        }
        let headerData = data[headerStart..<headerEnd]

        struct EntryRaw: Decodable {
            let dtype: String
            let shape: [Int]
            let data_offsets: [Int]
        }
        // The header is a JSON object whose keys are tensor names plus
        // an optional `__metadata__` reserved key. We can't decode it
        // straight into a `[String: EntryRaw]` because of that side
        // channel — strip it manually.
        guard
            var raw =
                try JSONSerialization.jsonObject(with: Data(headerData))
                as? [String: Any]
        else {
            throw Error.unreadableHeader("header is not a JSON object")
        }
        raw.removeValue(forKey: "__metadata__")

        var entries: [String: TensorEntry] = [:]
        for (name, value) in raw {
            guard let dict = value as? [String: Any],
                let dtype = dict["dtype"] as? String,
                let shape = dict["shape"] as? [Int],
                let offsets = dict["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw Error.unreadableHeader(
                    "tensor entry '\(name)' is malformed"
                )
            }
            let length = offsets[1] - offsets[0]
            guard length >= 0 else {
                throw Error.unreadableHeader(
                    "tensor '\(name)' has negative data length"
                )
            }
            entries[name] = TensorEntry(
                dtype: dtype,
                shape: shape,
                dataOffset: offsets[0],
                dataLength: length
            )
        }

        let payload = data.subdata(in: headerEnd..<data.count)
        return Bundle(entries: entries, payload: payload)
    }
}
