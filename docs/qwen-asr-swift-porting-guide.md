# Qwen ASR Model Components for Swift MLX Port

This document provides a comprehensive reference for porting the Qwen ASR (FunASR-Nano) model from Python MLX to Swift MLX.

## Architecture Overview

The Qwen ASR model is an end-to-end speech recognition system that combines:

1. **Audio Frontend** - Mel spectrogram extraction with Low Frame Rate (LFR) processing
2. **SenseVoice Encoder** - SANM-based audio encoder (70 layers)
3. **Audio Adaptor** - Projects encoder output to LLM embedding space
4. **Qwen3 LLM Decoder** - Generates text from audio embeddings

```
Audio Waveform
     │
     ▼
┌─────────────────┐
│ Audio Frontend  │  log_mel_spectrogram → LFR → CMVN
│ (80 mels → 560) │
└────────┬────────┘
         │ (batch, seq, 560)
         ▼
┌─────────────────┐
│ SenseVoice      │  70 SANM layers (1 + 49 + 20)
│ Encoder         │
│ (560 → 512)     │
└────────┬────────┘
         │ (batch, seq, 512)
         ▼
┌─────────────────┐
│ Audio Adaptor   │  Downsample + Linear + Transformer
│ (512 → 1024)    │
└────────┬────────┘
         │ (batch, seq/2, 1024)
         ▼
┌─────────────────┐
│ Qwen3 LLM       │  28 transformer layers
│ (1024 → vocab)  │
└────────┬────────┘
         │
         ▼
     Text Output
```

---

## 1. Audio Frontend

### Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `N_FFT` | 400 | FFT window size (25ms at 16kHz) |
| `HOP_LENGTH` | 160 | Hop length (10ms) |
| `N_MELS` | 80 | Number of mel filterbank channels |
| `LFR_M` | 7 | Number of frames to stack |
| `LFR_N` | 6 | Subsampling factor |

### Processing Pipeline

#### 1.1 Log Mel Spectrogram

```swift
// Input: audio waveform (samples,)
// Output: (n_frames, 80)

func logMelSpectrogram(audio: MLXArray) -> MLXArray {
    // 1. Apply Hamming window STFT
    let window = hamming(N_FFT)
    let freqs = stft(audio, window: window, nFft: N_FFT, hopLength: HOP_LENGTH)

    // 2. Compute power spectrum (discard last frame)
    let magnitudes = freqs[..<(-1), .all].abs().square()

    // 3. Apply mel filterbank (HTK scale, Slaney normalization)
    let filters = melFilters(sampleRate: SAMPLE_RATE, nFft: N_FFT, nMels: N_MELS,
                             norm: "slaney", melScale: "htk")
    let melSpec = matmul(magnitudes, filters.T)

    // 4. Log compression with floor
    return log(maximum(melSpec, 1e-10))
}
```

#### 1.2 Low Frame Rate (LFR) Processing

Stacks consecutive frames and subsamples to reduce sequence length.

```swift
// Input: (n_frames, 80)
// Output: (ceil(n_frames / 6), 560)

func applyLFR(features: MLXArray, lfrM: Int = 7, lfrN: Int = 6) -> MLXArray {
    let (T, nMels) = (features.shape[0], features.shape[1])
    let tLFR = Int(ceil(Double(T) / Double(lfrN)))

    // Left padding: replicate first frame
    let leftPad = (lfrM - 1) / 2
    var padded = features
    if leftPad > 0 {
        let leftPadding = broadcast(features[0..<1], to: [leftPad, nMels])
        padded = concatenate([leftPadding, padded], axis: 0)
    }

    // Right padding: replicate last frame
    let totalNeeded = (tLFR - 1) * lfrN + lfrM
    if totalNeeded > padded.shape[0] {
        let rightPad = totalNeeded - padded.shape[0]
        let rightPadding = broadcast(padded[(-1)...], to: [rightPad, nMels])
        padded = concatenate([padded, rightPadding], axis: 0)
    }

    // Gather indices for vectorized stacking
    let startIndices = arange(tLFR) * lfrN  // (tLFR,)
    let offsets = arange(lfrM)               // (lfrM,)
    let indices = startIndices[.all, .newAxis] + offsets[.newAxis, .all]  // (tLFR, lfrM)

    // Gather and reshape: (tLFR, lfrM, nMels) -> (tLFR, lfrM * nMels)
    let gathered = padded[indices]
    return gathered.reshaped([tLFR, -1])
}
```

#### 1.3 CMVN Normalization

```swift
// Per-utterance normalization when no precomputed stats available
func applyCMVN(features: MLXArray, mean: MLXArray? = nil, istd: MLXArray? = nil) -> MLXArray {
    if let mean = mean, let istd = istd {
        // Precomputed: (x + mean) * istd
        return (features + mean) * istd
    } else {
        // Per-utterance normalization
        let mean = features.mean(axis: 0, keepDims: true)
        let std = features.std(axis: 0, keepDims: true) + 1e-6
        return (features - mean) / std
    }
}
```

---

## 2. SenseVoice Encoder

### Configuration

```swift
struct SenseVoiceEncoderConfig {
    var inputDim: Int = 560       // 80 * 7 (nMels * lfrM)
    var encoderDim: Int = 512     // Hidden dimension
    var numHeads: Int = 4         // Attention heads
    var ffnDim: Int = 2048        // FFN hidden dimension
    var kernelSize: Int = 11      // FSMN kernel size
    var sanmShift: Int = 0        // Asymmetric context shift
    var numEncoders0: Int = 1     // Initial projection layers
    var numEncoders: Int = 49     // Main encoder layers
    var numTPEncoders: Int = 20   // Time-pooling encoder layers
    var dropout: Float = 0.0
}
```

### 2.1 MultiHeadedAttentionSANM (Self-Attention with Memory)

The core attention mechanism combining standard multi-head attention with FSMN for local context.

```swift
class MultiHeadedAttentionSANM: Module {
    let dK: Int           // Head dimension = nFeat / nHead
    let h: Int            // Number of heads
    let nFeat: Int        // Output feature dimension

    // Combined Q/K/V projection
    let linearQKV: Linear  // (inFeat, nFeat * 3)

    // Output projection
    let linearOut: Linear  // (nFeat, nFeat)

    // FSMN block - depthwise convolution
    let fsmnBlock: Conv1d  // (nFeat, nFeat, kernel=11, groups=nFeat, padding=0)

    let leftPadding: Int   // (kernelSize - 1) / 2 + sanmShift
    let rightPadding: Int  // kernelSize - 1 - leftPadding

    init(nHead: Int, inFeat: Int, nFeat: Int, kernelSize: Int = 11, sanmShift: Int = 0) {
        self.dK = nFeat / nHead
        self.h = nHead
        self.nFeat = nFeat

        self.linearQKV = Linear(inFeat, nFeat * 3, bias: true)
        self.linearOut = Linear(nFeat, nFeat, bias: true)

        // Depthwise conv with no built-in padding
        self.fsmnBlock = Conv1d(
            inputChannels: nFeat,
            outputChannels: nFeat,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            groups: nFeat,  // Depthwise
            bias: false
        )

        var lp = (kernelSize - 1) / 2
        if sanmShift > 0 { lp += sanmShift }
        self.leftPadding = lp
        self.rightPadding = kernelSize - 1 - lp
    }

    func forwardFSMN(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = inputs
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2])

        // Apply mask if provided
        if let mask = mask {
            let m = mask.reshaped([b, -1, 1])
            x = x * m
        }

        // Transpose for conv1d: (batch, seq, dim) -> (batch, dim, seq)
        x = x.swappedAxes(1, 2)

        // Explicit padding
        if leftPadding > 0 || rightPadding > 0 {
            x = pad(x, widths: [(0, 0), (0, 0), (leftPadding, rightPadding)])
        }

        // MLX conv1d expects (batch, seq, channels)
        x = x.swappedAxes(1, 2)
        x = fsmnBlock(x)

        // Residual connection
        x = x + inputs

        return x
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (batchSize, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

        // Combined Q/K/V projection
        let qkv = linearQKV(x)
        let splits = split(qkv, parts: 3, axis: -1)
        let (q, k, v) = (splits[0], splits[1], splits[2])

        // Apply FSMN to unprojected value (before multi-head reshape)
        let fsmnMemory = forwardFSMN(v, mask: mask)

        // Reshape for multi-head attention: (batch, seq, nFeat) -> (batch, nHead, seq, dK)
        let qH = q.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)
        let kH = k.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)
        let vH = v.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)

        // Prepare attention mask
        var attnMask: MLXArray? = nil
        if let mask = mask {
            // Convert binary mask to additive: 0 -> -inf
            attnMask = where(mask == 0, Float.infinity * -1, 0.0)
            attnMask = attnMask?.expandedDimensions(axes: [1, 2])  // (batch, 1, 1, seq)
        }

        // Scaled dot-product attention
        let context = MLX.fast.scaledDotProductAttention(
            queries: qH, keys: kH, values: vH,
            scale: pow(Float(dK), -0.5),
            mask: attnMask
        )

        // Reshape back: (batch, nHead, seq, dK) -> (batch, seq, nFeat)
        let contextReshaped = context.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, nFeat])

        // Output projection
        let attOuts = linearOut(contextReshaped)

        // Add FSMN memory AFTER attention
        return attOuts + fsmnMemory
    }
}
```

### 2.2 PositionwiseFeedForward

```swift
class PositionwiseFeedForward: Module {
    let w1: Linear  // (dModel, dFF)
    let w2: Linear  // (dFF, dModel)

    init(dModel: Int, dFF: Int) {
        self.w1 = Linear(dModel, dFF, bias: true)
        self.w2 = Linear(dFF, dModel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ReLU activation (original FunASR uses ReLU, not GELU)
        return w2(relu(w1(x)))
    }
}
```

### 2.3 EncoderLayerSANM

```swift
class EncoderLayerSANM: Module {
    let inSize: Int
    let size: Int

    let norm1: LayerNorm
    let selfAttn: MultiHeadedAttentionSANM
    let norm2: LayerNorm
    let feedForward: PositionwiseFeedForward

    init(inSize: Int, size: Int, nHead: Int, dFF: Int, kernelSize: Int = 11, sanmShift: Int = 0) {
        self.inSize = inSize
        self.size = size

        self.norm1 = LayerNorm(dimensions: inSize)
        self.selfAttn = MultiHeadedAttentionSANM(
            nHead: nHead, inFeat: inSize, nFeat: size,
            kernelSize: kernelSize, sanmShift: sanmShift
        )
        self.norm2 = LayerNorm(dimensions: size)
        self.feedForward = PositionwiseFeedForward(dModel: size, dFF: dFF)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x
        let residual1 = out

        // Pre-norm attention
        out = norm1(out)
        out = selfAttn(out, mask: mask)

        // Residual (only if dimensions match)
        if inSize == size {
            out = out + residual1
        }

        // Pre-norm feed-forward
        let residual2 = out
        out = norm2(out)
        out = feedForward(out)
        out = out + residual2

        return out
    }
}
```

### 2.4 SenseVoiceEncoder

```swift
class SenseVoiceEncoder: Module {
    let config: SenseVoiceEncoderConfig
    let outputSize: Int

    let encoders0: [EncoderLayerSANM]  // 1 layer (560 -> 512)
    let encoders: [EncoderLayerSANM]   // 49 layers (512 -> 512)
    let tpEncoders: [EncoderLayerSANM] // 20 layers (512 -> 512)

    let afterNorm: LayerNorm
    let tpNorm: LayerNorm

    init(config: SenseVoiceEncoderConfig) {
        self.config = config
        self.outputSize = config.encoderDim

        // Initial encoder(s) - handles dimension change
        self.encoders0 = (0..<config.numEncoders0).map { i in
            EncoderLayerSANM(
                inSize: i == 0 ? config.inputDim : config.encoderDim,
                size: config.encoderDim,
                nHead: config.numHeads,
                dFF: config.ffnDim,
                kernelSize: config.kernelSize,
                sanmShift: config.sanmShift
            )
        }

        // Main encoder layers
        self.encoders = (0..<config.numEncoders).map { _ in
            EncoderLayerSANM(
                inSize: config.encoderDim,
                size: config.encoderDim,
                nHead: config.numHeads,
                dFF: config.ffnDim,
                kernelSize: config.kernelSize,
                sanmShift: config.sanmShift
            )
        }

        // Time-pooling encoder layers
        self.tpEncoders = (0..<config.numTPEncoders).map { _ in
            EncoderLayerSANM(
                inSize: config.encoderDim,
                size: config.encoderDim,
                nHead: config.numHeads,
                dFF: config.ffnDim,
                kernelSize: config.kernelSize,
                sanmShift: config.sanmShift
            )
        }

        self.afterNorm = LayerNorm(dimensions: config.encoderDim)
        self.tpNorm = LayerNorm(dimensions: config.encoderDim)
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (batchSize, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

        let lengths = lengths ?? MLXArray.full([batchSize], value: Int32(seqLen))

        // Scale input by sqrt(output_size)
        var out = x * sqrt(Float(outputSize))

        // Initial encoder(s)
        for layer in encoders0 {
            out = layer(out, mask: nil)
        }

        // Main encoder
        for layer in encoders {
            out = layer(out, mask: nil)
        }

        // After norm
        out = afterNorm(out)

        // Time-pooling encoder
        for layer in tpEncoders {
            out = layer(out, mask: nil)
        }

        // Final normalization
        out = tpNorm(out)

        return (out, lengths)
    }
}
```

---

## 3. Audio Adaptor

### Configuration

```swift
struct AudioAdaptorConfig {
    var downsampleRate: Int = 2    // Group this many frames
    var encoderDim: Int = 512      // Input from encoder
    var llmDim: Int = 1024         // Output for LLM
    var ffnDim: Int = 2048         // Intermediate projection
    var nLayer: Int = 2            // Transformer blocks
    var attentionHeads: Int = 8
    var dropout: Float = 0.0
}
```

### 3.1 Standard MultiHeadedAttention (for Adaptor)

```swift
class MultiHeadedAttention: Module {
    let dK: Int
    let h: Int
    let nFeat: Int

    let linearQ: Linear
    let linearK: Linear
    let linearV: Linear
    let linearOut: Linear

    init(nHead: Int, nFeat: Int) {
        self.dK = nFeat / nHead
        self.h = nHead
        self.nFeat = nFeat

        self.linearQ = Linear(nFeat, nFeat, bias: true)
        self.linearK = Linear(nFeat, nFeat, bias: true)
        self.linearV = Linear(nFeat, nFeat, bias: true)
        self.linearOut = Linear(nFeat, nFeat, bias: true)
    }

    func callAsFunction(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batchSize = query.shape[0]

        let q = linearQ(query).reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)
        let k = linearK(key).reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)
        let v = linearV(value).reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)

        var attnMask: MLXArray? = nil
        if let mask = mask {
            attnMask = where(mask == 0, Float.infinity * -1, 0.0)
        }

        let context = MLX.fast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(dK), -0.5),
            mask: attnMask
        )

        let output = context.transposed(0, 2, 1, 3).reshaped([batchSize, -1, nFeat])
        return linearOut(output)
    }
}
```

### 3.2 Adaptor EncoderLayer

```swift
class AdaptorEncoderLayer: Module {
    let selfAttn: MultiHeadedAttention
    let feedForward: PositionwiseFeedForward
    let norm1: LayerNorm
    let norm2: LayerNorm

    init(size: Int, nHead: Int, dFF: Int) {
        self.selfAttn = MultiHeadedAttention(nHead: nHead, nFeat: size)
        self.feedForward = PositionwiseFeedForward(dModel: size, dFF: dFF)
        self.norm1 = LayerNorm(dimensions: size)
        self.norm2 = LayerNorm(dimensions: size)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x

        // Pre-norm attention
        let residual1 = out
        out = norm1(out)
        out = selfAttn(query: out, key: out, value: out, mask: mask)
        out = residual1 + out

        // Pre-norm feed-forward
        let residual2 = out
        out = norm2(out)
        out = feedForward(out)
        out = residual2 + out

        return out
    }
}
```

### 3.3 AudioAdaptor

```swift
class AudioAdaptor: Module {
    let config: AudioAdaptorConfig
    let k: Int  // Downsample rate

    let linear1: Linear  // (encoderDim * k, ffnDim)
    let linear2: Linear  // (ffnDim, llmDim)
    let blocks: [AdaptorEncoderLayer]?

    init(config: AudioAdaptorConfig) {
        self.config = config
        self.k = config.downsampleRate

        self.linear1 = Linear(config.encoderDim * k, config.ffnDim, bias: true)
        self.linear2 = Linear(config.ffnDim, config.llmDim, bias: true)

        if config.nLayer > 0 {
            let blockFFNDim = config.llmDim / 4  // 256 for llmDim=1024
            self.blocks = (0..<config.nLayer).map { _ in
                AdaptorEncoderLayer(
                    size: config.llmDim,
                    nHead: config.attentionHeads,
                    dFF: blockFFNDim
                )
            }
        } else {
            self.blocks = nil
        }
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (batchSize, seqLen, dim) = (x.shape[0], x.shape[1], x.shape[2])

        // Pad sequence to be divisible by k
        let chunkNum = (seqLen - 1) / k + 1
        let padNum = chunkNum * k - seqLen

        var out = x
        if padNum > 0 {
            out = pad(out, widths: [(0, 0), (0, padNum), (0, 0)])
        }

        // Reshape to group k consecutive frames: (batch, seq, dim) -> (batch, seq/k, dim*k)
        out = out.reshaped([batchSize, chunkNum, dim * k])

        // Linear projections with ReLU
        out = linear1(out)
        out = relu(out)
        out = linear2(out)

        // Compute output lengths
        let outLengths: MLXArray
        if let lengths = lengths {
            outLengths = (lengths - 1) / k + 1
        } else {
            outLengths = MLXArray.full([batchSize], value: Int32(chunkNum))
        }

        // Apply transformer blocks
        if let blocks = blocks {
            for block in blocks {
                out = block(out, mask: nil)
            }
        }

        return (out, outLengths)
    }
}
```

---

## 4. Qwen3 LLM Decoder

### Configuration

```swift
struct Qwen3Config {
    var vocabSize: Int = 151936
    var hiddenSize: Int = 1024
    var numHiddenLayers: Int = 28
    var numAttentionHeads: Int = 16
    var numKeyValueHeads: Int = 8     // GQA
    var intermediateSize: Int = 3072
    var maxPositionEmbeddings: Int = 40960
    var ropeTheta: Float = 1_000_000.0
    var rmsNormEps: Float = 1e-6
    var tieWordEmbeddings: Bool = true
    var headDim: Int = 64
}
```

### 4.1 RMSNorm

```swift
class RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = x.square().mean(axis: -1, keepDims: true)
        let normed = x * rsqrt(variance + eps)
        return normed * weight
    }
}
```

### 4.2 Qwen3 Attention (with GQA and RoPE)

```swift
class Qwen3Attention: Module {
    let config: Qwen3Config
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    let qProj: Linear    // (hiddenSize, nHeads * headDim)
    let kProj: Linear    // (hiddenSize, nKVHeads * headDim)
    let vProj: Linear    // (hiddenSize, nKVHeads * headDim)
    let oProj: Linear    // (nHeads * headDim, hiddenSize)

    // Per-head QK normalization (Qwen3 specific)
    let qNorm: RMSNorm   // (headDim)
    let kNorm: RMSNorm   // (headDim)

    // Rotary embeddings
    let rope: RoPE

    init(config: Qwen3Config) {
        self.config = config
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = config.hiddenSize

        self.qProj = Linear(dim, nHeads * headDim, bias: false)
        self.kProj = Linear(dim, nKVHeads * headDim, bias: false)
        self.vProj = Linear(dim, nKVHeads * headDim, bias: false)
        self.oProj = Linear(nHeads * headDim, dim, bias: false)

        self.qNorm = RMSNorm(dims: headDim, eps: config.rmsNormEps)
        self.kNorm = RMSNorm(dims: headDim, eps: config.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape for multi-head attention
        queries = queries.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        // Apply QK normalization
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Apply RoPE
        if let cache = cache {
            let offset = cache.0.shape[2]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            keys = concatenate([cache.0, keys], axis: 2)
            values = concatenate([cache.1, values], axis: 2)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let newCache = (keys, values)

        // Scaled dot-product attention
        let output = MLX.fast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )

        // Reshape back and project
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])

        return (oProj(outputReshaped), newCache)
    }
}
```

### 4.3 Qwen3 MLP (SwiGLU)

```swift
class Qwen3MLP: Module {
    let gateProj: Linear  // (hiddenSize, intermediateSize)
    let upProj: Linear    // (hiddenSize, intermediateSize)
    let downProj: Linear  // (intermediateSize, hiddenSize)

    init(config: Qwen3Config) {
        let dim = config.hiddenSize
        let hiddenDim = config.intermediateSize

        self.gateProj = Linear(dim, hiddenDim, bias: false)
        self.upProj = Linear(dim, hiddenDim, bias: false)
        self.downProj = Linear(hiddenDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate) * up)
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}
```

### 4.4 Qwen3 TransformerBlock

```swift
class Qwen3TransformerBlock: Module {
    let selfAttn: Qwen3Attention
    let mlp: Qwen3MLP
    let inputLayernorm: RMSNorm
    let postAttentionLayernorm: RMSNorm

    init(config: Qwen3Config) {
        self.selfAttn = Qwen3Attention(config: config)
        self.mlp = Qwen3MLP(config: config)
        self.inputLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.postAttentionLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        // Self-attention with pre-norm and residual
        var out = x
        let residual1 = out
        out = inputLayernorm(out)
        let (attnOut, newCache) = selfAttn(out, mask: mask, cache: cache)
        out = residual1 + attnOut

        // MLP with pre-norm and residual
        let residual2 = out
        out = postAttentionLayernorm(out)
        out = mlp(out)
        out = residual2 + out

        return (out, newCache)
    }
}
```

### 4.5 Qwen3Model

```swift
class Qwen3Model: Module {
    let config: Qwen3Config
    let embedTokens: Embedding  // (vocabSize, hiddenSize)
    let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    init(config: Qwen3Config) {
        self.config = config
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3TransformerBlock(config: config)
        }
        self.norm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var h: MLXArray
        if let embeddings = inputEmbeddings {
            h = embeddings
        } else {
            h = embedTokens(inputIds!)
        }

        // Create causal mask if needed
        var attnMask = mask
        if attnMask == nil && h.shape[1] > 1 {
            attnMask = MultiHeadAttention.createAdditiveCausalMask(h.shape[1])
        }

        let layerCache = cache ?? Array(repeating: nil, count: layers.count)
        var newCache: [(MLXArray, MLXArray)] = []

        for (layer, c) in zip(layers, layerCache) {
            let (output, layerNewCache) = layer(h, mask: attnMask, cache: c)
            h = output
            newCache.append(layerNewCache)
        }

        return (norm(h), newCache)
    }
}
```

### 4.6 Qwen3ForCausalLM

```swift
class Qwen3ForCausalLM: Module {
    let config: Qwen3Config
    let model: Qwen3Model
    let lmHead: Linear?  // Only if not using tied embeddings

    init(config: Qwen3Config) {
        self.config = config
        self.model = Qwen3Model(config: config)

        if !config.tieWordEmbeddings {
            self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        } else {
            self.lmHead = nil
        }
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        let (out, newCache) = model(
            inputIds: inputIds,
            inputEmbeddings: inputEmbeddings,
            mask: mask,
            cache: cache
        )

        let logits: MLXArray
        if config.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(out)
        } else {
            logits = lmHead!(out)
        }

        return (logits, newCache)
    }

    func getInputEmbeddings() -> Embedding {
        return model.embedTokens
    }
}
```

---

## 5. Special Tokens

| Token | Description |
|-------|-------------|
| `<\|startofspeech\|>` | Marks start of audio region |
| `<\|endofspeech\|>` | Marks end of audio region |
| `<\|im_start\|>` | Chat template - interaction start |
| `<\|im_end\|>` | Chat template - interaction end |

### Prompt Template

```
<|im_start|>system
{system_prompt}
<|im_end|><|im_start|>user
<|startofspeech|><|endofspeech|>
<|im_end|><|im_start|>assistant
```

---

## 6. Weight Conversion Notes

### Key Transformations

1. **FSMN Conv Weights**: PyTorch `(out, 1, kernel)` → MLX `(out, kernel, 1)`
2. **Other Conv Weights**: PyTorch `(out, in, kernel)` → MLX `(out, kernel, in)`

### Quantizable Components

Default 4-bit quantization with group size 64:
- LLM layers: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Adaptor: `linear1, linear2, attention projections`

---

## 7. Inference Pipeline

```swift
func generate(audio: MLXArray) -> String {
    // 1. Audio preprocessing
    let melFeatures = logMelSpectrogram(audio: audio)
    let lfrFeatures = applyLFR(features: melFeatures)
    let normalizedFeatures = applyCMVN(features: lfrFeatures)

    // 2. Encode audio
    let features = normalizedFeatures.expandedDimensions(axis: 0)  // Add batch dim
    let (encoderOut, lengths) = encoder(features)
    let (audioEmbeddings, _) = adaptor(encoderOut, lengths: lengths)

    // 3. Build prompt and merge embeddings
    let promptIds = tokenizer.encode(promptTemplate)
    let textEmbeddings = llm.getInputEmbeddings()(promptIds)
    let mergedEmbeddings = mergeAudioWithText(audioEmbeddings, textEmbeddings, promptIds)

    // 4. Generate tokens
    var cache: [(MLXArray, MLXArray)]? = nil
    var tokens: [Int] = []
    var inputEmbeddings = mergedEmbeddings

    for _ in 0..<maxTokens {
        let (logits, newCache) = llm(inputEmbeddings: inputEmbeddings, cache: cache)
        cache = newCache

        let tokenId = logits[.all, -1, .all].argmax(axis: -1).item(Int.self)

        if eosTokenIds.contains(tokenId) {
            break
        }

        tokens.append(tokenId)
        inputEmbeddings = llm.getInputEmbeddings()([tokenId]).expandedDimensions(axis: 0)
    }

    return tokenizer.decode(tokens)
}
```

---

## 8. Layer Summary

| Component | Layers | Parameters |
|-----------|--------|------------|
| SenseVoice Encoder (encoders0) | 1 | ~3.7M |
| SenseVoice Encoder (encoders) | 49 | ~147M |
| SenseVoice Encoder (tp_encoders) | 20 | ~60M |
| Audio Adaptor | 2 transformer + 2 linear | ~10M |
| Qwen3 LLM | 28 transformer | ~500M |

**Total**: ~720M parameters (varies with quantization)

---

## 9. Supported Languages

| Code | Language |
|------|----------|
| en | English |
| zh | Chinese |
| ja | Japanese |
| ko | Korean |
| es | Spanish |
| fr | French |
| de | German |
| it | Italian |
| pt | Portuguese |
| ru | Russian |
| ar | Arabic |
| th | Thai |
| vi | Vietnamese |
| auto | Auto-detect |

---

## 10. Source Files Reference

| File | Description |
|------|-------------|
| `mlx_audio/stt/models/funasr/audio.py` | Audio preprocessing |
| `mlx_audio/stt/models/funasr/encoder.py` | SenseVoice encoder |
| `mlx_audio/stt/models/funasr/adaptor.py` | Audio adaptor |
| `mlx_audio/stt/models/funasr/qwen3.py` | Qwen3 LLM |
| `mlx_audio/stt/models/funasr/funasr.py` | Main model class |
| `mlx_audio/stt/models/funasr/convert.py` | Weight conversion |
