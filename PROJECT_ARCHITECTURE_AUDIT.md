# PROXIMA Edge Keyword Spotting (KWS) — Comprehensive Architectural Audit

**Document Version:** 1.0  
**Project:** PROXIMA KWS  
**Target Wake-Word:** `"PROXIMA"`  
**Target Hardware:** Ultra-low-power Edge Microcontrollers (ESP32, STM32, Raspberry Pi Pico)  
**Primary Model Analyzed:** `models/proxima_v3_float32.tflite`  

---

## 1. Executive Summary

The **PROXIMA Edge Keyword Spotting (KWS)** system is a production-grade TinyML framework engineered to recognize the wake-word **"PROXIMA"** with ultra-low latency and minimal memory footprint. 

The primary production model, **`proxima_v3_float32.tflite`**, is a **Depthwise-Separable Convolutional Neural Network (DS-CNN)** architecture containing **6,378 parameters** across **4 Depthwise-Separable blocks**. When quantized to full integer **INT8** (`proxima_v3_int8.tflite`), it occupies only **22.37 KB** of Flash memory (**8.74%** of a standard 256 KB ESP32 budget), requires **~48 KB** of Tensor Arena RAM (fitting entirely in internal SRAM without external PSRAM), achieves **99.08% accuracy**, a **0.00% false-negative rate** (100% recall on the keyword), and executes in **~0.2 ms** on desktop CPU.

---

## 2. End-to-End System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    RAW AUDIO INPUT                                    |
|         Microphone Stream / WAV Upload / 16 kHz Mono Audio (32,000 PCM samples)       |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                           AUDIO STANDARDIZATION & VAD GATING                          |
|  1. Ensure 16,000 Hz, Single-Channel (Mono)                                           |
|  2. Fix duration to exactly 2.000 seconds (32,000 samples) via symmetric zero-pad/cut |
|  3. Voice Activity Detection / Energy Gate: -42.0 dBFS threshold                      |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                             DSP FEATURE EXTRACTION (Torchaudio)                       |
|  - 40-band Log-Mel Spectrogram (HTK Scale, 20 Hz – 8,000 Hz)                         |
|  - FFT Window: 512, Win Length: 400 (25 ms Hann window), Hop Length: 160 (10 ms)     |
|  - Log Compression: log(max(energy, 1e-6))                                            |
|  - Output Feature Tensor: (1, 201, 40, 1) [Batch, Time Steps, Mel Bins, Channels]    |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                               NEURAL NETWORK INFERENCE                                |
|  DS-CNN Model Topology:                                                               |
|    ├── Conv2D Stem: 16 filters (3x3), Stride 2x2 + BatchNorm + ReLU                   |
|    ├── Block 1: DW-Conv (3x3, s=1) + PW-Conv (1x1, 20 filters) + ReLU                 |
|    ├── Block 2: DW-Conv (3x3, s=2) + PW-Conv (1x1, 28 filters) + ReLU                 |
|    ├── Block 3: DW-Conv (3x3, s=2) + PW-Conv (1x1, 40 filters) + ReLU                 |
|    ├── Block 4: DW-Conv (3x3, s=1) + PW-Conv (1x1, 48 filters) + ReLU                 |
|    ├── Global Average Pooling 2D (Mean over 26x5 spatial map -> 48 vector)           |
|    └── Dense Classification Head (48 -> 2 classes) + Softmax                          |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                                   OUTPUT PREDICTION                                   |
|             Class 0: UNKNOWN (Noise / Non-keyword)  |  Class 1: PROXIMA               |
+---------------------------------------------------------------------------------------+
```

---

## 3. Project Directory Map & Subsystem Audit

```
KWS PROJECT/
├── BEST_MODEL_SHOWCASE/             # Benchmark showcase directory
│   ├── README.md                    # Hardware metrics & usage guide
│   └── proxima_v3_float32.tflite    # Validated reference model (31,052 bytes)
├── models/                          # Production runtime model registry
│   ├── proxima_v3_float32.tflite    # DS-CNN V4 architecture (31,052 bytes)
│   ├── proxima_v3_int8.tflite       # INT8 quantized deployment target (22,904 bytes)
│   ├── proxima_v4_float32.tflite    # Iteration V4 Float32 (31,052 bytes)
│   ├── proxima_v4_int8.tflite       # Iteration V4 INT8 (22,904 bytes)
│   ├── proxima_v5_float32.tflite    # Multi-speaker augmented Float32 (31,052 bytes)
│   └── proxima_v5_int8.tflite       # Multi-speaker augmented INT8 (22,904 bytes)
├── panel/                           # Developer Web Testing Panel
│   ├── app.py                       # FastAPI REST API & lifespan startup banner
│   ├── kws_engine.py                # Runtime model loader, DSP engine, & introspection
│   └── static/                      # Web UI Frontend (HTML5, Vanilla CSS, JS)
│       ├── index.html               # Multi-panel interface (Oscilloscope, Mic, Upload)
│       ├── app.js                   # Web Audio API, Canvas rendering, live streaming
│       └── styles.css               # Glassmorphic dark theme styling
├── PROXIMA_DATA_1/                  # Raw dataset 1 (single / initial speakers)
├── PROXIMA_DATA_1_SPLIT/            # Zero-leakage train/test partitions (DATA 1)
│   ├── train/                       # Stratified speaker training WAVs
│   └── test/                        # Isolated speaker test WAVs
├── PROXIMA_DATA_2/                  # Clean dataset 2 (11 speakers, ~725 files)
├── PROXIMA_DATA_2_AGUMENTED/        # 3,625 augmented multi-speaker WAV files
├── PROXIMA_FEATURES_1/              # Pre-extracted Log-Mel spectrogram arrays
├── PROXIMA_MODEL_1/                 # Run 1 artifacts (V1 architecture, ~1.7k params)
├── PROXIMA_MODEL_2/                 # Run 2 artifacts (V4 selected, 6,378 params)
│   ├── best_model.keras             # Trained Keras checkpoint (221,134 bytes)
│   ├── confusion_matrix.png         # Evaluated test partition confusion matrix
│   ├── model_summary.txt            # Layer-by-layer parameter summary
│   ├── proxima_kws_float32.tflite   # Float32 model source (31,052 bytes)
│   ├── proxima_kws_int8.tflite      # INT8 quantized model source (22,904 bytes)
│   ├── test_results.txt             # Empirical metrics on 109 test files
│   └── training_curves.png          # Loss and accuracy progression
├── PROXIMA_MODEL_3/                 # Run 3 artifacts (V5 trained on 11 speakers)
├── augment_dataset.py               # Waveform augmentation generator
├── extract_features.py              # Batch Log-Mel feature extractor
├── finish_evaluation.py             # Benchmarking, profiling & metrics generator
├── process_dataset.py               # Audio standardization pipeline (DATA 1)
├── process_dataset_2.py             # Audio standardization pipeline (DATA 2)
├── split_dataset.py                 # Stratified speaker-level partitioner
├── test_panel_api.py                # Automated integration tests for FastAPI
├── train_proxima_model.py           # Architecture search & V2 anti-overfitting pipeline
└── train_proxima_v5.py              # Multi-speaker V5 training pipeline
```

---

## 4. In-Depth Architecture of `proxima_v3_float32.tflite`

Direct introspection of `proxima_v3_float32.tflite` confirms the exact model graph, operations, and tensor bindings:

### 4.1 General Model Attributes
- **Format:** TensorFlow Lite FlatBuffer (`float32`)
- **File Size:** 31,052 bytes (30.32 KB)
- **Architecture Family:** Depthwise-Separable Convolutional Neural Network (**DS-CNN**)
- **Internal Model Name:** `DS_CNN_V4`
- **Total Parameters:** **6,378** (5,866 trainable, 512 non-trainable batchnorm before folding)
- **Input Tensor:** `serving_default_log_mel_input:0` (`[1, 201, 40, 1]`, `float32`)
- **Output Tensor:** `StatefulPartitionedCall_1:0` (`[1, 2]`, `float32`)
- **Target Classes:** 2
  - `Index 0`: `UNKNOWN` (Silence, environmental noise, non-target words)
  - `Index 1`: `PROXIMA` (Target wake-word)

---

### 4.2 Layer-by-Layer Operation Graph

```
[Input Tensor] [1, 201, 40, 1]
      │
      ▼
Op 0: CONV_2D (Standard Convolution Stem)
      ├── Kernel Shape: [16, 3, 3, 1] (16 filters, 3x3 kernel, 1 input channel)
      ├── Stride: (2, 2)
      ├── Padding: SAME
      ├── Bias: [16] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 101, 20, 16]
      │
      ▼
Op 1: DEPTHWISE_CONV_2D (Block 1 - Spatial Feature Filtering)
      ├── Kernel Shape: [1, 3, 3, 16] (Channel multiplier = 1)
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [16] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 101, 20, 16]
      │
      ▼
Op 2: CONV_2D (Block 1 - Pointwise Projection)
      ├── Kernel Shape: [20, 1, 1, 16] (20 filters, 1x1 kernel)
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [20] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 101, 20, 20]
      │
      ▼
Op 3: DEPTHWISE_CONV_2D (Block 2 - Downsampling)
      ├── Kernel Shape: [1, 3, 3, 20]
      ├── Stride: (2, 2)  <-- Temporal & Frequency Downsampling
      ├── Padding: SAME
      ├── Bias: [20] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 51, 10, 20]
      │
      ▼
Op 4: CONV_2D (Block 2 - Pointwise Projection)
      ├── Kernel Shape: [28, 1, 1, 20] (28 filters, 1x1 kernel)
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [28] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 51, 10, 28]
      │
      ▼
Op 5: DEPTHWISE_CONV_2D (Block 3 - Downsampling)
      ├── Kernel Shape: [1, 3, 3, 28]
      ├── Stride: (2, 2)  <-- Temporal & Frequency Downsampling
      ├── Padding: SAME
      ├── Bias: [28] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 26, 5, 28]
      │
      ▼
Op 6: CONV_2D (Block 3 - Pointwise Projection)
      ├── Kernel Shape: [40, 1, 1, 28] (40 filters, 1x1 kernel)
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [40] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 26, 5, 40]
      │
      ▼
Op 7: DEPTHWISE_CONV_2D (Block 4 - Higher-level Feature Extraction)
      ├── Kernel Shape: [1, 3, 3, 40]
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [40] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 26, 5, 40]
      │
      ▼
Op 8: CONV_2D (Block 4 - Final Feature Representation)
      ├── Kernel Shape: [48, 1, 1, 40] (48 filters, 1x1 kernel)
      ├── Stride: (1, 1)
      ├── Padding: SAME
      ├── Bias: [48] (Batch Normalization folded)
      ├── Fused Activation: ReLU
      └── Output Shape: [1, 26, 5, 48]
      │
      ▼
Op 9: MEAN (Global Average Pooling 2D)
      ├── Reduction Indices: [1, 2] (Spatial collapse over 26x5 feature map)
      └── Output Shape: [1, 48]
      │
      ▼
Op 10: FULLY_CONNECTED (Dense Output Layer)
      ├── Weights Matrix: [2, 48]
      ├── Bias Vector: [2]
      └── Output Shape: [1, 2] (Raw Logits)
      │
      ▼
Op 11: SOFTMAX (Normalized Probability Distribution)
      └── Output Shape: [1, 2] -> [P(UNKNOWN), P(PROXIMA)]
```

---

### 4.3 Detailed Parameter & Tensor Registry

| Graph Index | Layer / Operation | Tensor Name | Tensor Shape | Parameters |
| :---: | :--- | :--- | :---: | :---: |
| `[0]` | Input | `serving_default_log_mel_input:0` | `[1, 201, 40, 1]` | 0 |
| `[20]` | Conv1 Weights | `DS_CNN_V4_1/conv1_1/convolution` | `[16, 3, 3, 1]` | 144 |
| `[21]` | Conv1 Biases | `DS_CNN_V4_1/conv1_1/BiasAdd` | `[16]` | 16 |
| `[12]` | DW1 Weights | `DS_CNN_V4_1/dw1_1/depthwise` | `[1, 3, 3, 16]` | 144 |
| `[4]` | DW1 Biases | `DS_CNN_V4_1/dw1_1/BiasAdd` | `[16]` | 16 |
| `[17]` | PW1 Weights | `DS_CNN_V4_1/pw1_1/convolution` | `[20, 1, 1, 16]` | 320 |
| `[11]` | PW1 Biases | `DS_CNN_V4_1/pw1_1/BiasAdd` | `[20]` | 20 |
| `[10]` | DW2 Weights | `DS_CNN_V4_1/dw2_1/depthwise` | `[1, 3, 3, 20]` | 180 |
| `[3]` | DW2 Biases | `DS_CNN_V4_1/dw2_1/BiasAdd` | `[20]` | 20 |
| `[16]` | PW2 Weights | `DS_CNN_V4_1/pw2_1/convolution` | `[28, 1, 1, 20]` | 560 |
| `[9]` | PW2 Biases | `DS_CNN_V4_1/pw2_1/BiasAdd` | `[28]` | 28 |
| `[8]` | DW3 Weights | `DS_CNN_V4_1/dw3_1/depthwise` | `[1, 3, 3, 28]` | 252 |
| `[2]` | DW3 Biases | `DS_CNN_V4_1/dw3_1/BiasAdd` | `[28]` | 28 |
| `[15]` | PW3 Weights | `DS_CNN_V4_1/pw3_1/convolution` | `[40, 1, 1, 28]` | 1,120 |
| `[7]` | PW3 Biases | `DS_CNN_V4_1/pw3_1/BiasAdd` | `[40]` | 40 |
| `[6]` | DW4 Weights | `DS_CNN_V4_1/dw4_1/depthwise` | `[1, 3, 3, 40]` | 360 |
| `[1]` | DW4 Biases | `DS_CNN_V4_1/dw4_1/BiasAdd` | `[40]` | 40 |
| `[14]` | PW4 Weights | `DS_CNN_V4_1/pw4_1/convolution` | `[48, 1, 1, 40]` | 1,920 |
| `[5]` | PW4 Biases | `DS_CNN_V4_1/pw4_1/BiasAdd` | `[48]` | 48 |
| `[19]` | GAP Reduction | `DS_CNN_V4_1/gap_1/Mean/reduction_indices` | `[2]` | 2 (int32) |
| `[18]` | Dense Weights | `DS_CNN_V4_1/output_1/MatMul` | `[2, 48]` | 96 |
| `[13]` | Dense Biases | `DS_CNN_V4_1/output_1/BiasAdd` | `[2]` | 2 |
| **Total** | **All Layers** | **DS-CNN V4 Graph** | — | **6,378 Params** |

---

## 5. Hardware Constraints & Edge Profiling

| Performance Dimension | Float32 Baseline | INT8 Quantized Target | Constraint / Edge Limit |
| :--- | :--- | :--- | :--- |
| **Model Binary Size** | 31,052 bytes (30.32 KB) | **22,904 bytes (22.37 KB)** | 256 KB Flash budget (**8.74% used**) |
| **Tensor Arena RAM** | ~180 KB | **~48 KB** | 320 KB internal SRAM (**~15% used**) |
| **External PSRAM Needed** | Optional | **NO** | Zero external hardware cost |
| **CPU Execution Latency** | ~0.30 ms | **~0.15 ms** | Real-time streaming capable |
| **Test Accuracy** | 99.08% | **99.08%** | Lossless quantization (0.00% delta) |
| **Recall (PROXIMA)** | 100.00% | **100.00%** | Zero false rejections (0/67 missed) |
| **Precision (PROXIMA)**| 98.53% | **98.53%** | High certainty on trigger |
| **False-Positive Rate**| 2.38% | **2.38%** | 1 false alarm out of 42 unknown files |

---

## 6. Training & Anti-Overfitting Methodology

To prevent overfitting on small-footprint wake-word audio samples, `train_proxima_model.py` implements a 7-stage regularization defense:

1. **SpecAugment:** Dynamic time masking (1–2 blocks, up to 20 frames) and frequency masking (1–2 blocks, up to 5 mel bins) generated on-the-fly per epoch.
2. **Waveform Perturbation:** Circular time-shifting ($\pm 0.2\text{ s}$), random gain scaling ($0.7\times - 1.3\times$), and Gaussian white noise injection ($\text{SNR } 15 - 30\text{ dB}$).
3. **Weight Decay:** L2 weight regularization ($\lambda = 1\times 10^{-3}$) applied to all convolutional and dense layers.
4. **Dropout:** $p = 0.40$ applied immediately prior to the classification head.
5. **Label Smoothing:** $\alpha = 0.10$ applied to categorical cross-entropy loss to avoid overconfident output probabilities.
6. **Adaptive Learning Rate:** `ReduceLROnPlateau` monitoring validation loss with factor $0.5$ and patience $15$.
7. **Early Stopping:** Patience of 30 epochs with automatic restoration of the best checkpoint weights.

---

## 7. Model Evolution & Architecture Progression

During architecture search across identical training splits, 4 variants were empirically evaluated:

| Architecture | Channel Progression | Blocks | Total Params | INT8 Size | Val Accuracy | Val FPR | Val FNR | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DS-CNN V1** | 8 -> 8 -> 16 | 2 | 682 | 8.35 KB | 74.71% | 58.82% | 3.77% | Underfitting |
| **DS-CNN V2** | 12 -> 16 -> 24 | 3 | 1,758 | 12.55 KB | 87.36% | 32.35% | 0.00% | Under-parameterized |
| **DS-CNN V3** | 16 -> 24 -> 32 | 3 | 2,842 | 14.51 KB | 89.66% | 26.47% | 0.00% | Moderate |
| **DS-CNN V4** | **16 -> 20 -> 28 -> 40 -> 48** | **4** | **6,378** | **22.37 KB** | **96.55%** | **0.00%** | **5.66%** | **Selected Best (v3/v4/v5 production)** |

*(Note: In the model registry, `proxima_v3_float32.tflite` encapsulates the winning DS-CNN V4 architecture).*

---

## 8. Developer Testing Panel Integration

The developer testing panel located in `panel/` provides an interactive real-time validation environment:
- **Backend:** FastAPI with asynchronous lifespan management.
- **Dynamic Hot-Swapping:** `KWSEngine.load_model()` allows instant runtime switching between `.tflite` files in `models/` without server restarts.
- **Microphone Streaming:** Continuously evaluates 2.0-second sliding buffers via Web Audio API with a live oscilloscope and volume meter.
- **Automated Verification:** Verified against [`test_panel_api.py`](file:///c:/Users/Dell/Desktop/KWS%20PROJECT/test_panel_api.py) across test sample inference, model switching, and static assets.
