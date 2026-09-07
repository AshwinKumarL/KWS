# BEST MODEL SHOWCASE: PROXIMA V3 (Float32)

This directory contains the primary and best-performing Keyword Spotting (KWS) model for detecting the wake-word **"PROXIMA"**.

---

## Model Overview

- **Model File**: `proxima_v3_float32.tflite`
- **File Size**: ~30.3 KB (31,052 bytes)
- **Format**: TensorFlow Lite (Float32)
- **Architecture**: Depthwise-Separable Convolutional Neural Network (DS-CNN)
- **Target Keyword**: `"PROXIMA"`
- **Number of Output Classes**: 2
  - `Index 0`: `UNKNOWN` (Background noise, silence, or other speech)
  - `Index 1`: `PROXIMA` (Target wake-word)

---

## Audio & Feature Pipeline Specifications

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Input Audio Sample Rate** | 16,000 Hz (16 kHz) | Single channel (Mono) |
| **Duration** | 2.0 seconds | 32,000 PCM samples |
| **Feature Representation** | Log-Mel Spectrogram | HTK scale, 20 Hz - 8,000 Hz |
| **Mel Filter Banks** | 40 | Bins |
| **FFT Size / Window Size** | 512 samples / 400 samples (25ms) | Hann window |
| **Hop Length (Stride)** | 160 samples (10ms) | Generates 201 time steps |
| **Tensor Input Shape** | `[1, 201, 40, 1]` | `(batch, time_steps, mel_bins, channels)` |
| **Input Data Type** | `float32` | Standard normalized float |

---

## How to Test

You can test this model immediately using the developer panel at `http://localhost:8000`:
1. Navigate to the **Model Selector** dropdown.
2. Select `proxima_v3_float32.tflite`.
3. Use the **Live Microphone Listening** or **Upload Audio File** / **Reference Samples** to benchmark performance.
