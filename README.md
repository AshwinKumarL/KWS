# PROXIMA — Edge Keyword Spotting (KWS) System

An end-to-end, ultra-lightweight **Keyword Spotting (KWS)** system optimized for the wake-word **"PROXIMA"**. Designed for constrained embedded edge microcontrollers (such as the **ESP32**, **STM32**, and **Raspberry Pi Pico**) with strict flash (< 256 KB) and internal SRAM constraints.

Includes a complete Python training & preprocessing pipeline, full integer **INT8 TensorFlow Lite** quantized models, and a modular **Developer Testing Panel** with real-time microphone active listening, audio file upload, and dynamic model hot-swapping.

---

## Key Highlights

- **Target Keyword**: `"PROXIMA"`
- **Classes**: `0: UNKNOWN` | `1: PROXIMA`
- **Architecture**: Depthwise-Separable Convolutional Neural Network (**DS-CNN**)
- **Quantization**: Full Integer **INT8** post-training quantization with integer input/output tensors
- **Model Size**: **14.51 KB** (only **5.67%** of a 256 KB ESP32 flash budget)
- **Parameters**: **1,764** total parameters
- **Inference Latency**: **~0.2 ms** on desktop CPU (practical real-time on edge MCU)
- **Feature Extraction**: 40-band Log-Mel Spectrogram ($201 \times 40 \times 1$) computed over 2.0-second 16 kHz audio

---

## System Architecture

```
Microphone / Audio Input (16 kHz)
          ↓
2.0-Second Audio Standardization (32,000 samples)
          ↓
Log-Mel Feature Extraction (HTK scale, 40 Mel bins, 20-8000 Hz)
          ↓
Feature Matrix: 201 × 40 × 1
          ↓
DS-CNN INT8 Model:
  ├── Conv2D (16 Filters, 3×3, Stride 2) + ReLU
  ├── Depthwise Conv2D (16 Filters, 3×3) + Pointwise Conv2D (16 Filters, 1×1) + ReLU
  ├── Depthwise Conv2D (24 Filters, 3×3) + Pointwise Conv2D (24 Filters, 1×1) + ReLU
  ├── Depthwise Conv2D (32 Filters, 3×3) + Pointwise Conv2D (32 Filters, 1×1) + ReLU
  ├── Global Average Pooling 2D
  └── Dense Softmax Classification Head (2 Classes)
          ↓
Output: UNKNOWN vs. PROXIMA Probability
```

---

## Model Specifications & Edge Footprint

| Metric | DS-CNN V3 (INT8 Quantized) | Float32 Reference |
| :--- | :--- | :--- |
| **Model File** | `models/proxima_v3_int8.tflite` | `models/proxima_v3_float32.tflite` |
| **File Size** | **14.51 KB** (14,856 bytes) | 16.52 KB |
| **Quantization** | Full Integer INT8 | Float32 |
| **Total Parameters** | 1,764 | 1,764 |
| **ESP32 256 KB Budget** | **5.67%** | 6.45% |
| **Estimated Tensor Arena** | ~47 KB SRAM | ~166 KB SRAM |
| **PSRAM Requirement** | None (fits in internal SRAM) | None |

---

## Developer Testing Panel

A browser-based developer dashboard for benchmarking, testing, and hot-swapping KWS models before hardware deployment.

### Features
1. **Live Voice Listening**: Continuous 2.0s rolling window with real-time volume level meter (dB), oscilloscope visualizer, and instant green **`● PROXIMA DETECTED`** indicator.
2. **Audio File Upload**: Drag-and-drop or select any audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`) to inspect confidence percentages and inference latency.
3. **Microphone Recording**: Native `MediaRecorder` audio capture with normal-pitch WAV playback review and one-click model evaluation.
4. **Dynamic Model Hot-Swapping**: Switch between models in `models/` on the fly without restarting the server or reloading the frontend.
5. **Dynamic Neural Network Pipeline**: Automatically introspects the active `.tflite` model and renders its layer graph.
6. **Detection History & Sensitivity**: Timestamped developer logs with an adjustable confidence threshold slider (40%–95%).

---

## Quick Start

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/AshwinKumarL/KWS.git
cd KWS
pip install fastapi uvicorn python-multipart soundfile numpy scipy torch torchaudio tensorflow
```

### 2. Launch the Developer Testing Panel

Start the panel server:

```bash
python -m uvicorn panel.app:app --host 127.0.0.1 --port 8000
```

Open your browser at:
```
http://127.0.0.1:8000/
```

---

## Repository Structure

```
.
├── models/                         # Hot-swappable TFLite models
│   ├── proxima_v1_int8.tflite      # Baseline model (8 channels, 6.08 KB)
│   ├── proxima_v2_int8.tflite      # Intermediate model (12 channels, 9.41 KB)
│   ├── proxima_v3_int8.tflite      # Active production model (14.51 KB)
│   └── proxima_v3_float32.tflite   # Float32 reference model
├── panel/                          # Developer Testing Dashboard
│   ├── app.py                      # FastAPI REST API server
│   ├── kws_engine.py               # Standalone inference & preprocessing engine
│   └── static/                     # Web UI frontend
│       ├── index.html              # Modern AI-lab interface
│       ├── styles.css              # Cyberpunk dark theme styling
│       └── app.js                  # Client Web Audio API & audio controller
├── PROXIMA_MODEL_1/                # Training run outputs & evaluation
│   ├── best_model.keras            # Best trained Keras checkpoint
│   ├── proxima_kws_int8.tflite     # Exported INT8 model
│   ├── confusion_matrix.png        # Test set evaluation matrix
│   └── test_results.txt            # Benchmark metrics
├── process_dataset.py              # Audio standardization (16 kHz, mono, 2.0s)
├── split_dataset.py                # Train/test stratified dataset splitter
├── extract_features.py             # Log-Mel spectrogram feature extractor
├── train_proxima_model.py          # PyTorch / Keras DS-CNN training script
├── test_panel_api.py               # Automated test suite for panel API
└── README.md
```

---

## License

MIT License. Open for development, research, and embedded deployment.
