# Proxima KWS TinyML Model V2 (Anti-Overfitting Pipeline)

## Overview
Trained with comprehensive anti-overfitting measures: SpecAugment, waveform augmentation, L2 decay, label smoothing, and adaptive LR scheduling.

## Key Specifications
- **Architecture**: DS-CNN V4 (16-48 channels, 4 DS blocks)
- **Input Shape**: `(201, 40, 1)` Log-Mel Spectrogram
- **Classes**: `0 = UNKNOWN`, `1 = PROXIMA`
- **Parameters**: `6,378`
- **INT8 Model Size**: `22,904 bytes` (**22.37 KB** - well below 256 KB budget)
- **Estimated Tensor Arena**: `~48 KB`

## Performance on Independent TEST Partition
- **Test Accuracy**: **99.08%**
- **Precision**: **98.53%**
- **Recall**: **100.00%**
- **F1-Score**: **99.26%**
- **PROXIMA False-Negative Rate**: **0.00%**
- **UNKNOWN False-Positive Rate**: **2.38%**

## Files
- `best_model.keras`: Best trained Float32 Keras model checkpoint.
- `proxima_kws_float32.tflite`: Standard TensorFlow Lite Float32 model.
- `proxima_kws_int8.tflite`: Full integer INT8 quantized model for ESP32.
- `model_summary.txt`: Layer-by-layer parameter summary.
- `training_history.csv`: Per-epoch train/val loss and accuracy.
- `training_curves.png`: Loss and accuracy training curves.
- `test_results.txt`: Detailed evaluation report.
- `confusion_matrix.png`: Test confusion matrix.
