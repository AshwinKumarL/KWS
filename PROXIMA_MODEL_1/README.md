# Proxima KWS TinyML Model (Depthwise-Separable CNN)

## Overview
This directory contains the trained, validated, and INT8 quantized Depthwise-Separable CNN (DS-CNN) designed for keyword spotting ('Proxima') on microcontrollers such as the ESP32.

## Key Specifications
- **Architecture**: Depthwise-Separable CNN (DS-CNN V3)
- **Input Shape**: `(201, 40, 1)` Log-Mel Spectrogram
- **Classes**: `0 = UNKNOWN`, `1 = PROXIMA`
- **Parameters**: `2,842`
- **INT8 Model Size**: `14,856 bytes` (**14.51 KB**) — *well below the 256 KB budget*
- **Estimated Tensor Arena**: `~48 KB`

## Performance on Independent TEST Partition
- **Test Accuracy**: **90.83%**
- **Precision**: **90.14%**
- **Recall**: **95.52%**
- **F1-Score**: **92.75%**
- **PROXIMA False-Negative Rate**: **4.48%**
- **UNKNOWN False-Positive Rate**: **16.67%**

## Files in this Directory
- `best_model.keras`: Best trained Float32 Keras model checkpoint.
- `proxima_kws_float32.tflite`: Standard TensorFlow Lite Float32 model.
- `proxima_kws_int8.tflite`: Full integer INT8 quantized model for ESP32.
- `model_summary.txt`: Layer-by-layer parameter summary and progression history.
- `training_history.csv`: Per-epoch train/val loss and accuracy.
- `test_results.txt`: Detailed evaluation report on the test set.
- `confusion_matrix.png`: Annotated visualization of test confusion matrix.
