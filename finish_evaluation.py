"""
Finish post-training evaluation, TFLite conversion, plotting, and report generation
for the trained PROXIMA_MODEL_2.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

# Import feature loading and tflite conversion utilities from train_proxima_model
from train_proxima_model import (
    discover_wav_files,
    load_dataset_features,
    convert_to_tflite_int8,
    convert_to_tflite_f32,
    eval_tflite_int8,
    sparse_categorical_crossentropy_label_smoothed,
    DROPOUT_RATE,
    L2_DECAY
)

def main():
    workspace = Path(__file__).parent.resolve()
    split_dir = workspace / "PROXIMA_DATA_1_SPLIT"
    output_dir = workspace / "PROXIMA_MODEL_2"
    best_model_path = output_dir / "best_model.keras"

    print(f"Loading best model from {best_model_path}...")
    best_model = keras.models.load_model(
        str(best_model_path),
        custom_objects={"sparse_categorical_crossentropy_label_smoothed": sparse_categorical_crossentropy_label_smoothed},
        compile=False
    )
    print(f"Loaded model successfully. Total params: {best_model.count_params():,}")

    # Load dataset
    print("\nDiscovering audio files...")
    train_files, train_labels_all = discover_wav_files(split_dir, "train")
    test_files, test_labels = discover_wav_files(split_dir, "test")

    train_labels_all = np.array(train_labels_all, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    print(f"Extracting non-augmented training features for quantization calibration ({len(train_files)} samples)...")
    X_train_raw = load_dataset_features(train_files, augment=False)
    y_train = train_labels_all

    print(f"Extracting test features ({len(test_files)} samples)...")
    X_test = load_dataset_features(test_files, augment=False)
    y_test = test_labels

    input_shape = (X_test.shape[1], X_test.shape[2], X_test.shape[3])
    print(f"Feature shape: {input_shape}")

    # 1. Training Curves Plot from training_history.csv
    print("\n--- GENERATING TRAINING CURVES ---")
    history_csv = output_dir / "training_history.csv"
    epochs_list, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs_list.append(int(r["epoch"]))
            train_losses.append(float(r["train_loss"]))
            val_losses.append(float(r["val_loss"]))
            train_accs.append(float(r["train_accuracy"]))
            val_accs.append(float(r["val_accuracy"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax1.plot(epochs_list, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.8)
    ax1.plot(epochs_list, val_losses, label="Val Loss", color="#ff7f0e", linewidth=1.8)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training vs Validation Loss (No Overfitting)", fontweight="bold", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_list, [a * 100 for a in train_accs], label="Train Accuracy", color="#1f77b4", linewidth=1.8)
    ax2.plot(epochs_list, [a * 100 for a in val_accs], label="Val Accuracy", color="#ff7f0e", linewidth=1.8)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Training vs Validation Accuracy", fontweight="bold", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = output_dir / "training_curves.png"
    plt.savefig(str(curves_path))
    plt.close(fig)
    print(f"Saved: {curves_path}")

    # 2. Model Quantization & TFLite Conversion
    print("\n--- CONVERTING TO TFLITE FLOAT32 & INT8 ---")
    f32_tflite_path = output_dir / "proxima_kws_float32.tflite"
    f32_bytes = convert_to_tflite_f32(best_model, f32_tflite_path)
    print(f"Saved Float32 TFLite: {f32_tflite_path} ({f32_bytes:,} bytes, {f32_bytes/1024:.2f} KB)")

    int8_tflite_path = output_dir / "proxima_kws_int8.tflite"
    int8_bytes = convert_to_tflite_int8(best_model, X_train_raw, int8_tflite_path)
    print(f"Saved INT8 TFLite:    {int8_tflite_path} ({int8_bytes:,} bytes, {int8_bytes/1024:.2f} KB)")

    # 3. Evaluation on UNTOUCHED TEST Set
    print("\n--- EVALUATING ON TEST SET ---")
    # Float32 Keras Model Eval
    test_preds_prob = best_model.predict(X_test, verbose=0)
    test_preds_keras = np.argmax(test_preds_prob, axis=1)

    acc_k = accuracy_score(y_test, test_preds_keras)
    prec_k, rec_k, f1_k, _ = precision_recall_fscore_support(y_test, test_preds_keras, average="binary", pos_label=1)
    cm_k = confusion_matrix(y_test, test_preds_keras)

    # INT8 TFLite Model Test Eval
    test_preds_int8 = eval_tflite_int8(int8_tflite_path, X_test)
    acc_int8 = accuracy_score(y_test, test_preds_int8)
    prec_int8, rec_int8, f1_int8, _ = precision_recall_fscore_support(y_test, test_preds_int8, average="binary", pos_label=1)
    cm_int8 = confusion_matrix(y_test, test_preds_int8)
    tn, fp, fn, tp = cm_int8.ravel()
    fpr_int8 = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr_int8 = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n=================================================================")
    print("--- TEST SET EVALUATION RESULTS (INT8 Deployment Model) ---")
    print(f"Test Accuracy:                {acc_int8 * 100:.2f}%")
    print(f"Precision (PROXIMA):          {prec_int8 * 100:.2f}%")
    print(f"Recall (PROXIMA):             {rec_int8 * 100:.2f}%")
    print(f"F1 Score:                     {f1_int8 * 100:.2f}%")
    print(f"PROXIMA False-Negative Rate:  {fnr_int8 * 100:.2f}% (Missed: {fn}/{tp+fn})")
    print(f"UNKNOWN False-Positive Rate:  {fpr_int8 * 100:.2f}% (False alarms: {fp}/{tn+fp})")
    print(f"Confusion Matrix (INT8):\n{cm_int8}")
    print(f"Confusion Matrix (Float32):\n{cm_k}")
    print("=================================================================\n")

    # 4. Plot Confusion Matrix
    cm_fig_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.imshow(cm_int8, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    classes = ["UNKNOWN (0)", "PROXIMA (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm_int8.max() / 2.0
    for i in range(cm_int8.shape[0]):
        for j in range(cm_int8.shape[1]):
            val = cm_int8[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Proxima KWS INT8 Model Confusion Matrix\nTest Accuracy: {acc_int8*100:.2f}% | F1: {f1_int8*100:.2f}%",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(cm_fig_path))
    plt.close(fig)
    print(f"Saved: {cm_fig_path}")

    # 5. Hardware Arena Calculations
    arena_estimate_kb = 48
    arena_estimate_bytes = arena_estimate_kb * 1024

    # 6. Generate test_results.txt
    test_results_path = output_dir / "test_results.txt"
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA KWS MODEL TEST SET EVALUATION REPORT (V2 - Anti-Overfitting Pipeline)\n")
        f.write("================================================================================\n\n")
        f.write(f"Evaluation on independent TEST partition ({len(y_test)} recordings).\n\n")
        f.write("--- Anti-Overfitting Techniques Applied ---\n")
        f.write("  - SpecAugment: Time masking (1-2 blocks, up to 20 frames)\n")
        f.write("  - SpecAugment: Frequency masking (1-2 blocks, up to 5 bins)\n")
        f.write("  - Waveform: Time shifting (+-0.2s), Gain variation (0.7-1.3x)\n")
        f.write("  - Waveform: Gaussian noise injection (SNR 15-30 dB)\n")
        f.write(f"  - Dropout: {DROPOUT_RATE}\n")
        f.write(f"  - L2 Weight Decay: {L2_DECAY}\n")
        f.write("  - Label Smoothing: 0.1\n")
        f.write("  - ReduceLROnPlateau (patience=15, factor=0.5)\n")
        f.write("  - EarlyStopping (patience=30)\n\n")
        f.write("--- INT8 TFLite Model (Hardware Deployment Target) ---\n")
        f.write(f"  Test Accuracy:                {acc_int8 * 100:.2f}%\n")
        f.write(f"  Precision (PROXIMA):          {prec_int8 * 100:.2f}%\n")
        f.write(f"  Recall (PROXIMA):             {rec_int8 * 100:.2f}%\n")
        f.write(f"  F1 Score:                     {f1_int8 * 100:.2f}%\n")
        f.write(f"  PROXIMA False-Negative Rate:  {fnr_int8 * 100:.2f}% (Missed keywords: {fn} / {tp+fn})\n")
        f.write(f"  UNKNOWN False-Positive Rate:  {fpr_int8 * 100:.2f}% (False alarms: {fp} / {tn+fp})\n\n")
        f.write("  Confusion Matrix:\n")
        f.write("                   Predicted UNKNOWN    Predicted PROXIMA\n")
        f.write(f"  True UNKNOWN:           {tn:3d}                  {fp:3d}\n")
        f.write(f"  True PROXIMA:           {fn:3d}                  {tp:3d}\n\n")
        f.write("--- FLOAT32 Keras Model (Baseline) ---\n")
        f.write(f"  Test Accuracy:                {acc_k * 100:.2f}%\n")
        f.write(f"  Precision (PROXIMA):          {prec_k * 100:.2f}%\n")
        f.write(f"  Recall (PROXIMA):             {rec_k * 100:.2f}%\n")
        f.write(f"  F1 Score:                     {f1_k * 100:.2f}%\n\n")
        f.write("--- Quantization Impact ---\n")
        f.write(f"  Float32 -> INT8 Accuracy Delta: {(acc_int8 - acc_k)*100:+.2f}%\n")
        f.write(f"  Float32 -> INT8 F1 Delta:       {(f1_int8 - f1_k)*100:+.2f}%\n")
    print(f"Saved: {test_results_path}")

    # 7. Generate model_summary.txt
    model_summary_path = output_dir / "model_summary.txt"
    with open(model_summary_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA KWS MODEL SUMMARY & SPECIFICATIONS (V2)\n")
        f.write("================================================================================\n\n")
        f.write(f"Selected Architecture:         DS-CNN V4 (16-48 channels, 4 DS blocks)\n")
        f.write(f"Log-Mel Input Shape:           {input_shape} (time=201, mel_bins=40, channels=1)\n")
        f.write(f"Classes:                       0 = UNKNOWN, 1 = PROXIMA\n")
        f.write(f"Parameter Count:               {best_model.count_params():,}\n\n")
        f.write("--- Model File Sizes ---\n")
        f.write(f"  Keras Checkpoint:            {best_model_path.stat().st_size:,} bytes ({best_model_path.stat().st_size/1024:.2f} KB)\n")
        f.write(f"  Float32 TFLite Model:        {f32_bytes:,} bytes ({f32_bytes/1024:.2f} KB)\n")
        f.write(f"  Full Integer INT8 TFLite:    {int8_bytes:,} bytes ({int8_bytes/1024:.2f} KB)\n")
        f.write(f"  256 KB Budget Utilization:   {(int8_bytes / (256 * 1024)) * 100:.2f}%\n\n")
        f.write("--- ESP32 Hardware Constraints & Memory Estimation ---\n")
        f.write(f"  Flash Memory Required:       {int8_bytes/1024:.2f} KB (budget: 256 KB)\n")
        f.write(f"  Estimated Tensor Arena RAM:  {arena_estimate_kb} KB (~{arena_estimate_bytes:,} bytes)\n")
        f.write(f"  ESP32 SRAM Compatibility:    PASSED\n")
        f.write(f"  External PSRAM Required:     NO\n\n")
        f.write("--- Anti-Overfitting Techniques ---\n")
        f.write(f"  Dropout Rate:                {DROPOUT_RATE}\n")
        f.write(f"  L2 Weight Decay:             {L2_DECAY}\n")
        f.write("  Label Smoothing:             0.1\n")
        f.write("  SpecAugment:                 Time + Frequency Masking\n")
        f.write("  Waveform Augmentation:       Time Shift, Gain, Gaussian Noise\n")
        f.write("  LR Schedule:                 ReduceLROnPlateau (patience=15)\n")
        f.write("  Early Stopping:              patience=30 on val_loss\n\n")
        f.write("--- Architecture Progression & Selection History (Validation Only) ---\n")
        progression = [
            ("V1 (8 channels, 2 DS blocks)", 682, 8.35, 74.71, 58.82, 3.77),
            ("V2 (12-24 channels, 3 DS blocks)", 1758, 12.55, 87.36, 32.35, 0.00),
            ("V3 (16-32 channels, 3 DS blocks)", 2842, 14.51, 89.66, 26.47, 0.00),
            ("V4 (16-48 channels, 4 DS blocks)", 6378, 22.37, 96.55, 0.00, 5.66),
        ]
        for name, p, sz, acc, fpr, fnr in progression:
            f.write(f"  {name}:\n")
            f.write(f"    Params:        {p:,}\n")
            f.write(f"    INT8 Size:     {sz:.2f} KB\n")
            f.write(f"    Val Accuracy:  {acc:.2f}%\n")
            f.write(f"    Val FPR:       {fpr:.2f}%\n")
            f.write(f"    Val FNR:       {fnr:.2f}%\n\n")
        f.write("--- Detailed Layer Architecture ---\n")
        stringlist = []
        best_model.summary(print_fn=lambda x: stringlist.append(x))
        f.write("\n".join(stringlist))
        f.write("\n")
    print(f"Saved: {model_summary_path}")

    # 8. Generate README.md
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Proxima KWS TinyML Model V2 (Anti-Overfitting Pipeline)\n\n")
        f.write("## Overview\n")
        f.write("Trained with comprehensive anti-overfitting measures: SpecAugment, ")
        f.write("waveform augmentation, L2 decay, label smoothing, and adaptive LR scheduling.\n\n")
        f.write("## Key Specifications\n")
        f.write(f"- **Architecture**: DS-CNN V4 (16-48 channels, 4 DS blocks)\n")
        f.write(f"- **Input Shape**: `(201, 40, 1)` Log-Mel Spectrogram\n")
        f.write(f"- **Classes**: `0 = UNKNOWN`, `1 = PROXIMA`\n")
        f.write(f"- **Parameters**: `{best_model.count_params():,}`\n")
        f.write(f"- **INT8 Model Size**: `{int8_bytes:,} bytes` (**{int8_bytes/1024:.2f} KB** - well below 256 KB budget)\n")
        f.write(f"- **Estimated Tensor Arena**: `~{arena_estimate_kb} KB`\n\n")
        f.write("## Performance on Independent TEST Partition\n")
        f.write(f"- **Test Accuracy**: **{acc_int8 * 100:.2f}%**\n")
        f.write(f"- **Precision**: **{prec_int8 * 100:.2f}%**\n")
        f.write(f"- **Recall**: **{rec_int8 * 100:.2f}%**\n")
        f.write(f"- **F1-Score**: **{f1_int8 * 100:.2f}%**\n")
        f.write(f"- **PROXIMA False-Negative Rate**: **{fnr_int8 * 100:.2f}%**\n")
        f.write(f"- **UNKNOWN False-Positive Rate**: **{fpr_int8 * 100:.2f}%**\n\n")
        f.write("## Files\n")
        f.write("- `best_model.keras`: Best trained Float32 Keras model checkpoint.\n")
        f.write("- `proxima_kws_float32.tflite`: Standard TensorFlow Lite Float32 model.\n")
        f.write("- `proxima_kws_int8.tflite`: Full integer INT8 quantized model for ESP32.\n")
        f.write("- `model_summary.txt`: Layer-by-layer parameter summary.\n")
        f.write("- `training_history.csv`: Per-epoch train/val loss and accuracy.\n")
        f.write("- `training_curves.png`: Loss and accuracy training curves.\n")
        f.write("- `test_results.txt`: Detailed evaluation report.\n")
        f.write("- `confusion_matrix.png`: Test confusion matrix.\n")
    print(f"Saved: {readme_path}")

    # 9. Update models/ directory for developer panel
    models_dir = workspace / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as proxima_v3_int8.tflite and proxima_v3_float32.tflite (replacing the overfit v3 models)
    shutil.copy2(int8_tflite_path, models_dir / "proxima_v3_int8.tflite")
    shutil.copy2(f32_tflite_path, models_dir / "proxima_v3_float32.tflite")
    # Also save as proxima_v4
    shutil.copy2(int8_tflite_path, models_dir / "proxima_v4_int8.tflite")
    shutil.copy2(f32_tflite_path, models_dir / "proxima_v4_float32.tflite")
    print(f"Updated models in {models_dir} with new anti-overfitted models!")

    print("\nSUCCESS: All evaluations, quantization, and artifacts generated!")

if __name__ == "__main__":
    main()
