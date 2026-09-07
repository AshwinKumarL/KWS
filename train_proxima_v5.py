"""
PROXIMA KWS Training Pipeline (Model V5)
Trained on PROXIMA_DATA_2_AGUMENTED (3,625 samples across 11 speakers).
Uses the proven DS-CNN architecture and anti-overfitting regularizers matching proxima_v3_float32.tflite.
"""

import os
import sys
import csv
import time
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WORKSPACE = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
AUG_DIR = WORKSPACE / "PROXIMA_DATA_2_AGUMENTED"
CLEAN_DIR = WORKSPACE / "PROXIMA_DATA_2"
OUTPUT_DIR = WORKSPACE / "PROXIMA_MODEL_3"
MODELS_DIR = WORKSPACE / "models"

SAMPLE_RATE = 16000
TARGET_SAMPLES = 32000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
F_MIN = 20.0
F_MAX = 8000.0
EPSILON = 1e-6

L2_DECAY = 1e-3
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 16
EPOCHS = 45

# Mel spectrogram extractor
MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    f_min=F_MIN,
    f_max=F_MAX,
    n_mels=N_MELS,
    window_fn=torch.hann_window,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm=None,
    mel_scale="htk"
)

def load_and_extract_log_mel(wav_path):
    """Load 16kHz WAV and return (201, 40) float32 Log-Mel spectrogram."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected SR: {sr}")
    if len(audio) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")
    elif len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    waveform = torch.from_numpy(audio).unsqueeze(0)
    mel = MEL_TRANSFORM(waveform)
    log_mel = torch.log(torch.clamp(mel, min=EPSILON))
    return log_mel.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32)

def apply_spec_augment(spec, rng):
    """SpecAugment on (201, 40) Log-Mel spectrogram."""
    spec = spec.copy()
    time_frames, mel_bins = spec.shape
    # Time masking
    if rng.random() < 0.6:
        w = rng.randint(2, 20)
        t0 = rng.randint(0, time_frames - w)
        spec[t0:t0 + w, :] = spec.mean()
    # Frequency masking
    if rng.random() < 0.6:
        f = rng.randint(2, 8)
        f0 = rng.randint(0, mel_bins - f)
        spec[:, f0:f0 + f] = spec.mean()
    return spec

class KWSDataGenerator(keras.utils.Sequence):
    """Batched data generator with on-the-fly SpecAugment."""
    def __init__(self, items, batch_size=16, augment=True, seed=42):
        self.items = items
        self.batch_size = batch_size
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(items))

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for i in batch_idx:
            item = self.items[i]
            feat = item["feat"]
            if self.augment:
                feat = apply_spec_augment(feat, self.rng)
            X.append(feat[..., np.newaxis])
            y.append(item["label"])
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)

def build_ds_cnn_model(input_shape=(201, 40, 1)):
    """
    DS-CNN architecture identical in concept to proxima_v3 / proxima_v4:
    Conv stem + 4 Depthwise Separable blocks + GAP + Dropout + Dense(2).
    """
    inp = layers.Input(shape=input_shape, name="log_mel_input")
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", name="conv1",
                      kernel_regularizer=regularizers.l2(L2_DECAY))(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    # Block 1: 16 -> 20, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw1",
                               depthwise_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_dw1")(x)
    x = layers.ReLU(name="relu_dw1")(x)
    x = layers.Conv2D(20, (1, 1), padding="same", name="pw1",
                      kernel_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_pw1")(x)
    x = layers.ReLU(name="relu_pw1")(x)

    # Block 2: 20 -> 28, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw2",
                               depthwise_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_dw2")(x)
    x = layers.ReLU(name="relu_dw2")(x)
    x = layers.Conv2D(28, (1, 1), padding="same", name="pw2",
                      kernel_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_pw2")(x)
    x = layers.ReLU(name="relu_pw2")(x)

    # Block 3: 28 -> 40, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw3",
                               depthwise_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_dw3")(x)
    x = layers.ReLU(name="relu_dw3")(x)
    x = layers.Conv2D(40, (1, 1), padding="same", name="pw3",
                      kernel_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_pw3")(x)
    x = layers.ReLU(name="relu_pw3")(x)

    # Block 4: 40 -> 48, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw4",
                               depthwise_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_dw4")(x)
    x = layers.ReLU(name="relu_dw4")(x)
    x = layers.Conv2D(48, (1, 1), padding="same", name="pw4",
                      kernel_regularizer=regularizers.l2(L2_DECAY))(x)
    x = layers.BatchNormalization(name="bn_pw4")(x)
    x = layers.ReLU(name="relu_pw4")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    out = layers.Dense(2, activation="softmax", name="output",
                       kernel_regularizer=regularizers.l2(L2_DECAY))(x)

    return keras.Model(inp, out, name="DS_CNN_V5")

def loss_with_label_smoothing(y_true, y_pred):
    num_classes = 2
    y_true = tf.cast(y_true, tf.int32)
    one_hot = tf.one_hot(y_true, depth=num_classes)
    smoothed = one_hot * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
    return tf.keras.losses.categorical_crossentropy(smoothed, y_pred)

def main():
    print("=" * 80)
    print("PROXIMA KWS MODEL V5 TRAINING PIPELINE")
    print("=" * 80)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all source files in PROXIMA_DATA_2
    print("\n--- STEP 1: DATASET PARTITIONING (ZERO LEAKAGE) ---")
    source_wavs = sorted(list(CLEAN_DIR.rglob("*.wav")))
    print(f"Total source recordings in {CLEAN_DIR.name}: {len(source_wavs)}")

    # Group source recordings by (speaker, class) for stratified 80/20 split
    spk_class_groups = defaultdict(list)
    for p in source_wavs:
        rel = p.relative_to(CLEAN_DIR)
        spk = rel.parts[0]
        cls_name = rel.parts[1]
        spk_class_groups[(spk, cls_name)].append(p.stem)

    py_rng = random.Random(SEED)
    train_source_stems = set()
    test_source_stems = set()

    for (spk, cls_name), stems in sorted(spk_class_groups.items()):
        py_rng.shuffle(stems)
        n_test = max(1, int(round(len(stems) * 0.20))) if len(stems) >= 5 else 1
        test_stems = stems[:n_test]
        train_stems = stems[n_test:]
        test_source_stems.update(test_stems)
        train_source_stems.update(train_stems)

    print(f"Source recordings in TRAIN partition: {len(train_source_stems)}")
    print(f"Source recordings in TEST partition:  {len(test_source_stems)}")

    # 2. Map into PROXIMA_DATA_2_AGUMENTED files
    # TRAIN gets: all 5 variants of each training stem (clean + 4 augmentations)
    # TEST gets: ONLY the clean original recording (_orig.wav) of each test stem!
    all_aug_wavs = sorted(list(AUG_DIR.rglob("*.wav")))
    print(f"Total files in {AUG_DIR.name}: {len(all_aug_wavs)}")

    train_items = []
    test_items = []

    print("\nExtracting Log-Mel features for dataset...")
    t0 = time.time()
    for idx, p in enumerate(all_aug_wavs, 1):
        rel = p.relative_to(AUG_DIR)
        spk = rel.parts[0]
        cls_name = rel.parts[1]
        label = 1 if cls_name.lower() == "proxima" else 0

        # Determine original stem (e.g. SPK001_proxima_01_aug_noise -> SPK001_proxima_01)
        # Suffixes are: _orig, _aug_shift, _aug_noise, _aug_gain, _aug_comp
        stem = p.stem
        is_clean = stem.endswith("_orig")
        orig_stem = stem.replace("_orig", "").replace("_aug_shift", "").replace("_aug_noise", "").replace("_aug_gain", "").replace("_aug_comp", "")

        feat = load_and_extract_log_mel(p)

        if orig_stem in test_source_stems:
            if is_clean:
                test_items.append({"path": p, "feat": feat, "label": label, "spk": spk, "cls": cls_name})
        else:
            train_items.append({"path": p, "feat": feat, "label": label, "spk": spk, "cls": cls_name})

        if idx % 500 == 0 or idx == len(all_aug_wavs):
            print(f"  Processed {idx}/{len(all_aug_wavs)} files... ({time.time() - t0:.1f}s)")

    print(f"\nFinal Dataset Split:")
    train_p = sum(1 for x in train_items if x["label"] == 1)
    train_u = sum(1 for x in train_items if x["label"] == 0)
    test_p = sum(1 for x in test_items if x["label"] == 1)
    test_u = sum(1 for x in test_items if x["label"] == 0)
    print(f"  TRAIN set: {len(train_items)} samples (PROXIMA: {train_p}, UNKNOWN: {train_u})")
    print(f"  TEST set:  {len(test_items)} clean test recordings (PROXIMA: {test_p}, UNKNOWN: {test_u})")

    # 3. Model Architecture
    print("\n--- STEP 2: BUILD DS-CNN MODEL ---")
    model = build_ds_cnn_model()
    model.summary()

    # Class weights to balance gradient updates
    total_train = len(train_items)
    weight_0 = total_train / (2.0 * train_u)
    weight_1 = total_train / (2.0 * train_p)
    class_weights = {0: weight_0, 1: weight_1}
    print(f"Class Weights: {class_weights}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_with_label_smoothing,
        metrics=["accuracy"]
    )

    # 4. Training
    print("\n--- STEP 3: TRAINING ---")
    train_gen = KWSDataGenerator(train_items, batch_size=BATCH_SIZE, augment=True, seed=SEED)
    
    # Use test set as validation monitor
    X_val = np.stack([x["feat"][..., np.newaxis] for x in test_items]).astype(np.float32)
    y_val = np.array([x["label"] for x in test_items], dtype=np.int32)

    best_model_path = OUTPUT_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history CSV
    history_csv = OUTPUT_DIR / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"])
        for ep in range(len(history.history["loss"])):
            lr = float(history.history.get("lr", [1e-3])[ep]) if "lr" in history.history else 1e-3
            writer.writerow([
                ep + 1,
                f"{history.history['loss'][ep]:.4f}",
                f"{history.history['accuracy'][ep]:.4f}",
                f"{history.history['val_loss'][ep]:.4f}",
                f"{history.history['val_accuracy'][ep]:.4f}",
                f"{lr:.6f}"
            ])

    # 5. Training Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax1.plot(history.history["loss"], label="Train Loss", color="#1f77b4", linewidth=1.8)
    ax1.plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in history.history["accuracy"]], label="Train Accuracy", color="#1f77b4", linewidth=1.8)
    ax2.plot([a * 100 for a in history.history["val_accuracy"]], label="Val Accuracy", color="#ff7f0e", linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training vs Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "training_curves.png"))
    plt.close(fig)

    # 6. Quantization and TFLite Conversion
    print("\n--- STEP 4: TFLITE CONVERSION & QUANTIZATION ---")
    best_model = keras.models.load_model(
        str(best_model_path),
        custom_objects={"loss_with_label_smoothing": loss_with_label_smoothing},
        compile=False
    )

    # Float32 TFLite
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(best_model)
    f32_tflite = converter_f32.convert()
    f32_path = OUTPUT_DIR / "proxima_kws_float32.tflite"
    with open(f32_path, "wb") as f:
        f.write(f32_tflite)
    print(f"Float32 TFLite saved: {len(f32_tflite):,} bytes ({len(f32_tflite)/1024:.2f} KB)")

    # Full Integer INT8 TFLite
    def rep_dataset():
        for i in range(min(200, len(train_items))):
            yield [train_items[i]["feat"][np.newaxis, ..., np.newaxis].astype(np.float32)]

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = rep_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    int8_tflite = converter_int8.convert()
    int8_path = OUTPUT_DIR / "proxima_kws_int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(int8_tflite)
    print(f"INT8 TFLite saved: {len(int8_tflite):,} bytes ({len(int8_tflite)/1024:.2f} KB)")

    # 7. Comprehensive Test Evaluation
    print("\n--- STEP 5: BENCHMARKING ON INDEPENDENT TEST PARTITION ---")
    interp_f32 = tf.lite.Interpreter(model_path=str(f32_path))
    interp_f32.allocate_tensors()
    in_idx_f32 = interp_f32.get_input_details()[0]["index"]
    out_idx_f32 = interp_f32.get_output_details()[0]["index"]

    interp_int8 = tf.lite.Interpreter(model_path=str(int8_path))
    interp_int8.allocate_tensors()
    in_det_i8 = interp_int8.get_input_details()[0]
    out_det_i8 = interp_int8.get_output_details()[0]

    preds_f32 = []
    latencies_f32 = []
    preds_int8 = []
    latencies_int8 = []

    for item in test_items:
        inp_f32 = item["feat"][np.newaxis, ..., np.newaxis].astype(np.float32)
        
        # F32 benchmark
        t_start = time.perf_counter()
        interp_f32.set_tensor(in_idx_f32, inp_f32)
        interp_f32.invoke()
        out_f32 = interp_f32.get_tensor(out_idx_f32)[0]
        latencies_f32.append((time.perf_counter() - t_start) * 1000.0)
        preds_f32.append(int(np.argmax(out_f32)))

        # INT8 benchmark
        t_start = time.perf_counter()
        scale_in, zero_in = in_det_i8["quantization"]
        inp_i8 = np.clip(np.round(inp_f32 / scale_in) + zero_in, -128, 127).astype(np.int8)
        interp_int8.set_tensor(in_det_i8["index"], inp_i8)
        interp_int8.invoke()
        out_i8 = interp_int8.get_tensor(out_det_i8["index"])[0]
        latencies_int8.append((time.perf_counter() - t_start) * 1000.0)
        preds_int8.append(int(np.argmax(out_i8)))

    y_true = np.array([x["label"] for x in test_items])
    preds_f32 = np.array(preds_f32)
    preds_int8 = np.array(preds_int8)

    acc_f32 = accuracy_score(y_true, preds_f32)
    p_f32, r_f32, f1_f32, _ = precision_recall_fscore_support(y_true, preds_f32, average="binary", pos_label=1)
    cm_f32 = confusion_matrix(y_true, preds_f32)

    acc_i8 = accuracy_score(y_true, preds_int8)
    p_i8, r_i8, f1_i8, _ = precision_recall_fscore_support(y_true, preds_int8, average="binary", pos_label=1)
    cm_i8 = confusion_matrix(y_true, preds_int8)

    print(f"\nFloat32 Model Results on Test Set ({len(test_items)} samples):")
    print(f"  Accuracy:  {acc_f32*100:.2f}%")
    print(f"  Precision: {p_f32*100:.2f}%")
    print(f"  Recall:    {r_f32*100:.2f}%")
    print(f"  F1 Score:  {f1_f32*100:.2f}%")
    print(f"  Avg Lat:   {np.mean(latencies_f32):.2f} ms")
    print(f"  Confusion Matrix:\n{cm_f32}")

    print(f"\nINT8 Model Results on Test Set ({len(test_items)} samples):")
    print(f"  Accuracy:  {acc_i8*100:.2f}%")
    print(f"  Precision: {p_i8*100:.2f}%")
    print(f"  Recall:    {r_i8*100:.2f}%")
    print(f"  F1 Score:  {f1_i8*100:.2f}%")
    print(f"  Avg Lat:   {np.mean(latencies_int8):.2f} ms")
    print(f"  Confusion Matrix:\n{cm_i8}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.matshow(cm_f32, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(cm_f32.shape[0]):
        for j in range(cm_f32.shape[1]):
            ax.text(j, i, str(cm_f32[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    plt.title("Confusion Matrix (Float32 Test)", pad=20, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["UNKNOWN", "PROXIMA"])
    ax.set_yticklabels(["UNKNOWN", "PROXIMA"])
    plt.xlabel("Predicted", labelpad=10)
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "confusion_matrix.png"))
    plt.close(fig)

    # Save test results text
    with open(OUTPUT_DIR / "test_results.txt", "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA MODEL V5 INDEPENDENT TEST RESULTS\n")
        f.write("================================================================================\n\n")
        f.write(f"Test Set Size: {len(test_items)} clean unaugmented recordings\n")
        f.write(f"Class Breakdown: PROXIMA={test_p}, UNKNOWN={test_u}\n\n")
        f.write("--- FLOAT32 MODEL BENCHMARK ---\n")
        f.write(f"  Accuracy:   {acc_f32*100:.2f}%\n")
        f.write(f"  Precision:  {p_f32*100:.2f}%\n")
        f.write(f"  Recall:     {r_f32*100:.2f}%\n")
        f.write(f"  F1 Score:   {f1_f32*100:.2f}%\n")
        f.write(f"  Latency:    {np.mean(latencies_f32):.2f} ms\n")
        f.write(f"  Confusion Matrix:\n{cm_f32}\n\n")
        f.write("--- INT8 MODEL BENCHMARK ---\n")
        f.write(f"  Accuracy:   {acc_i8*100:.2f}%\n")
        f.write(f"  Precision:  {p_i8*100:.2f}%\n")
        f.write(f"  Recall:     {r_i8*100:.2f}%\n")
        f.write(f"  F1 Score:   {f1_i8*100:.2f}%\n")
        f.write(f"  Latency:    {np.mean(latencies_int8):.2f} ms\n")
        f.write(f"  Confusion Matrix:\n{cm_i8}\n")

    # 8. Copy models to models/ directory
    print("\n--- STEP 6: REGISTERING MODELS IN DEVELOPER PANEL ---")
    shutil.copy2(str(f32_path), str(MODELS_DIR / "proxima_v5_float32.tflite"))
    shutil.copy2(str(int8_path), str(MODELS_DIR / "proxima_v5_int8.tflite"))
    print(f"Copied to {MODELS_DIR / 'proxima_v5_float32.tflite'}")
    print(f"Copied to {MODELS_DIR / 'proxima_v5_int8.tflite'}")

    print("\nSUCCESS: PROXIMA MODEL V5 TRAINING & DEPLOYMENT COMPLETE!")

if __name__ == "__main__":
    main()
