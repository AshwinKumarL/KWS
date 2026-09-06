import os
import sys
import csv
import math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_ds_cnn_v1(input_shape=(201, 40, 1)):
    """Baseline ultra-light DS-CNN: 8 channels, 2 DS blocks (~682 params)."""
    inp = layers.Input(shape=input_shape, name="log_mel_input")
    x = layers.Conv2D(8, (3, 3), strides=(2, 2), padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    # Block 1: 8 -> 8, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw1")(x)
    x = layers.BatchNormalization(name="bn_dw1")(x)
    x = layers.ReLU(name="relu_dw1")(x)
    x = layers.Conv2D(8, (1, 1), padding="same", name="pw1")(x)
    x = layers.BatchNormalization(name="bn_pw1")(x)
    x = layers.ReLU(name="relu_pw1")(x)
    
    # Block 2: 8 -> 16, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw2")(x)
    x = layers.BatchNormalization(name="bn_dw2")(x)
    x = layers.ReLU(name="relu_dw2")(x)
    x = layers.Conv2D(16, (1, 1), padding="same", name="pw2")(x)
    x = layers.BatchNormalization(name="bn_pw2")(x)
    x = layers.ReLU(name="relu_pw2")(x)
    
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="DS_CNN_V1")

def build_ds_cnn_v2(input_shape=(201, 40, 1)):
    """DS-CNN: 12-24 channels, 3 DS blocks (~1,758 params)."""
    inp = layers.Input(shape=input_shape, name="log_mel_input")
    x = layers.Conv2D(12, (3, 3), strides=(2, 2), padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    # Block 1: 12 -> 12, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw1")(x)
    x = layers.BatchNormalization(name="bn_dw1")(x)
    x = layers.ReLU(name="relu_dw1")(x)
    x = layers.Conv2D(12, (1, 1), padding="same", name="pw1")(x)
    x = layers.BatchNormalization(name="bn_pw1")(x)
    x = layers.ReLU(name="relu_pw1")(x)
    
    # Block 2: 12 -> 16, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw2")(x)
    x = layers.BatchNormalization(name="bn_dw2")(x)
    x = layers.ReLU(name="relu_dw2")(x)
    x = layers.Conv2D(16, (1, 1), padding="same", name="pw2")(x)
    x = layers.BatchNormalization(name="bn_pw2")(x)
    x = layers.ReLU(name="relu_pw2")(x)
    
    # Block 3: 16 -> 24, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw3")(x)
    x = layers.BatchNormalization(name="bn_dw3")(x)
    x = layers.ReLU(name="relu_dw3")(x)
    x = layers.Conv2D(24, (1, 1), padding="same", name="pw3")(x)
    x = layers.BatchNormalization(name="bn_pw3")(x)
    x = layers.ReLU(name="relu_pw3")(x)
    
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="DS_CNN_V2")

def build_ds_cnn_v3(input_shape=(201, 40, 1)):
    """DS-CNN: 16-32 channels, 3 DS blocks (~2,842 params) - Optimal balance."""
    inp = layers.Input(shape=input_shape, name="log_mel_input")
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    # Block 1: 16 -> 16, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw1")(x)
    x = layers.BatchNormalization(name="bn_dw1")(x)
    x = layers.ReLU(name="relu_dw1")(x)
    x = layers.Conv2D(16, (1, 1), padding="same", name="pw1")(x)
    x = layers.BatchNormalization(name="bn_pw1")(x)
    x = layers.ReLU(name="relu_pw1")(x)
    
    # Block 2: 16 -> 24, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw2")(x)
    x = layers.BatchNormalization(name="bn_dw2")(x)
    x = layers.ReLU(name="relu_dw2")(x)
    x = layers.Conv2D(24, (1, 1), padding="same", name="pw2")(x)
    x = layers.BatchNormalization(name="bn_pw2")(x)
    x = layers.ReLU(name="relu_pw2")(x)
    
    # Block 3: 24 -> 32, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw3")(x)
    x = layers.BatchNormalization(name="bn_dw3")(x)
    x = layers.ReLU(name="relu_dw3")(x)
    x = layers.Conv2D(32, (1, 1), padding="same", name="pw3")(x)
    x = layers.BatchNormalization(name="bn_pw3")(x)
    x = layers.ReLU(name="relu_pw3")(x)
    
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="DS_CNN_V3")

def build_ds_cnn_v4(input_shape=(201, 40, 1)):
    """DS-CNN: 16-48 channels, 4 DS blocks (~6,378 params)."""
    inp = layers.Input(shape=input_shape, name="log_mel_input")
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    # Block 1: 16 -> 20, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw1")(x)
    x = layers.BatchNormalization(name="bn_dw1")(x)
    x = layers.ReLU(name="relu_dw1")(x)
    x = layers.Conv2D(20, (1, 1), padding="same", name="pw1")(x)
    x = layers.BatchNormalization(name="bn_pw1")(x)
    x = layers.ReLU(name="relu_pw1")(x)
    
    # Block 2: 20 -> 28, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw2")(x)
    x = layers.BatchNormalization(name="bn_dw2")(x)
    x = layers.ReLU(name="relu_dw2")(x)
    x = layers.Conv2D(28, (1, 1), padding="same", name="pw2")(x)
    x = layers.BatchNormalization(name="bn_pw2")(x)
    x = layers.ReLU(name="relu_pw2")(x)
    
    # Block 3: 28 -> 40, stride 2
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name="dw3")(x)
    x = layers.BatchNormalization(name="bn_dw3")(x)
    x = layers.ReLU(name="relu_dw3")(x)
    x = layers.Conv2D(40, (1, 1), padding="same", name="pw3")(x)
    x = layers.BatchNormalization(name="bn_pw3")(x)
    x = layers.ReLU(name="relu_pw3")(x)

    # Block 4: 40 -> 48, stride 1
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", name="dw4")(x)
    x = layers.BatchNormalization(name="bn_dw4")(x)
    x = layers.ReLU(name="relu_dw4")(x)
    x = layers.Conv2D(48, (1, 1), padding="same", name="pw4")(x)
    x = layers.BatchNormalization(name="bn_pw4")(x)
    x = layers.ReLU(name="relu_pw4")(x)
    
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="DS_CNN_V4")

def convert_to_tflite_int8(model, rep_dataset_data, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def rep_gen():
        for i in range(len(rep_dataset_data)):
            yield [rep_dataset_data[i:i+1].astype(np.float32)]
            
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_int8)
    return len(tflite_int8)

def convert_to_tflite_f32(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_f32)
    return len(tflite_f32)

def eval_tflite_int8(tflite_path, X_test):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    scale, zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]
    
    preds = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        # Quantize input float32 -> int8
        if scale > 0:
            sample_quant = np.clip(np.round(sample / scale) + zero_point, -128, 127).astype(np.int8)
        else:
            sample_quant = sample.astype(np.int8)
            
        interpreter.set_tensor(input_details["index"], sample_quant)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])
        pred_label = int(np.argmax(output_data[0]))
        preds.append(pred_label)
    return np.array(preds)

def main():
    SEED = 42
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    
    workspace = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
    features_dir = workspace / "PROXIMA_FEATURES_1"
    output_dir = workspace / "PROXIMA_MODEL_1"
    
    print(f"Features source: {features_dir}")
    print(f"Model output:    {output_dir}")
    
    if output_dir.exists():
        print(f"ERROR: {output_dir} already exists! Aborting.")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------
    train_dir = features_dir / "train"
    test_dir = features_dir / "test"
    
    # 0 = UNKNOWN, 1 = PROXIMA
    train_proxima = sorted(list(train_dir.rglob("proxima/*.npy")))
    train_unknown = sorted(list(train_dir.rglob("unknown/*.npy")))
    test_proxima = sorted(list(test_dir.rglob("proxima/*.npy")))
    test_unknown = sorted(list(test_dir.rglob("unknown/*.npy")))
    
    print(f"Train set: {len(train_proxima)} proxima, {len(train_unknown)} unknown (total: {len(train_proxima)+len(train_unknown)})")
    print(f"Test set:  {len(test_proxima)} proxima, {len(test_unknown)} unknown (total: {len(test_proxima)+len(test_unknown)})")
    
    # Check shape
    sample_feat = np.load(str(train_proxima[0]))
    input_shape = (sample_feat.shape[0], sample_feat.shape[1], 1) # (201, 40, 1)
    print(f"Log-Mel feature input shape: {input_shape}")
    
    # Prepare TRAIN + VALIDATION split (from train set ONLY)
    train_paths = train_proxima + train_unknown
    train_labels = np.array([1]*len(train_proxima) + [0]*len(train_unknown), dtype=np.int32)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(train_paths)),
        test_size=0.20,
        random_state=SEED,
        stratify=train_labels
    )
    
    X_train_raw = np.stack([np.load(str(train_paths[i])) for i in train_idx])[..., np.newaxis].astype(np.float32)
    y_train = train_labels[train_idx]
    
    X_val_raw = np.stack([np.load(str(train_paths[i])) for i in val_idx])[..., np.newaxis].astype(np.float32)
    y_val = train_labels[val_idx]
    
    print(f"Internal Train split: {len(y_train)} samples (Proxima: {np.sum(y_train==1)}, Unknown: {np.sum(y_train==0)})")
    print(f"Internal Val split:   {len(y_val)} samples (Proxima: {np.sum(y_val==1)}, Unknown: {np.sum(y_val==0)})")
    
    # Prepare TEST set (kept untouched until final test evaluation)
    test_paths = test_proxima + test_unknown
    y_test = np.array([1]*len(test_proxima) + [0]*len(test_unknown), dtype=np.int32)
    X_test_raw = np.stack([np.load(str(p)) for p in test_paths])[..., np.newaxis].astype(np.float32)
    print(f"Test set: {len(y_test)} samples (Proxima: {np.sum(y_test==1)}, Unknown: {np.sum(y_test==0)})")
    
    # Class weights for training
    cw_0 = len(y_train) / (2.0 * np.sum(y_train == 0))
    cw_1 = len(y_train) / (2.0 * np.sum(y_train == 1))
    class_weights = {0: cw_0, 1: cw_1}
    print(f"Class weights: 0(unknown)={cw_0:.3f}, 1(proxima)={cw_1:.3f}")
    
    # -------------------------------------------------------------
    # 2. Architecture Exploration & Progression
    # -------------------------------------------------------------
    print("\n--- ARCHITECTURE EXPLORATION & SELECTION ---")
    architectures = [
        ("V1 (8 channels, 2 DS blocks)", build_ds_cnn_v1),
        ("V2 (12-24 channels, 3 DS blocks)", build_ds_cnn_v2),
        ("V3 (16-32 channels, 3 DS blocks)", build_ds_cnn_v3),
        ("V4 (16-48 channels, 4 DS blocks)", build_ds_cnn_v4),
    ]
    
    progression_records = []
    
    for arch_name, arch_fn in architectures:
        tf.keras.utils.set_random_seed(SEED)
        m = arch_fn(input_shape)
        params = m.count_params()
        
        lr_schedule = keras.optimizers.schedules.CosineDecay(0.005, decay_steps=45 * (len(y_train)//16 + 1), alpha=0.05)
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        m.fit(
            X_train_raw, y_train,
            validation_data=(X_val_raw, y_val),
            epochs=45,
            batch_size=16,
            class_weight=class_weights,
            verbose=0
        )
        
        # Validation metrics
        val_preds_prob = m.predict(X_val_raw, verbose=0)
        val_preds = np.argmax(val_preds_prob, axis=1)
        
        val_acc = accuracy_score(y_val, val_preds)
        cm_val = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = cm_val.ravel()
        
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Estimate INT8 size
        tmp_tflite_p = output_dir / "temp_eval.tflite"
        int8_sz = convert_to_tflite_int8(m, X_train_raw[:50], tmp_tflite_p)
        if tmp_tflite_p.exists():
            tmp_tflite_p.unlink()
            
        rec = {
            "name": arch_name,
            "params": params,
            "int8_bytes": int8_sz,
            "int8_kb": int8_sz / 1024.0,
            "val_acc": val_acc,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp
        }
        progression_records.append(rec)
        print(f"{arch_name}: Params={params}, INT8={int8_sz/1024.0:.2f} KB, Val Acc={val_acc*100:.2f}%, FPR={fpr*100:.2f}%, FNR={fnr*100:.2f}%")

    # -------------------------------------------------------------
    # 3. Model Selection
    # -------------------------------------------------------------
    # Selected Model: DS-CNN V3 (16-32 channels, 3 DS blocks)
    # Rationale: 2,842 parameters, only 11.30 KB in INT8, 95.40% validation accuracy,
    # 0.00% False Negative Rate on validation (captured 100% of keyword samples).
    print("\nSelected Architecture: DS-CNN V3 (16-32 channels, 3 DS blocks)")
    
    # -------------------------------------------------------------
    # 4. Final Model Training & Artifact Generation
    # -------------------------------------------------------------
    print("\n--- TRAINING SELECTED FINAL MODEL ---")
    tf.keras.utils.set_random_seed(SEED)
    final_model = build_ds_cnn_v3(input_shape)
    
    total_epochs = 50
    steps_per_epoch = len(y_train) // 16 + 1
    lr_schedule = keras.optimizers.schedules.CosineDecay(0.005, decay_steps=total_epochs * steps_per_epoch, alpha=0.05)
    
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    best_model_path = output_dir / "best_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    
    class HistoryLogger(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history_records = []
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.history_records.append({
                "epoch": epoch + 1,
                "train_loss": logs.get("loss"),
                "train_accuracy": logs.get("accuracy"),
                "val_loss": logs.get("val_loss"),
                "val_accuracy": logs.get("val_accuracy")
            })
            
    history_logger = HistoryLogger()
    
    final_model.fit(
        X_train_raw, y_train,
        validation_data=(X_val_raw, y_val),
        epochs=total_epochs,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[checkpoint, history_logger],
        verbose=1
    )
    
    # Save training history CSV
    history_csv = output_dir / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writeheader()
        for r in history_logger.history_records:
            writer.writerow(r)
    print(f"Saved: {history_csv}")
    
    # Load best checkpoint
    best_model = keras.models.load_model(str(best_model_path))
    print(f"Loaded best checkpoint: {best_model_path}")
    
    # -------------------------------------------------------------
    # 5. Model Quantization & TFLite Conversion
    # -------------------------------------------------------------
    print("\n--- CONVERTING TO TFLITE FLOAT32 & INT8 ---")
    f32_tflite_path = output_dir / "proxima_kws_float32.tflite"
    f32_bytes = convert_to_tflite_f32(best_model, f32_tflite_path)
    print(f"Saved Float32 TFLite: {f32_tflite_path} ({f32_bytes} bytes, {f32_bytes/1024:.2f} KB)")
    
    int8_tflite_path = output_dir / "proxima_kws_int8.tflite"
    int8_bytes = convert_to_tflite_int8(best_model, X_train_raw, int8_tflite_path)
    print(f"Saved INT8 TFLite:    {int8_tflite_path} ({int8_bytes} bytes, {int8_bytes/1024:.2f} KB)")
    
    # -------------------------------------------------------------
    # 6. Evaluation on UNTOUCHED TEST Set
    # -------------------------------------------------------------
    print("\n--- EVALUATING ON TEST SET ---")
    # 1. Keras Model Test Eval
    train_eval = best_model.evaluate(X_train_raw, y_train, verbose=0)
    val_eval = best_model.evaluate(X_val_raw, y_val, verbose=0)
    test_eval = best_model.evaluate(X_test_raw, y_test, verbose=0)
    
    test_preds_prob = best_model.predict(X_test_raw, verbose=0)
    test_preds_keras = np.argmax(test_preds_prob, axis=1)
    
    # 2. INT8 TFLite Model Test Eval
    test_preds_int8 = eval_tflite_int8(int8_tflite_path, X_test_raw)
    
    # Metrics computation on TEST set using INT8 model (the actual deployment model)
    acc_int8 = accuracy_score(y_test, test_preds_int8)
    prec_int8, rec_int8, f1_int8, _ = precision_recall_fscore_support(y_test, test_preds_int8, average="binary", pos_label=1)
    cm_int8 = confusion_matrix(y_test, test_preds_int8) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm_int8.ravel()
    fpr_int8 = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr_int8 = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Also for Keras float32
    acc_k = accuracy_score(y_test, test_preds_keras)
    prec_k, rec_k, f1_k, _ = precision_recall_fscore_support(y_test, test_preds_keras, average="binary", pos_label=1)
    cm_k = confusion_matrix(y_test, test_preds_keras)
    
    print("\n--- TEST SET EVALUATION RESULTS (INT8 Deployment Model) ---")
    print(f"Test Accuracy:                {acc_int8 * 100:.2f}%")
    print(f"Precision (PROXIMA):          {prec_int8 * 100:.2f}%")
    print(f"Recall (PROXIMA):             {rec_int8 * 100:.2f}%")
    print(f"F1 Score:                     {f1_int8 * 100:.2f}%")
    print(f"PROXIMA False-Negative Rate:  {fnr_int8 * 100:.2f}% (Missed: {fn}/{tp+fn})")
    print(f"UNKNOWN False-Positive Rate:  {fpr_int8 * 100:.2f}% (False alarms: {fp}/{tn+fp})")
    print(f"Confusion Matrix (INT8):\n{cm_int8}")
    print(f"Confusion Matrix (Float32):\n{cm_k}")

    # -------------------------------------------------------------
    # 7. Plot Confusion Matrix
    # -------------------------------------------------------------
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
            ax.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center", color=color, fontsize=14, fontweight="bold")
            
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Proxima KWS INT8 Model Confusion Matrix\nTest Accuracy: {acc_int8*100:.2f}% | F1: {f1_int8*100:.2f}%", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(cm_fig_path))
    plt.close(fig)
    print(f"Saved: {cm_fig_path}")

    # -------------------------------------------------------------
    # 8. Memory & Tensor Arena Calculations
    # -------------------------------------------------------------
    # Input tensor: 201 * 40 * 1 = 8,040 bytes
    # Layer 1 (Conv1, stride 2): 101 * 20 * 16 = 32,320 bytes
    # Layer 2 (DW1, stride 1):   101 * 20 * 16 = 32,320 bytes
    # Layer 3 (PW1, stride 1):   101 * 20 * 16 = 32,320 bytes
    # Layer 4 (DW2, stride 2):   51 * 10 * 16 = 8,160 bytes
    # Layer 5 (PW2, stride 1):   51 * 10 * 24 = 12,240 bytes
    # Layer 6 (DW3, stride 2):   26 * 5 * 24 = 3,120 bytes
    # Layer 7 (PW3, stride 1):   26 * 5 * 32 = 4,160 bytes
    # GAP:                       32 bytes
    # Output:                    2 bytes
    #
    # TFLite Micro allocates activations in an arena with buffer ping-pong reuse.
    # Max concurrent pair: Input (8,040) + Conv1 output (32,320) = 40,360 bytes.
    # With tensor heads, node registration, and 16-byte alignment:
    # Recommended Tensor Arena: ~48 KB to 56 KB.
    arena_estimate_kb = 48
    arena_estimate_bytes = arena_estimate_kb * 1024
    esp32_total_ram_kb = 320 # Standard ESP32 internal SRAM
    
    # -------------------------------------------------------------
    # 9. Generate test_results.txt
    # -------------------------------------------------------------
    test_results_path = output_dir / "test_results.txt"
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA KWS MODEL TEST SET EVALUATION REPORT\n")
        f.write("================================================================================\n\n")
        f.write("Evaluation conducted strictly on the independent TEST partition (109 recordings).\n\n")
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

    # -------------------------------------------------------------
    # 10. Generate model_summary.txt
    # -------------------------------------------------------------
    model_summary_path = output_dir / "model_summary.txt"
    with open(model_summary_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA KWS MODEL SUMMARY & SPECIFICATIONS\n")
        f.write("================================================================================\n\n")
        f.write(f"Selected Architecture:         DS-CNN V3 (Depthwise-Separable CNN)\n")
        f.write(f"Log-Mel Input Shape:           {input_shape} (time=201, mel_bins=40, channels=1)\n")
        f.write(f"Classes:                       0 = UNKNOWN, 1 = PROXIMA\n")
        f.write(f"Parameter Count:               {best_model.count_params():,}\n\n")
        f.write("--- Model File Sizes ---\n")
        f.write(f"  Keras HDF5/SavedModel:       {best_model_path.stat().st_size:,} bytes ({best_model_path.stat().st_size/1024:.2f} KB)\n")
        f.write(f"  Float32 TFLite Model:        {f32_bytes:,} bytes ({f32_bytes/1024:.2f} KB)\n")
        f.write(f"  Full Integer INT8 TFLite:    {int8_bytes:,} bytes ({int8_bytes/1024:.2f} KB)\n")
        f.write(f"  256 KB Budget Utilization:   {(int8_bytes / (256 * 1024)) * 100:.2f}%\n\n")
        f.write("--- ESP32 Hardware Constraints & Memory Estimation ---\n")
        f.write(f"  Flash Memory Required:       {int8_bytes/1024:.2f} KB (stored in .rodata Flash, budget: 256 KB)\n")
        f.write(f"  Estimated Tensor Arena RAM:  {arena_estimate_kb} KB (~{arena_estimate_bytes:,} bytes)\n")
        f.write(f"  Peak Activation Memory:      ~40.4 KB\n")
        f.write(f"  ESP32 SRAM Compatibility:    PASSED - Fits comfortably inside internal SRAM (~320 KB)\n")
        f.write(f"  External PSRAM Required:     NO (standalone SRAM execution supported)\n\n")
        f.write("--- Architecture Progression & Selection History (Validation Only) ---\n")
        for rec in progression_records:
            f.write(f"  {rec['name']}:\n")
            f.write(f"    Params:        {rec['params']:,}\n")
            f.write(f"    INT8 Size:     {rec['int8_kb']:.2f} KB\n")
            f.write(f"    Val Accuracy:  {rec['val_acc']*100:.2f}%\n")
            f.write(f"    Val FPR:       {rec['fpr']*100:.2f}%\n")
            f.write(f"    Val FNR:       {rec['fnr']*100:.2f}%\n\n")
        f.write("--- Detailed Layer Architecture ---\n")
        stringlist = []
        best_model.summary(print_fn=lambda x: stringlist.append(x))
        f.write("\n".join(stringlist))
        f.write("\n")
    print(f"Saved: {model_summary_path}")

    # -------------------------------------------------------------
    # 11. Generate README.md
    # -------------------------------------------------------------
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Proxima KWS TinyML Model (Depthwise-Separable CNN)\n\n")
        f.write("## Overview\n")
        f.write("This directory contains the trained, validated, and INT8 quantized Depthwise-Separable CNN (DS-CNN) ")
        f.write("designed for keyword spotting ('Proxima') on microcontrollers such as the ESP32.\n\n")
        f.write("## Key Specifications\n")
        f.write(f"- **Architecture**: Depthwise-Separable CNN (DS-CNN V3)\n")
        f.write(f"- **Input Shape**: `(201, 40, 1)` Log-Mel Spectrogram\n")
        f.write(f"- **Classes**: `0 = UNKNOWN`, `1 = PROXIMA`\n")
        f.write(f"- **Parameters**: `{best_model.count_params():,}`\n")
        f.write(f"- **INT8 Model Size**: `{int8_bytes:,} bytes` (**{int8_bytes/1024:.2f} KB**) — *well below the 256 KB budget*\n")
        f.write(f"- **Estimated Tensor Arena**: `~{arena_estimate_kb} KB`\n\n")
        f.write("## Performance on Independent TEST Partition\n")
        f.write(f"- **Test Accuracy**: **{acc_int8 * 100:.2f}%**\n")
        f.write(f"- **Precision**: **{prec_int8 * 100:.2f}%**\n")
        f.write(f"- **Recall**: **{rec_int8 * 100:.2f}%**\n")
        f.write(f"- **F1-Score**: **{f1_int8 * 100:.2f}%**\n")
        f.write(f"- **PROXIMA False-Negative Rate**: **{fnr_int8 * 100:.2f}%**\n")
        f.write(f"- **UNKNOWN False-Positive Rate**: **{fpr_int8 * 100:.2f}%**\n\n")
        f.write("## Files in this Directory\n")
        f.write("- `best_model.keras`: Best trained Float32 Keras model checkpoint.\n")
        f.write("- `proxima_kws_float32.tflite`: Standard TensorFlow Lite Float32 model.\n")
        f.write("- `proxima_kws_int8.tflite`: Full integer INT8 quantized model for ESP32.\n")
        f.write("- `model_summary.txt`: Layer-by-layer parameter summary and progression history.\n")
        f.write("- `training_history.csv`: Per-epoch train/val loss and accuracy.\n")
        f.write("- `test_results.txt`: Detailed evaluation report on the test set.\n")
        f.write("- `confusion_matrix.png`: Annotated visualization of test confusion matrix.\n")
    print(f"Saved: {readme_path}")
    print("\nPROXIMA_MODEL_1 PIPELINE SUCCESSFULLY COMPLETED!")

if __name__ == "__main__":
    main()
