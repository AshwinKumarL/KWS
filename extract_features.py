import os
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_feature_extraction():
    workspace = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
    source_split_dir = workspace / "PROXIMA_DATA_1_SPLIT"
    features_dir = workspace / "PROXIMA_FEATURES_1"
    
    print(f"Source dataset: {source_split_dir}")
    print(f"Features directory: {features_dir}")
    
    # 1. Safety check
    if features_dir.exists():
        print(f"ERROR: Features directory {features_dir} already exists! Aborting for safety.")
        sys.exit(1)
        
    if not source_split_dir.exists():
        print(f"ERROR: Source directory {source_split_dir} does not exist!")
        sys.exit(1)
        
    # Feature extraction configuration
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 2.0
    EXPECTED_SAMPLES = 32000
    N_FFT = 512
    WIN_LENGTH = 400     # 25 ms at 16 kHz
    HOP_LENGTH = 160     # 10 ms at 16 kHz
    N_MELS = 40
    F_MIN = 20.0
    F_MAX = 8000.0
    EPSILON = 1e-6
    DTYPE_STR = "float32"
    
    # Instantiate MelSpectrogram transform
    # Using Hann window, power=2.0 (power spectrogram), center=True (reflect pad by n_fft // 2 = 256)
    mel_transform = T.MelSpectrogram(
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
    
    # 2. Discover all WAV files in TRAIN and TEST
    splits = ["train", "test"]
    wav_inventory = []
    
    for split in splits:
        split_dir = source_split_dir / split
        if not split_dir.exists():
            continue
        spk_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for spk_d in spk_dirs:
            for cls_name in ["proxima", "unknown"]:
                cls_d = spk_d / cls_name
                if not cls_d.exists():
                    continue
                for wav_p in sorted(cls_d.glob("*.wav")):
                    wav_inventory.append({
                        "source_wav": wav_p,
                        "rel_source": wav_p.relative_to(workspace).as_posix(),
                        "split": split,
                        "speaker": spk_d.name,
                        "class": cls_name,
                        "filename": wav_p.name,
                        "stem": wav_p.stem
                    })

    print(f"Total source WAV files discovered: {len(wav_inventory)}")
    assert len(wav_inventory) == 544, f"Expected 544 WAV files, got {len(wav_inventory)}"
    
    # Create output directory structure
    for item in wav_inventory:
        out_sub = features_dir / item["split"] / item["speaker"] / item["class"]
        out_sub.mkdir(parents=True, exist_ok=True)
        
    # Also create empty speaker folders in features_dir if present in source
    for split in splits:
        for spk_d in (source_split_dir / split).iterdir():
            if spk_d.is_dir():
                for cls_name in ["proxima", "unknown"]:
                    (features_dir / split / spk_d.name / cls_name).mkdir(parents=True, exist_ok=True)

    # 3. Extract features
    print("\n--- EXTRACTING LOG-MEL SPECTROGRAMS ---")
    
    results = []
    global_min = float("inf")
    global_max = float("-inf")
    nan_count = 0
    inf_count = 0
    fixed_shape = None
    
    for idx, item in enumerate(wav_inventory, 1):
        wav_p = item["source_wav"]
        npy_name = f"{item['stem']}.npy"
        out_npy_p = features_dir / item["split"] / item["speaker"] / item["class"] / npy_name
        rel_feature = out_npy_p.relative_to(workspace).as_posix()
        
        try:
            # 1. Read WAV file
            data, sr = sf.read(str(wav_p), dtype="float32")
            if sr != SAMPLE_RATE:
                raise ValueError(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")
            if len(data) != EXPECTED_SAMPLES:
                raise ValueError(f"Sample count mismatch: {len(data)} != {EXPECTED_SAMPLES}")
                
            # Convert to torch tensor: shape (1, 32000)
            waveform = torch.from_numpy(data).unsqueeze(0)
            
            # 2. Extract Power Mel Spectrogram: shape (1, 40, time_frames)
            mel_energies = mel_transform(waveform)
            
            # 3. Logarithmic compression: log(max(mel_energy, epsilon))
            log_mel = torch.log(torch.clamp(mel_energies, min=EPSILON))
            
            # 4. Transpose to (time_frames, 40 Mel bins)
            feature_2d = log_mel.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32)
            
            # Verify shape consistency
            curr_shape = feature_2d.shape
            if fixed_shape is None:
                fixed_shape = curr_shape
                print(f"Determined fixed feature shape (time_frames, mel_bins): {fixed_shape}")
            elif curr_shape != fixed_shape:
                raise ValueError(f"Shape inconsistency: {curr_shape} != {fixed_shape}")
                
            # Check values
            if np.isnan(feature_2d).any():
                nan_count += 1
                raise ValueError("NaN values detected in feature")
            if np.isinf(feature_2d).any():
                inf_count += 1
                raise ValueError("Infinite values detected in feature")
                
            f_min_val = float(np.min(feature_2d))
            f_max_val = float(np.max(feature_2d))
            if f_min_val < global_min:
                global_min = f_min_val
            if f_max_val > global_max:
                global_max = f_max_val
                
            # 5. Save as NumPy .npy file
            np.save(str(out_npy_p), feature_2d)
            
            results.append({
                "source_wav": item["rel_source"],
                "feature_file": rel_feature,
                "split": item["split"],
                "speaker": item["speaker"],
                "class": item["class"],
                "sample_rate": SAMPLE_RATE,
                "duration": AUDIO_DURATION,
                "n_fft": N_FFT,
                "window_length": WIN_LENGTH,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "fmin": F_MIN,
                "fmax": F_MAX,
                "feature_shape": f"{curr_shape[0]}x{curr_shape[1]}",
                "dtype": DTYPE_STR,
                "status": "SUCCESS",
                "abs_feature_path": out_npy_p
            })
            
            if idx % 100 == 0 or idx == len(wav_inventory):
                print(f"Extracted {idx}/{len(wav_inventory)} features...")
                
        except Exception as e:
            results.append({
                "source_wav": item["rel_source"],
                "feature_file": rel_feature,
                "split": item["split"],
                "speaker": item["speaker"],
                "class": item["class"],
                "sample_rate": SAMPLE_RATE,
                "duration": AUDIO_DURATION,
                "n_fft": N_FFT,
                "window_length": WIN_LENGTH,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "fmin": F_MIN,
                "fmax": F_MAX,
                "feature_shape": "",
                "dtype": DTYPE_STR,
                "status": f"FAILED: {str(e)}",
                "abs_feature_path": out_npy_p
            })

    # -------------------------------------------------------------
    # 4. Manifest Generation
    # -------------------------------------------------------------
    manifest_path = features_dir / "feature_manifest.csv"
    print(f"\nWriting manifest to {manifest_path}...")
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_wav",
            "feature_file",
            "split",
            "speaker",
            "class",
            "sample_rate",
            "duration",
            "n_fft",
            "window_length",
            "hop_length",
            "n_mels",
            "fmin",
            "fmax",
            "feature_shape",
            "dtype",
            "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row_dict = {k: r[k] for k in fieldnames}
            writer.writerow(row_dict)
    print("Manifest created successfully.")

    # -------------------------------------------------------------
    # 5. Independent Validation
    # -------------------------------------------------------------
    print("\n--- INDEPENDENT VALIDATION ---")
    val_errors = []
    
    total_success = sum(1 for r in results if r["status"] == "SUCCESS")
    total_failed = len(results) - total_success
    
    if total_failed > 0:
        val_errors.append(f"{total_failed} files failed during extraction")
        
    for r in results:
        if r["status"] != "SUCCESS":
            continue
            
        # 1. Source WAV exists and readable
        orig_p = workspace / r["source_wav"]
        if not orig_p.is_file():
            val_errors.append(f"Source WAV missing: {orig_p}")
            break
            
        # 2. Feature file exists on disk
        feat_p = r["abs_feature_path"]
        if not feat_p.is_file():
            val_errors.append(f"Feature file missing: {feat_p}")
            break
            
        # 3. Feature file can be loaded
        try:
            arr = np.load(str(feat_p))
        except Exception as e:
            val_errors.append(f"Failed to load feature file {feat_p}: {e}")
            break
            
        # 4. Dtype check
        if arr.dtype != np.float32:
            val_errors.append(f"Invalid dtype {arr.dtype} in {feat_p}, expected float32")
            break
            
        # 5. Shape check
        if arr.shape != fixed_shape:
            val_errors.append(f"Shape mismatch in {feat_p}: {arr.shape} != {fixed_shape}")
            break
            
        # 6. NaN / Inf check
        if np.isnan(arr).any():
            val_errors.append(f"NaN value found in {feat_p}")
            break
        if np.isinf(arr).any():
            val_errors.append(f"Inf value found in {feat_p}")
            break
            
        # 7. Split origin isolation
        if r["split"] == "train" and "train" not in r["source_wav"]:
            val_errors.append(f"TRAIN feature came from non-train source: {r['source_wav']}")
            break
        if r["split"] == "test" and "test" not in r["source_wav"]:
            val_errors.append(f"TEST feature came from non-test source: {r['source_wav']}")
            break

    # 8. Check total count
    all_npy = list(features_dir.rglob("*.npy"))
    if len(all_npy) != len(wav_inventory):
        val_errors.append(f"Total .npy count on disk ({len(all_npy)}) != expected ({len(wav_inventory)})")
        
    if val_errors:
        print("VALIDATION FAILED:")
        for err in val_errors:
            print("  ", err)
        sys.exit(1)
    else:
        print(f"ALL VALIDATION CHECKS PASSED: {len(all_npy)} feature files verified!")

    # -------------------------------------------------------------
    # 6. Generate Visual Sanity Check Images
    # -------------------------------------------------------------
    print("\n--- GENERATING VISUAL SANITY CHECK SPECTROGRAMS ---")
    examples_dir = features_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Pick 2 samples from:
    # 1. train / proxima
    # 2. train / unknown
    # 3. test / proxima
    # 4. test / unknown
    sample_categories = {
        ("train", "proxima"): [],
        ("train", "unknown"): [],
        ("test", "proxima"): [],
        ("test", "unknown"): []
    }
    
    for r in results:
        key = (r["split"], r["class"])
        if len(sample_categories[key]) < 2:
            sample_categories[key].append(r)
            
    for (split_name, cls_name), items in sample_categories.items():
        for i, item in enumerate(items, 1):
            feat_arr = np.load(str(item["abs_feature_path"])) # (201, 40)
            
            # Plot spectrogram: time on x-axis (0 to 2.0s), Mel bins on y-axis (0 to 40)
            # feat_arr is (time, mels), transpose to (mels, time) for standard spectrogram plot
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            im = ax.imshow(
                feat_arr.T,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                extent=[0.0, AUDIO_DURATION, 0, N_MELS]
            )
            cbar = fig.colorbar(im, ax=ax, format="%+2.1f dB")
            cbar.set_label("Log-Mel Energy")
            
            source_filename = Path(item["source_wav"]).name
            ax.set_title(
                f"Log-Mel Spectrogram | {split_name.upper()} | Speaker: {item['speaker']} | Class: {cls_name}\nSource: {source_filename}",
                fontsize=11,
                fontweight="bold"
            )
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Mel Filter Bank Bins (0 - 40)", fontsize=10)
            plt.tight_layout()
            
            out_img_name = f"{split_name}_{item['speaker']}_{cls_name}_{item['source_wav'].split('/')[-1].replace('.wav', '')}.png"
            # sanitize filename
            out_img_name = out_img_name.replace(" ", "_").replace("(", "").replace(")", "")
            out_img_p = examples_dir / out_img_name
            plt.savefig(str(out_img_p))
            plt.close(fig)
            print(f"Saved example plot: {out_img_p.name}")

    # -------------------------------------------------------------
    # 7. Summary Generation
    # -------------------------------------------------------------
    print("\n--- GENERATING FEATURE SUMMARY ---")
    summary_path = features_dir / "feature_summary.txt"
    
    train_count = sum(1 for r in results if r["split"] == "train")
    test_count = sum(1 for r in results if r["split"] == "test")
    proxima_count = sum(1 for r in results if r["class"] == "proxima")
    unknown_count = sum(1 for r in results if r["class"] == "unknown")
    
    spk_counter = Counter(r["speaker"] for r in results)
    all_spks = sorted([d.name for d in (source_split_dir / "train").iterdir() if d.is_dir()])
    
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA_FEATURES_1 LOG-MEL SPECTROGRAM EXTRACTION SUMMARY\n")
        f.write("================================================================================\n\n")
        
        f.write(f"Total WAV files processed:     {len(wav_inventory)}\n")
        f.write(f"Total feature files created:   {total_success}\n")
        f.write(f"Failed files:                  {total_failed}\n\n")
        
        f.write(f"Train feature count:           {train_count} ({(train_count/len(wav_inventory))*100:.2f}%)\n")
        f.write(f"Test feature count:            {test_count} ({(test_count/len(wav_inventory))*100:.2f}%)\n\n")
        
        f.write(f"Proxima recordings:            {proxima_count}\n")
        f.write(f"Unknown recordings:            {unknown_count}\n\n")
        
        f.write("--- Log-Mel Extraction Parameters ---\n")
        f.write(f"  Sample Rate:                 {SAMPLE_RATE} Hz\n")
        f.write(f"  Audio Duration:              {AUDIO_DURATION} seconds (32,000 samples)\n")
        f.write(f"  FFT Size (n_fft):            {N_FFT}\n")
        f.write(f"  Window Function:             Hann window\n")
        f.write(f"  Window Length:               {WIN_LENGTH} samples (25.0 ms)\n")
        f.write(f"  Hop Length:                  {HOP_LENGTH} samples (10.0 ms)\n")
        f.write(f"  Number of Mel Bins:          {N_MELS}\n")
        f.write(f"  Minimum Frequency (fmin):    {F_MIN} Hz\n")
        f.write(f"  Maximum Frequency (fmax):    {F_MAX} Hz\n")
        f.write(f"  Power Spectrogram:           |STFT|^2 (power=2.0)\n")
        f.write(f"  STFT Framing Convention:     center=True, pad_mode='reflect' (padding = n_fft // 2 = 256)\n")
        f.write(f"  Logarithmic Compression:     log(max(mel_energy, {EPSILON}))\n\n")
        
        f.write("--- Feature Tensor Specifications ---\n")
        f.write(f"  Feature Shape:               {fixed_shape} (time_frames={fixed_shape[0]}, mel_bins={fixed_shape[1]})\n")
        f.write(f"  Feature Data Type:           {DTYPE_STR} (NumPy .npy format)\n")
        f.write(f"  Minimum Feature Value:       {global_min:.4f}\n")
        f.write(f"  Maximum Feature Value:       {global_max:.4f}\n")
        f.write(f"  NaN Count:                   {nan_count}\n")
        f.write(f"  Infinity Count:              {inf_count}\n\n")
        
        f.write("--- Speaker Breakdown ---\n")
        for spk in all_spks:
            cnt = spk_counter.get(spk, 0)
            tr = sum(1 for r in results if r["speaker"] == spk and r["split"] == "train")
            te = sum(1 for r in results if r["speaker"] == spk and r["split"] == "test")
            f.write(f"  {spk}: total={cnt} (train={tr}, test={te})\n")
        f.write("\n")
        
        f.write("--- Validation Checklist ---\n")
        f.write("  [PASS] 1. Every source WAV exists and was verified.\n")
        f.write("  [PASS] 2. Every source WAV is readable and valid.\n")
        f.write("  [PASS] 3. Every feature file exists on disk as a .npy file.\n")
        f.write("  [PASS] 4. Every feature file was loaded and verified.\n")
        f.write("  [PASS] 5. Feature dtype is FLOAT32 across 100% of files.\n")
        f.write("  [PASS] 6. Feature shape is strictly identical: (201, 40).\n")
        f.write("  [PASS] 7. Zero NaN values detected.\n")
        f.write("  [PASS] 8. Zero infinite values detected.\n")
        f.write("  [PASS] 9. TRAIN features derived strictly from TRAIN WAVs.\n")
        f.write("  [PASS] 10. TEST features derived strictly from TEST WAVs.\n")
        f.write("  [PASS] 11. Generated feature count exactly matches processed WAV count (544).\n")
        
    print(f"Summary written to {summary_path}")
    print("\nFEATURE EXTRACTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_feature_extraction()
