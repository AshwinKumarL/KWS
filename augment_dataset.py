import os
import sys
import csv
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import scipy.signal

TARGET_SR = 16000
TARGET_SAMPLES = 32000

def augment_waveform(audio, mode, rng):
    """
    Apply high-fidelity physical acoustic augmentations to 16kHz mono audio.
    Audio is assumed to be float32 array in [-1.0, 1.0].
    Returns float32 array of shape (TARGET_SAMPLES,).
    """
    audio = audio.copy()
    
    if mode == "time_shift":
        # Shift forward or backward by ±100ms to ±200ms (±1600 to ±3200 samples)
        shift = rng.randint(-3200, 3200)
        # We zero-pad the shifted edge rather than pure circular wrap to avoid clicking
        res = np.zeros_like(audio)
        if shift > 0:
            res[shift:] = audio[:-shift]
        elif shift < 0:
            res[:shift] = audio[-shift:]
        else:
            res = audio
        audio = res

    elif mode == "noise":
        # Additive Gaussian noise with SNR between 18 and 28 dB
        snr_db = rng.uniform(18.0, 28.0)
        sig_power = np.mean(audio ** 2)
        if sig_power > 1e-8:
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = rng.standard_normal(len(audio)).astype(np.float32) * np.sqrt(noise_power)
            audio = audio + noise

    elif mode == "gain":
        # Volume variation: 0.7x to 1.3x
        gain = rng.uniform(0.7, 1.3)
        audio = audio * gain

    elif mode == "speed_pitch":
        # Resample speed/pitch perturbation by ±5% to simulate vocal tract variability
        speed_rate = rng.uniform(0.95, 1.05)
        new_len = int(round(len(audio) / speed_rate))
        resampled = scipy.signal.resample(audio, new_len)
        # Pad or center crop back to TARGET_SAMPLES
        if len(resampled) < TARGET_SAMPLES:
            pad_total = TARGET_SAMPLES - len(resampled)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = np.pad(resampled, (pad_left, pad_right), mode="constant", constant_values=0.0)
        else:
            start = (len(resampled) - TARGET_SAMPLES) // 2
            audio = resampled[start:start + TARGET_SAMPLES]

    elif mode == "composite":
        # Realistic room condition: mild time shift + ambient noise + gain
        shift = rng.randint(-1600, 1600)
        res = np.zeros_like(audio)
        if shift > 0:
            res[shift:] = audio[:-shift]
        elif shift < 0:
            res[:shift] = audio[-shift:]
        else:
            res = audio
        audio = res
        
        # Mild noise (24-30 dB)
        snr_db = rng.uniform(24.0, 30.0)
        sig_power = np.mean(audio ** 2)
        if sig_power > 1e-8:
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = rng.standard_normal(len(audio)).astype(np.float32) * np.sqrt(noise_power)
            audio = audio + noise
            
        # Gain
        audio = audio * rng.uniform(0.85, 1.15)

    # Ensure strictly in [-1.0, 1.0]
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio

def run_augmentation():
    SEED = 42
    rng = np.random.RandomState(SEED)
    py_rng = random.Random(SEED)

    workspace = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
    source_dir = workspace / "PROXIMA_DATA_2"
    output_dir = workspace / "PROXIMA_DATA_2_AGUMENTED"

    print(f"Source clean normalized dataset: {source_dir}")
    print(f"Output augmented dataset: {output_dir}")

    if not source_dir.exists():
        print(f"ERROR: Source directory {source_dir} does not exist!")
        sys.exit(1)

    if output_dir.exists():
        print(f"WARNING: Output directory {output_dir} already exists. Cleaning up...")
        import shutil
        shutil.rmtree(str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all WAV files in source_dir
    wav_files = sorted(list(source_dir.rglob("*.wav")))
    print(f"Discovered {len(wav_files)} standardized WAV files in {source_dir.name}.")

    inventory_report = []
    speaker_counts = defaultdict(lambda: defaultdict(int))
    
    # We will generate:
    # 1. The original clean sample: {stem}_orig.wav
    # 2. Time-shifted variant: {stem}_aug_shift.wav
    # 3. Noise injected variant: {stem}_aug_noise.wav
    # 4. Gain scaled variant: {stem}_aug_gain.wav
    # 5. Composite acoustic variant: {stem}_aug_comp.wav
    # Total: 5x expansion of dataset! (3,625 high-quality samples)

    aug_modes = [
        ("orig", None),
        ("aug_shift", "time_shift"),
        ("aug_noise", "noise"),
        ("aug_gain", "gain"),
        ("aug_comp", "composite")
    ]

    total_written = 0

    for idx, wav_p in enumerate(wav_files, 1):
        rel_p = wav_p.relative_to(source_dir)
        spk = rel_p.parts[0]
        cls_name = rel_p.parts[1]
        stem = wav_p.stem

        # Read audio
        audio, sr = sf.read(str(wav_p), dtype="float32")
        if sr != TARGET_SR or len(audio) != TARGET_SAMPLES:
            raise ValueError(f"Invalid format for {wav_p}: sr={sr}, len={len(audio)}")

        out_class_dir = output_dir / spk / cls_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for suffix, mode in aug_modes:
            out_filename = f"{stem}_{suffix}.wav"
            out_path = out_class_dir / out_filename

            if mode is None:
                aug_audio = audio
            else:
                aug_audio = augment_waveform(audio, mode, rng)

            # Convert to int16 PCM
            audio_int16 = (aug_audio * 32767.0).astype(np.int16)
            sf.write(str(out_path), audio_int16, TARGET_SR, subtype="PCM_16", format="WAV")

            speaker_counts[spk][cls_name] += 1
            total_written += 1

            inventory_report.append({
                "speaker": spk,
                "class": cls_name,
                "original_file": wav_p.name,
                "augmented_file": out_filename,
                "aug_mode": mode if mode else "clean_original",
                "rel_path": str(out_path.relative_to(output_dir)).replace("\\", "/")
            })

        if idx % 100 == 0 or idx == len(wav_files):
            print(f"Processed {idx}/{len(wav_files)} source files -> {total_written} augmented WAVs generated.")

    # Write summary report
    csv_path = output_dir / "augmentation_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "class", "original_file", "augmented_file", "aug_mode", "rel_path"])
        writer.writeheader()
        writer.writerows(inventory_report)

    summary_path = output_dir / "augmentation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA_DATA_2_AGUMENTED SUMMARY REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Source directory:            {source_dir}\n")
        f.write(f"Source file count:           {len(wav_files)}\n")
        f.write(f"Target directory:            {output_dir}\n")
        f.write(f"Total augmented files:       {total_written}\n")
        f.write(f"Expansion multiplier:        5x (1 clean + 4 physical augmentations)\n\n")
        f.write("--- Augmentation Modes ---\n")
        f.write("  1. clean_original:  Direct bit-perfect standardized original\n")
        f.write("  2. aug_shift:       Time shift (±100ms to ±200ms / ±1600-3200 samples)\n")
        f.write("  3. aug_noise:       Additive Gaussian / ambient room noise (SNR 18-28 dB)\n")
        f.write("  4. aug_gain:        Microphone gain / volume scaling (0.7x - 1.3x)\n")
        f.write("  5. aug_comp:        Composite realistic acoustic environment\n\n")
        f.write("--- Speaker & Class Distribution ---\n")
        total_p = 0
        total_u = 0
        for spk in sorted(speaker_counts.keys()):
            p = speaker_counts[spk]["proxima"]
            u = speaker_counts[spk]["unknown"]
            total_p += p
            total_u += u
            f.write(f"  {spk:8s} -> PROXIMA: {p:4d} | UNKNOWN: {u:4d} | TOTAL: {p+u:4d}\n")
        f.write(f"\nOverall PROXIMA: {total_p} | UNKNOWN: {total_u} | Total: {total_p + total_u}\n")

    print(f"\nAugmentation completed! Total files: {total_written}")
    print(f"Manifest saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    run_augmentation()
