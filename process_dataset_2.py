import os
import sys
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import av
import soundfile as sf
import torch
import silero_vad

def run_pipeline():
    workspace = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
    source_dir = workspace / "PROXIMA_DATASET"
    output_dir = workspace / "PROXIMA_DATA_2"
    
    print(f"Source dataset: {source_dir}")
    print(f"Target dataset: {output_dir}")
    
    # Safety Check: If target directory already exists, abort
    if output_dir.exists():
        print(f"ERROR: Target directory {output_dir} already exists! Aborting for safety.")
        sys.exit(1)
        
    # Verify source directory exists
    if not source_dir.exists():
        print(f"ERROR: Source directory {source_dir} does not exist!")
        sys.exit(1)

    # Load Silero VAD model
    print("Loading Silero VAD model...")
    vad_model = silero_vad.load_silero_vad()
    
    # -------------------------------------------------------------
    # STEP 1: Inventory & Discovery
    # -------------------------------------------------------------
    print("\n--- STEP 1: INVENTORY & DISCOVERY ---")
    all_files = sorted([f for f in source_dir.rglob("*") if f.is_file()])
    print(f"Total files discovered in {source_dir}: {len(all_files)}")
    
    inventory = []
    
    for f in all_files:
        rel = f.relative_to(source_dir)
        parts = rel.parts
        # parts: (speaker, class_name, filename) or more
        spk = parts[0] if len(parts) > 0 else "UNKNOWN_SPK"
        cls = parts[1] if len(parts) > 1 else "UNKNOWN_CLASS"
        ext = f.suffix.lower()
        
        # Probe media file
        valid = False
        sr = None
        channels = None
        duration = None
        n_samples = None
        probe_err = None
        
        try:
            with av.open(str(f)) as container:
                audio_streams = container.streams.audio
                if not audio_streams:
                    probe_err = "No audio stream found"
                else:
                    stream = audio_streams[0]
                    sr = stream.rate
                    channels = stream.channels
                    tot_s = 0
                    for frame in container.decode(stream):
                        tot_s += frame.samples
                    n_samples = tot_s
                    duration = tot_s / sr if (sr and sr > 0) else 0.0
                    valid = True
        except Exception as e:
            probe_err = str(e)
            
        inventory.append({
            "original_path": str(f),
            "rel_path": str(rel),
            "speaker": spk,
            "class": cls,
            "filename": f.name,
            "stem": f.stem,
            "ext": ext,
            "valid": valid,
            "sr": sr,
            "channels": channels,
            "duration": duration,
            "samples": n_samples,
            "probe_err": probe_err
        })

    print(f"Inventory completed: {len(inventory)} items.")
    valid_count = sum(1 for item in inventory if item["valid"])
    invalid_count = len(inventory) - valid_count
    print(f"Valid media files: {valid_count}, Invalid: {invalid_count}")
    
    # Check for filename collisions within each (speaker, class)
    output_names = {}
    collision_counts = defaultdict(int)
    for item in inventory:
        key = (item["speaker"], item["class"], item["stem"])
        collision_counts[key] += 1
        
    # Map each file to deterministic unique output filename
    seen_stems = defaultdict(int)
    for item in inventory:
        key = (item["speaker"], item["class"], item["stem"])
        if collision_counts[key] > 1:
            idx = seen_stems[key]
            seen_stems[key] += 1
            out_filename = f"{item['stem']}_{idx:02d}.wav"
        else:
            out_filename = f"{item['stem']}.wav"
            
        out_rel = Path(item["speaker"]) / item["class"] / out_filename
        item["output_rel_path"] = str(out_rel)
        item["output_path"] = str(output_dir / out_rel)
        item["output_filename"] = out_filename

    # -------------------------------------------------------------
    # STEP 2-8: Processing & Audio Standardization
    # -------------------------------------------------------------
    print("\n--- PROCESSING & AUDIO STANDARDIZATION ---")
    
    TARGET_SR = 16000
    TARGET_SAMPLES = 32000
    
    processing_results = []
    
    # Create target directories
    for item in inventory:
        if item["valid"]:
            out_parent = Path(item["output_path"]).parent
            out_parent.mkdir(parents=True, exist_ok=True)
            
    # For amplitude statistics reporting
    raw_peak_amplitudes = []
    raw_rms_amplitudes = []
    
    for idx, item in enumerate(inventory, 1):
        if not item["valid"]:
            processing_results.append({
                "original_path": item["original_path"],
                "output_path": "",
                "speaker": item["speaker"],
                "class": item["class"],
                "original_format": item["ext"],
                "original_duration": item["duration"] if item["duration"] is not None else "",
                "original_sample_rate": item["sr"] if item["sr"] is not None else "",
                "original_channels": item["channels"] if item["channels"] is not None else "",
                "processing_status": "FAILED",
                "failure_reason": item["probe_err"]
            })
            continue

        try:
            # Decode and resample to 16kHz mono PCM
            with av.open(item["original_path"]) as container:
                audio_streams = container.streams.audio
                if not audio_streams:
                    raise RuntimeError("No audio stream")
                stream = audio_streams[0]
                
                # Resample directly to 16kHz mono s16 using FFmpeg's libswresample
                resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
                frames = []
                for frame in container.decode(stream):
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames:
                        frames.extend(resampled_frames)
                resampled_frames = resampler.resample(None)
                if resampled_frames:
                    frames.extend(resampled_frames)
                    
            if not frames:
                raise RuntimeError("Decoded audio yielded 0 frames")
                
            audio_arr = np.concatenate([f.to_ndarray() for f in frames], axis=1)[0] # 1D int16
            
            # Record original amplitude stats (without normalizing)
            peak_val = int(np.max(np.abs(audio_arr)))
            rms_val = float(np.sqrt(np.mean(audio_arr.astype(np.float64)**2)))
            raw_peak_amplitudes.append(peak_val)
            raw_rms_amplitudes.append(rms_val)
            
            curr_samples = len(audio_arr)
            
            # Duration Normalization to exactly 32000 samples (2.000s)
            if curr_samples < TARGET_SAMPLES:
                # Shorter than 2s: Symmetric padding with digital silence (zeros)
                pad_total = TARGET_SAMPLES - curr_samples
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                final_audio = np.pad(audio_arr, (pad_left, pad_right), mode="constant", constant_values=0)
                norm_method = f"pad_silence (left={pad_left}, right={pad_right})"
            elif curr_samples == TARGET_SAMPLES:
                # Exactly 2s
                final_audio = audio_arr
                norm_method = "exact_match"
            else:
                # Longer than 2s: Select 2.0-second window
                # Use Silero VAD to locate speech activity
                audio_float = audio_arr.astype(np.float32) / 32768.0
                tensor = torch.from_numpy(audio_float)
                timestamps = silero_vad.get_speech_timestamps(tensor, vad_model, sampling_rate=TARGET_SR)
                
                if timestamps:
                    speech_start = timestamps[0]["start"]
                    speech_end = timestamps[-1]["end"]
                    speech_len = speech_end - speech_start
                    
                    if speech_len <= TARGET_SAMPLES:
                        # Center speech in 2s window
                        center = (speech_start + speech_end) // 2
                        w_start = center - TARGET_SAMPLES // 2
                        # Clamp window while strictly covering the speech if possible
                        w_start = max(0, min(w_start, curr_samples - TARGET_SAMPLES))
                        if w_start > speech_start:
                            w_start = max(0, speech_start)
                        if w_start + TARGET_SAMPLES < speech_end:
                            w_start = min(curr_samples - TARGET_SAMPLES, speech_end - TARGET_SAMPLES)
                        norm_method = f"vad_fit_center (speech=[{speech_start}:{speech_end}])"
                    else:
                        # Speech spans > 2s: choose max energy 2s window within speech segment
                        sub = audio_arr[speech_start:speech_end].astype(np.float64)
                        conv = np.convolve(sub**2, np.ones(TARGET_SAMPLES), mode="valid")
                        best_offset = int(np.argmax(conv))
                        w_start = speech_start + best_offset
                        norm_method = f"vad_max_energy (speech=[{speech_start}:{speech_end}])"
                else:
                    # Fallback: choose 2s window with maximum energy across entire audio
                    conv = np.convolve(audio_arr.astype(np.float64)**2, np.ones(TARGET_SAMPLES), mode="valid")
                    w_start = int(np.argmax(conv))
                    norm_method = "energy_fallback"
                    
                # Final clamp bounds
                w_start = max(0, min(w_start, curr_samples - TARGET_SAMPLES))
                final_audio = audio_arr[w_start : w_start + TARGET_SAMPLES]
                
            # Verify exactly 32000 samples
            if len(final_audio) != TARGET_SAMPLES:
                raise RuntimeError(f"Unexpected audio length: {len(final_audio)} != {TARGET_SAMPLES}")
                
            # Write standardized WAV file: PCM 16-bit little endian, mono, 16000Hz
            out_file = Path(item["output_path"])
            sf.write(str(out_file), final_audio, TARGET_SR, subtype="PCM_16", format="WAV")
            
            processing_results.append({
                "original_path": item["original_path"],
                "output_path": item["output_path"],
                "speaker": item["speaker"],
                "class": item["class"],
                "original_format": item["ext"],
                "original_duration": f"{item['duration']:.4f}",
                "original_sample_rate": item["sr"],
                "original_channels": item["channels"],
                "processing_status": "SUCCESS",
                "failure_reason": ""
            })
            
            if idx % 50 == 0 or idx == len(inventory):
                print(f"Processed {idx}/{len(inventory)} files...")
                
        except Exception as e:
            processing_results.append({
                "original_path": item["original_path"],
                "output_path": "",
                "speaker": item["speaker"],
                "class": item["class"],
                "original_format": item["ext"],
                "original_duration": f"{item['duration']:.4f}" if item["duration"] is not None else "",
                "original_sample_rate": item["sr"] if item["sr"] is not None else "",
                "original_channels": item["channels"] if item["channels"] is not None else "",
                "processing_status": "FAILED",
                "failure_reason": str(e)
            })

    # -------------------------------------------------------------
    # STEP 9: Independent Validation of Output Files
    # -------------------------------------------------------------
    print("\n--- STEP 9: INDEPENDENT VALIDATION ---")
    
    validation_failures = []
    validated_files = 0
    
    for res in processing_results:
        if res["processing_status"] != "SUCCESS":
            continue
            
        out_p = Path(res["output_path"])
        # 1. file exists
        if not out_p.is_file():
            validation_failures.append((str(out_p), "File does not exist on disk"))
            continue
            
        # 2. readable by audio decoder & check properties
        try:
            info = sf.info(str(out_p))
            # Format check
            if info.format != "WAV":
                validation_failures.append((str(out_p), f"Format is {info.format}, expected WAV"))
                continue
            # Subtype check: PCM 16-bit
            if info.subtype != "PCM_16":
                validation_failures.append((str(out_p), f"Subtype is {info.subtype}, expected PCM_16"))
                continue
            # Channels check: 1 (mono)
            if info.channels != 1:
                validation_failures.append((str(out_p), f"Channels is {info.channels}, expected 1 (mono)"))
                continue
            # Sample rate check: 16000 Hz
            if info.samplerate != 16000:
                validation_failures.append((str(out_p), f"Samplerate is {info.samplerate}, expected 16000"))
                continue
            # Exactly 32000 samples
            if info.frames != 32000:
                validation_failures.append((str(out_p), f"Frames count is {info.frames}, expected 32000"))
                continue
            # Exactly 2.000 seconds
            if abs(info.duration - 2.0) > 1e-6:
                validation_failures.append((str(out_p), f"Duration is {info.duration:.6f}s, expected 2.000s"))
                continue
                
            # Read actual data array to verify integrity
            data, sr = sf.read(str(out_p), dtype="int16")
            if len(data) != 32000 or sr != 16000:
                validation_failures.append((str(out_p), f"Read data length mismatch: {len(data)} samples"))
                continue
                
            validated_files += 1
        except Exception as e:
            validation_failures.append((str(out_p), f"Decoder error: {str(e)}"))

    print(f"Validation finished: {validated_files} files PASSED, {len(validation_failures)} FAILED.")
    if validation_failures:
        print("Validation failures:")
        for vf in validation_failures[:10]:
            print("  ", vf)

    # -------------------------------------------------------------
    # STEP 10: Generate Reports
    # -------------------------------------------------------------
    print("\n--- STEP 10: GENERATING REPORTS ---")
    
    # 1. processing_report.csv
    csv_path = output_dir / "processing_report.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "original_path",
            "output_path",
            "speaker",
            "class",
            "original_format",
            "original_duration",
            "original_sample_rate",
            "original_channels",
            "processing_status",
            "failure_reason"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in processing_results:
            writer.writerow(row)
    print(f"Created: {csv_path}")
    
    # 2. processing_summary.txt
    total_source = len(inventory)
    total_success = sum(1 for r in processing_results if r["processing_status"] == "SUCCESS")
    total_failed = sum(1 for r in processing_results if r["processing_status"] == "FAILED")
    total_skipped = sum(1 for r in processing_results if r["processing_status"] == "SKIPPED")
    
    speakers = sorted(list(set(r["speaker"] for r in inventory)))
    proxima_count = sum(1 for r in inventory if r["class"] == "proxima")
    unknown_count = sum(1 for r in inventory if r["class"] == "unknown")
    
    format_counts = Counter(r["ext"] for r in inventory)
    speaker_counts = Counter(r["speaker"] for r in inventory)
    class_counts = Counter(r["class"] for r in inventory)
    
    peak_min = min(raw_peak_amplitudes) if raw_peak_amplitudes else 0
    peak_max = max(raw_peak_amplitudes) if raw_peak_amplitudes else 0
    peak_mean = np.mean(raw_peak_amplitudes) if raw_peak_amplitudes else 0
    rms_min = min(raw_rms_amplitudes) if raw_rms_amplitudes else 0
    rms_max = max(raw_rms_amplitudes) if raw_rms_amplitudes else 0
    rms_mean = np.mean(raw_rms_amplitudes) if raw_rms_amplitudes else 0
    
    summary_path = output_dir / "processing_summary.txt"
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA_DATA_2 DATASET PROCESSING & STANDARDIZATION SUMMARY\n")
        f.write("================================================================================\n\n")
        
        f.write(f"Total source files discovered: {total_source}\n")
        f.write(f"Total successfully processed:  {total_success}\n")
        f.write(f"Total failed:                  {total_failed}\n")
        f.write(f"Total skipped:                 {total_skipped}\n\n")
        
        f.write(f"Number of SPK folders:         {len(speakers)}\n")
        f.write(f"Number of proxima recordings:  {proxima_count}\n")
        f.write(f"Number of unknown recordings:  {unknown_count}\n\n")
        
        f.write("--- Count by Original File Format ---\n")
        for fmt, cnt in sorted(format_counts.items()):
            f.write(f"  {fmt}: {cnt}\n")
        f.write("\n")
        
        f.write("--- Count by Class ---\n")
        for cls_name, cnt in sorted(class_counts.items()):
            f.write(f"  {cls_name}: {cnt}\n")
        f.write("\n")
        
        f.write("--- Count by Speaker ---\n")
        for spk, cnt in sorted(speaker_counts.items()):
            succ_spk = sum(1 for r in processing_results if r["speaker"] == spk and r["processing_status"] == "SUCCESS")
            f.write(f"  {spk}: {cnt} source -> {succ_spk} output WAVs\n")
        f.write("\n")
        
        f.write("--- Validation Results ---\n")
        f.write(f"Total output files generated:  {validated_files}\n")
        f.write(f"Validation failures:           {len(validation_failures)}\n")
        if validation_failures:
            for vf in validation_failures:
                f.write(f"  FAILED: {vf[0]} -> {vf[1]}\n")
        else:
            f.write("  ALL output files passed validation checks:\n")
            f.write("    - Container: WAV\n")
            f.write("    - Codec: PCM signed 16-bit little-endian (PCM_16)\n")
            f.write("    - Channel: 1 (mono)\n")
            f.write("    - Sample rate: 16000 Hz\n")
            f.write("    - Frame count: exactly 32000 samples\n")
            f.write("    - Duration: exactly 2.000 seconds\n\n")
            
        f.write("--- Amplitude Statistics (Raw, Preserved) ---\n")
        f.write(f"Peak Amplitude (int16 range 0..32767):\n")
        f.write(f"  Min:  {peak_min}\n")
        f.write(f"  Max:  {peak_max}\n")
        f.write(f"  Mean: {peak_mean:.2f}\n")
        f.write(f"RMS Amplitude:\n")
        f.write(f"  Min:  {rms_min:.2f}\n")
        f.write(f"  Max:  {rms_max:.2f}\n")
        f.write(f"  Mean: {rms_mean:.2f}\n")
        f.write("Note: Amplitude characteristics were kept intact as requested (no unrequested normalization applied).\n")
        
    print(f"Created: {summary_path}")
    print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline()
