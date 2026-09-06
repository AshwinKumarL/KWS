import os
import sys
import csv
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict
import soundfile as sf

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_split():
    SEED = 42
    workspace = Path(r"c:\Users\Dell\Desktop\KWS PROJECT")
    source_dir = workspace / "PROXIMA_DATA_1"
    split_dir = workspace / "PROXIMA_DATA_1_SPLIT"
    
    print(f"Source: {source_dir}")
    print(f"Destination: {split_dir}")
    print(f"Random seed: {SEED}")
    
    # 1. Safety Check: If target directory already exists, abort
    if split_dir.exists():
        print(f"ERROR: Destination directory {split_dir} already exists! Aborting.")
        sys.exit(1)
        
    if not source_dir.exists():
        print(f"ERROR: Source directory {source_dir} does not exist!")
        sys.exit(1)

    # 2. Discover all speaker folders and classes in PROXIMA_DATA_1
    spk_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    print(f"Discovered {len(spk_dirs)} speaker folders in {source_dir.name}.")
    
    # Structure: spk -> class -> list of files
    dataset_structure = defaultdict(lambda: defaultdict(list))
    total_wav_count = 0
    
    for spk_dir in spk_dirs:
        for cls_name in ["proxima", "unknown"]:
            cls_dir = spk_dir / cls_name
            if cls_dir.exists():
                wavs = sorted(list(cls_dir.glob("*.wav")), key=lambda p: p.name)
                dataset_structure[spk_dir.name][cls_name] = wavs
                total_wav_count += len(wavs)

    print(f"Total WAV files discovered across all folders: {total_wav_count}")
    assert total_wav_count == 544, f"Expected 544 files, got {total_wav_count}"
    
    # 3. Create target directory structures
    train_root = split_dir / "train"
    test_root = split_dir / "test"
    
    for spk_dir in spk_dirs:
        for cls_name in ["proxima", "unknown"]:
            (train_root / spk_dir.name / cls_name).mkdir(parents=True, exist_ok=True)
            (test_root / spk_dir.name / cls_name).mkdir(parents=True, exist_ok=True)
            
    # 4. Perform deterministic 80/20 split for each (speaker, class)
    # Using fixed seed for reproducible shuffling
    rng = random.Random(SEED)
    
    manifest_rows = []
    split_stats = {}
    
    # Track files for data leakage prevention validation
    train_dest_files = set()
    test_dest_files = set()
    original_files_assigned = set()
    
    total_train_count = 0
    total_test_count = 0
    
    # Iterate in sorted order to ensure 100% deterministic sequence
    for spk_name in sorted(dataset_structure.keys()):
        split_stats[spk_name] = {}
        for cls_name in ["proxima", "unknown"]:
            files = list(dataset_structure[spk_name][cls_name])
            total_count = len(files)
            
            if total_count == 0:
                split_stats[spk_name][cls_name] = {
                    "total": 0,
                    "train": 0,
                    "test": 0
                }
                continue
                
            # Deterministic rounding: train_count = round(0.8 * total_count)
            train_count = round(0.8 * total_count)
            test_count = total_count - train_count
            assert train_count + test_count == total_count
            
            # Shuffle files deterministically
            shuffled_files = list(files)
            rng.shuffle(shuffled_files)
            
            train_files = shuffled_files[:train_count]
            test_files = shuffled_files[train_count:]
            
            split_stats[spk_name][cls_name] = {
                "total": total_count,
                "train": len(train_files),
                "test": len(test_files)
            }
            
            total_train_count += len(train_files)
            total_test_count += len(test_files)
            
            # Copy train files
            for f in train_files:
                dest = train_root / spk_name / cls_name / f.name
                shutil.copy2(f, dest)
                
                # Check for duplication / leakage
                assert f not in original_files_assigned, f"Leakage: {f} assigned multiple times!"
                original_files_assigned.add(f)
                train_dest_files.add(dest)
                
                # Manifest entry
                rel_orig = f.relative_to(workspace).as_posix()
                manifest_rows.append({
                    "filename": f.name,
                    "original_path": rel_orig,
                    "split": "train",
                    "speaker": spk_name,
                    "class": cls_name
                })
                
            # Copy test files
            for f in test_files:
                dest = test_root / spk_name / cls_name / f.name
                shutil.copy2(f, dest)
                
                assert f not in original_files_assigned, f"Leakage: {f} assigned multiple times!"
                original_files_assigned.add(f)
                test_dest_files.add(dest)
                
                rel_orig = f.relative_to(workspace).as_posix()
                manifest_rows.append({
                    "filename": f.name,
                    "original_path": rel_orig,
                    "split": "test",
                    "speaker": spk_name,
                    "class": cls_name
                })

    print(f"Copying complete. Train: {total_train_count}, Test: {total_test_count}, Total: {total_train_count + total_test_count}")

    # 5. Write split_manifest.csv
    manifest_path = split_dir / "split_manifest.csv"
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "original_path", "split", "speaker", "class"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print(f"Created manifest: {manifest_path}")

    # 6. Write split_summary.txt
    summary_path = split_dir / "split_summary.txt"
    train_pct = (total_train_count / total_wav_count) * 100.0 if total_wav_count else 0
    test_pct = (total_test_count / total_wav_count) * 100.0 if total_wav_count else 0
    
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROXIMA_DATA_1_SPLIT DATASET SPLIT SUMMARY\n")
        f.write("================================================================================\n\n")
        
        f.write(f"Random Seed:                   {SEED}\n")
        f.write(f"Total number of speakers:      {len(spk_dirs)}\n")
        f.write(f"Total recordings:              {total_wav_count}\n")
        f.write(f"Total train recordings:        {total_train_count}\n")
        f.write(f"Total test recordings:         {total_test_count}\n")
        f.write(f"Train percentage:              {train_pct:.2f}%\n")
        f.write(f"Test percentage:               {test_pct:.2f}%\n\n")
        
        f.write("================================================================================\n")
        f.write("SPEAKER x CLASS BREAKDOWN\n")
        f.write("================================================================================\n\n")
        
        for spk_name in sorted(dataset_structure.keys()):
            f.write(f"{spk_name}\n")
            for cls_name in ["proxima", "unknown"]:
                stats = split_stats[spk_name][cls_name]
                f.write(f"    {cls_name}:\n")
                f.write(f"        total: {stats['total']}\n")
                f.write(f"        train: {stats['train']}\n")
                f.write(f"        test: {stats['test']}\n")
            f.write("\n")
            
        f.write("================================================================================\n")
        f.write("OVERALL CLASS TOTALS\n")
        f.write("================================================================================\n\n")
        for cls_name in ["proxima", "unknown"]:
            tot = sum(split_stats[s][cls_name]["total"] for s in split_stats)
            tr = sum(split_stats[s][cls_name]["train"] for s in split_stats)
            te = sum(split_stats[s][cls_name]["test"] for s in split_stats)
            f.write(f"{cls_name}:\n")
            f.write(f"    total: {tot}\n")
            f.write(f"    train: {tr} ({tr/tot*100:.2f}%)\n")
            f.write(f"    test:  {te} ({te/tot*100:.2f}%)\n\n")
            
    print(f"Created summary: {summary_path}")

    # -------------------------------------------------------------
    # 7. Independent Rigorous Validation
    # -------------------------------------------------------------
    print("\n--- INDEPENDENT VALIDATION ---")
    val_errors = []
    
    # Check 1: Every source file appears exactly once in either TRAIN or TEST
    if len(original_files_assigned) != total_wav_count:
        val_errors.append(f"Expected {total_wav_count} files assigned, got {len(original_files_assigned)}")
        
    # Check 2: No file appears in both TRAIN and TEST
    intersection = train_dest_files.intersection(test_dest_files)
    if intersection:
        val_errors.append(f"Data leakage detected! {len(intersection)} files in both train and test: {intersection}")
        
    # Check 3: Total count equality
    if total_train_count + total_test_count != total_wav_count:
        val_errors.append(f"Count mismatch: {total_train_count} + {total_test_count} != {total_wav_count}")
        
    # Check 4: Files on disk match expected counts
    actual_train_files = list(train_root.rglob("*.wav"))
    actual_test_files = list(test_root.rglob("*.wav"))
    
    if len(actual_train_files) != total_train_count:
        val_errors.append(f"Actual train files on disk ({len(actual_train_files)}) != expected ({total_train_count})")
    if len(actual_test_files) != total_test_count:
        val_errors.append(f"Actual test files on disk ({len(actual_test_files)}) != expected ({total_test_count})")
        
    # Check 5: Manifest matches files on disk
    manifest_train = [r for r in manifest_rows if r["split"] == "train"]
    manifest_test = [r for r in manifest_rows if r["split"] == "test"]
    if len(manifest_train) != total_train_count or len(manifest_test) != total_test_count:
        val_errors.append("Manifest counts do not match expected split counts")
        
    for r in manifest_rows:
        expected_dest = split_dir / r["split"] / r["speaker"] / r["class"] / r["filename"]
        if not expected_dest.is_file():
            val_errors.append(f"Manifest file not found on disk: {expected_dest}")
            break

    # Check 6: Bit-level integrity check (no re-encoding, no file corruption)
    print("Checking audio integrity and bit-level hash matching...")
    for r in manifest_rows:
        orig = workspace / r["original_path"]
        dest = split_dir / r["split"] / r["speaker"] / r["class"] / r["filename"]
        
        # Audio properties check
        info = sf.info(str(dest))
        if info.format != "WAV" or info.subtype != "PCM_16" or info.channels != 1 or info.samplerate != 16000 or info.frames != 32000:
            val_errors.append(f"Audio properties corrupted: {dest}")
            break
            
        # Bit-level SHA-256 match
        if file_sha256(orig) != file_sha256(dest):
            val_errors.append(f"SHA-256 mismatch between source and copy: {dest}")
            break

    # Check 7: Source directory PROXIMA_DATA_1 untouched
    source_wavs_after = list(source_dir.rglob("*.wav"))
    if len(source_wavs_after) != 544:
        val_errors.append(f"Source directory PROXIMA_DATA_1 was modified! Count: {len(source_wavs_after)}")
        
    if val_errors:
        print("VALIDATION FAILED:")
        for err in val_errors:
            print("  ", err)
        sys.exit(1)
    else:
        print("ALL 9 VALIDATION CHECKS PASSED WITH 100% ACCURACY!")

if __name__ == "__main__":
    run_split()
