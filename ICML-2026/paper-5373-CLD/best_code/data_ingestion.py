import os
import sys
import csv
import uuid
import shutil
import random
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Tuple
from pydub import AudioSegment
import numpy as np
from scipy.signal import resample
import noisereduce as nr  
from datasets import Dataset, Features, Value, Audio
from tqdm import tqdm

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, DatasetDict

from audiomentations import (
    Compose, TimeStretch, Gain, PitchShift, OneOf, 
    AddBackgroundNoise, AddGaussianNoise, PolarityInversion
)

@dataclass
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


PEAK_DBFS_MIN = -10  # Lower bound for peak dBFS
PEAK_DBFS_MAX = -7   # Upper bound for peak dBFS

MAX_INPUT_LENGTH = 30
MAX_LABEL_LENGTH = 448 # whisper max tokens=448

TARGET_SR = 16000

COMMON_VOICE_USED = {}

### HELPER FUNCS

def measure_peak_dbfs(audio):
    # audio = audio[:, 0]  # Use only one channel if stereo
    peak_amplitude = np.max(np.abs(audio))  # Find peak amplitude

    # Check if the audio is integer-based (e.g., int16, int32)
    if np.issubdtype(audio.dtype, np.integer):
        max_possible_amplitude = np.iinfo(audio.dtype).max  # e.g., 32767 for int16
    else:
        max_possible_amplitude = 1.0  # Float WAVs are usually in range [-1,1]

    # Compute peak dBFS relative to max amplitude
    peak_dbfs = 20 * np.log10(peak_amplitude / max_possible_amplitude) if peak_amplitude > 0 else -np.inf
    return peak_dbfs

### AUDIO PROCESSOR FUNCS

def normalize_audio(audio):
    array = audio["array"]
    sr = audio["sampling_rate"]

    # Convert to mono if stereo
    if len(array.shape) > 1:
        array = np.mean(array, axis=0)

    # Resample if needed
    waveform = torch.from_numpy(array).unsqueeze(0).to(torch.float32)  # (1, time)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )
        audio["sampling_rate"] = TARGET_SR

    array = waveform.squeeze().numpy().astype(np.float32)

    current_peak_dbfs = measure_peak_dbfs(array)

    if PEAK_DBFS_MIN <= current_peak_dbfs <= PEAK_DBFS_MAX:
        audio["array"] = array
        return audio

    # Determine the target peak dBFS
    target_peak_dbfs = PEAK_DBFS_MAX if current_peak_dbfs < PEAK_DBFS_MIN else PEAK_DBFS_MIN
    gain = target_peak_dbfs - current_peak_dbfs  # Gain in dB to apply

    # Convert to int16 for pydub
    int16_array = (array * 32767).astype(np.int16)

    # Load audio and apply gain
    audio_seg = AudioSegment(
        data=int16_array.tobytes(),
        frame_rate=audio["sampling_rate"],
        sample_width=2,  # 2 bytes for int16
        channels=1
    )
    audio_seg = audio_seg.apply_gain(gain)

    # Convert back to float32
    new_int16 = np.frombuffer(audio_seg.raw_data, dtype=np.int16)
    new_array = (new_int16 / 32767).astype(np.float32)

    audio["array"] = new_array
    return audio

def reduce_noise(audio):
    array = audio["array"]
    sr = audio["sampling_rate"]

    # Convert to mono if stereo
    if len(array.shape) > 1:
        array = np.mean(array, axis=0)

    if np.max(np.abs(array)) < 1e-4:  # If audio is too quiet, skip noise reduction
        print('Skipping noise reduction due to low signal level.')
        audio["array"] = array
        return audio

    # Avoid divide-by-zero 
    array = array + np.random.normal(0, 1e-6, array.shape)

    audio["array"] = nr.reduce_noise(y=array, sr=sr, prop_decrease=0.85)
    return audio

### FILTERS FUNCS

def is_audio_in_length_range(audio):
    length = len(audio["array"]) / audio["sampling_rate"]
    return length <= MAX_INPUT_LENGTH

# def is_labels_in_length_range(labels):
#     return len(labels) < MAX_LABEL_LENGTH

# by max input length (30s)

### LOADER FUNCS

# func(lang, accent, accent_config, common_voice_dir) -> iter

def load_common_voice(lang, accent_config, common_voice_dir=None):
    if common_voice_dir is None:
        raise ValueError("common_voice_dir is required for local Common Voice loading")
    tsv_path = os.path.join(common_voice_dir, accent_config.get("override_code") if accent_config.get("override_code") else lang, "validated.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"validated.tsv not found at {tsv_path}")

    global COMMON_VOICE_USED
    if tsv_path not in COMMON_VOICE_USED:
        # Initialize usage tracking per TSV file (exclude header)
        num_lines = sum(1 for _ in open(tsv_path, 'r', encoding='utf-8'))
        COMMON_VOICE_USED[tsv_path] = [False] * max(0, num_lines - 1)
    
    column_name = accent_config.get("column_name")
    code = accent_config.get("code")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            accents = row.get('accents', '').split(',')
            if column_name not in accents and column_name != "":
                continue
            audio_path = os.path.join(common_voice_dir, accent_config.get("override_code") if accent_config.get("override_code") else lang, "clips", row['path'])
            if not os.path.exists(audio_path):
                continue
            if COMMON_VOICE_USED[tsv_path][i]:
                continue
            try:
                waveform, sr = torchaudio.load(audio_path)
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                array = waveform.squeeze().numpy().astype(np.float32)
                text = row['sentence']
                COMMON_VOICE_USED[tsv_path][i] = True
                yield {
                    "audio": {
                        "array": array,
                        "sampling_rate": sr,
                    },
                    "text": text,
                    "lang": lang,
                    "accent": code,
                }
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

def load_lahaja(lang, accent_config, common_voice_dir=None):
    ds = load_dataset("ai4bharat/Lahaja", split="test")

    for ex in ds:
        audio_data = ex.get("audio_filepath", None)  # Changed from "audio_filepath" to "audio"
        if audio_data is not None and isinstance(audio_data, dict):
            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate")
            if array is None or sr is None:
                continue
        else:
            continue
        text = ex.get("text", "")
        lang = ex.get("lang", "")
        native_language = ex.get("native_language", "")
        
        if(native_language != accent_config.get("column_name")):
            continue
            
        yield {
            "audio": {
                "array": array,
                "sampling_rate": sr,
            },
            "text": text,
            "lang": lang,
            "accent": accent_config.get("code"),
        }


LOADER_FUNC_MAPPING = {
    "common_voice": load_common_voice,
    "lahaja": load_lahaja
}

### INGESTION CORE


def parse_json_config(cfg_path: str) -> Dict:
    import json
    with open(cfg_path, "r") as f:
        data = json.load(f)
    return data

def ingest(config, out_path):
    """Ingests all the datasets in cfg and outputs 3 files in the directory (train.parquet, val.parquet, test.parquet)"""

    params = config.get("params", {})
    samples_per_class = params.get("samples_per_class")
    split_cfg = params.get("split", None)
    split = SplitRatios(**split_cfg) if isinstance(split_cfg, dict) else SplitRatios()


    if config.get("augment"):
        augmentation = Compose([
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.1),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            OneOf([
                AddBackgroundNoise(
                    sounds_path=config.musan_dir,
                    min_snr_db=1.0,
                    max_snr_db=5.0,
                    noise_transform=PolarityInversion(),
                    p=1.0
                ),
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
            ], p=0.2),
        ])
    else:
        augmentation = None

    processed_samples = []

    for lang_code, lang_dict in config.get("languages", {}).items():
        samples_per_accent = round(samples_per_class/len(lang_dict.get("accents", [])) if samples_per_class is not None else 1e9)
        for accent_params in lang_dict.get("accents", []):
            loader_func = LOADER_FUNC_MAPPING.get(accent_params.get("dataset"))
            if(not loader_func) :
                print(f"WARNING: unkonwn dataset {accent_params.get('dataset')}")
            
            iterable = loader_func(lang_code, accent_params, config.get("common_voice_dir"))
    
            # FILTER BY LENGTH
            filtered = []
            print(f'Ingesting {lang_code}_{accent_params.get("code")} from {accent_params.get("dataset")} dataset')
            for a in iterable:
                if len(filtered) == samples_per_accent:
                    break
                if(is_audio_in_length_range(a["audio"])):
                    filtered.append(a)
            
            random.shuffle(filtered)

            cleaned = []


            for a in tqdm(filtered):
                # Resample to 16khz
                a["audio"] = normalize_audio(a["audio"])
                a["audio"] = reduce_noise(a["audio"])

                if config.get("augment"):
                    a["audio"]["array"] = augmentation(a["audio"]["array"], sample_rate=TARGET_SR)
                    
                cleaned.append(a)
            
            processed_samples.extend(cleaned)

    features = Features({
        "audio": Audio(sampling_rate=TARGET_SR),
        "text": Value("string"),
        "lang": Value("string"),
        "accent": Value("string")
    })

    ds = Dataset.from_dict({
        "audio": [sample["audio"] for sample in processed_samples],
        "text": [sample["text"] for sample in processed_samples],
        "lang": [sample["lang"] for sample in processed_samples],
        "accent": [sample["accent"] for sample in processed_samples]
    }, features=features)

    ds_train_devtest = ds.train_test_split(test_size=split.test+split.val, seed=42)
    ds_devtest = ds_train_devtest['test'].train_test_split(test_size=split.test/(split.test+split.val), seed=42)


    ds_splits = DatasetDict({
        'train': ds_train_devtest['train'],
        'valid': ds_devtest['train'],
        'test': ds_devtest['test']
    })

    ds_splits.save_to_disk(out_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data ingestion pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--augment", action="store_true", help="Run augmentation or not")
    parser.add_argument("--musan-dir", type=str, required=False, help="Musan directory")
    parser.add_argument("--common-voice-dir", type=str, default=None, help="Path to Common Voice directory")
    args = parser.parse_args()

    cfg = parse_json_config(args.config)
    if args.common_voice_dir:
        cfg["common_voice_dir"] = args.common_voice_dir
    ingest(cfg, args.out)


if __name__ == "__main__":
    main()