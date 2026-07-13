# solve/whisper_dataloader.py
# Standalone dataloader for binary language detection on HF dataset with Whisper features
# Loads local HF dataset, extracts pooled encoder hidden states on-the-fly, returns JAX arrays
# Binary: +1 for target_lang (e.g., 'en'), -1 otherwise
# For efficiency: Extracts all train data into memory (assume <10k samples); splits 80/20
# Caller: 'defrun' for 90% usage (as in original)

import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperModel
import torch
import torchaudio
from typing import Tuple
import os

def load_data(dataset_path: str, target_lang: str = None, caller_script: str = None, data_seed: int = 42, dataset_split: str = "train") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
    """
    Load HF dataset, extract pooled Whisper hidden states, return train/test splits.
    
    Args:
        dataset_path (str): Path to local HF dataset dir (splits: train, valid, test).
        target_lang (str): POS language code (e.g., 'en').
        caller_script (str): 'defrun' for 90% data (convex training); else full.
        data_seed (int): Seed for shuffle/split.
    
    Returns:
        Atr, ytr, Atst, ytst, ntr, ntst: JAX arrays for features/labels (pooled to 768 dim).
    """
    np.random.seed(data_seed)
    
    # Load train split (main data for training)
    dataset = load_from_disk(dataset_path)
    train_data = dataset[dataset_split]
    print(f"Loaded {len(train_data)} train samples")
    
    # Load Whisper encoder
    model_name = 'openai/whisper-small'
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    def extract_pooled_hidden(audio) -> np.ndarray:
        """Extract and pool last hidden states to (768,)."""
        # Handle audio dict or path
        if isinstance(audio, dict):
            if audio.get('array') is not None:
                audio_arr = audio['array']
                sr = audio['sampling_rate']
            else:
                audio_path = audio['path']
                if not os.path.exists(audio_path):
                    return None
                waveform, sr = torchaudio.load(audio_path)
                audio_arr = waveform.mean(0).numpy()
        else:
            # Assume path if not dict
            if not os.path.exists(audio):
                return None
            waveform, sr = torchaudio.load(audio)
            audio_arr = waveform.mean(0).numpy()
        
        # Resample to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_arr = resampler(torch.tensor(audio_arr)).numpy()
        
        # Process to input_features
        inputs = processor(audio_arr, sampling_rate=16000, return_tensors='pt').to(device)
        
        # Encoder last hidden
        with torch.no_grad():
            encoder_outputs = model.encoder(inputs.input_features, output_hidden_states=True)
            hidden = encoder_outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
        
        # Pool: Mean over seq_len
        pooled = hidden.mean(0).cpu().numpy()  # (768,)
        return pooled
    
    # Extract features and labels for all train samples
    features = []
    labels = []
    valid_count = 0

    classes = set(sample['lang'] for sample in train_data)
    classes_mapping = {lang: i for i, lang in enumerate(sorted(classes))}
    for sample in train_data:
        hidden = extract_pooled_hidden(sample['audio'])
        if hidden is None:
            continue  # Skip invalid audio
        
        if len(classes) == 2:
            # binary
            label = 1.0 if sample['lang'] == target_lang else -1.0
        else:
            # multiclass
            label = classes_mapping[sample['lang']]

        features.append(hidden)
        labels.append(label)
        valid_count += 1
    
    if valid_count == 0:
        raise ValueError("No valid audio samples found")
    
    # print(f"Extracted {valid_count} valid samples: {np.sum(np.array(labels) == 1)} POS, {np.sum(np.array(labels) == -1)} NEG")
    
    # To JAX
    A = jnp.array(features)  # (n, 768)
    y = jnp.array(labels)    # (n,)
    
    # Shuffle
    n = A.shape[0]
    perm = np.random.permutation(n)
    A = A[perm]
    y = y[perm]

    return A, y, len(classes)
    
    # Split logic
    # if caller_script == "defrun":
    #     # 90% for convex training (80/20 within)
    #     split_idx = int(0.9 * n)
    #     ntr = int(0.8 * split_idx)
    #     ntst = split_idx - ntr
    #     Atr = A[:ntr]
    #     Atst = A[ntr:split_idx]
    #     ytr = y[:ntr]
    #     ytst = y[ntr:split_idx]
    # else:
    #     # Full 80/20
    #     ntr = int(0.8 * n)
    #     ntst = n - ntr
    #     Atr = A[:ntr]
    #     Atst = A[ntr:]
    #     ytr = y[:ntr]
    #     ytst = y[ntr:]
    
    # print(f"Train: {ntr} samples, Test: {ntst} samples")
    # return Atr, ytr, Atst, ytst, ntr, ntst