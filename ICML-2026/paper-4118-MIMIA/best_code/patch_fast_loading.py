#!/usr/bin/env python3
"""Monkey-patch CramedDataset for fast data loading with precomputed spectrograms.

Usage: At the top of main_multimodal.py, add:
    import patch_fast_loading
    patch_fast_loading.patch()
"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CACHE_DIR = os.path.join(os.environ.get("CREMAD_DATA_ROOT", "/datasets/CREMA-D"), "precomputed")

# Store original CramedDataset class
_original_getitem = None


def _fast_getitem(self, idx):
    """Faster __getitem__ that loads precomputed spectrograms."""
    # Load precomputed audio spectrogram
    spec_file = os.path.join(CACHE_DIR, f"{os.path.basename(self.audio[idx].replace('.wav', ''))}_audio.npy")
    if os.path.exists(spec_file):
        spectrogram = np.load(spec_file)
    else:
        # Fallback to original librosa
        import librosa
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

    # Visual - original code path (fast enough)
    if self.mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_samples = os.listdir(self.image[idx])
    select_index = np.random.choice(len(image_samples), size=1, replace=False)
    select_index.sort()
    images = torch.zeros((1, 3, 224, 224))
    for i in range(1):
        img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
        img = transform(img)
        images[i] = img
    images = torch.permute(images, (1, 0, 2, 3))

    label = self.label[idx]
    return idx, spectrogram, images, label


def patch():
    """Apply monkey-patch to CramedDataset."""
    from utils.datasets import CramedDataset

    # Replace __getitem__
    CramedDataset.__getitem__ = _fast_getitem
    print("Patched CramedDataset.__getitem__ for fast loading")
