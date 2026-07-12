#!/usr/bin/env python3
"""Fast cached CREMAD dataset using precomputed features."""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CramedDatasetFast(Dataset):
    """Cached version of CramedDataset with precomputed spectrograms and image tensors."""

    def __init__(self, mode='train', data_root=None):
        self.specs = []
        self.images = []
        self.labels = []
        self.indices = []
        self.mode = mode

        if data_root is None:
            data_root = os.environ.get('CREMAD_DATA_ROOT', '/datasets/CREMA-D')
        self.data_root = data_root
        self.cache_dir = os.path.join(data_root, "precomputed")

        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_feature_path = data_root
        self.audio_feature_path = os.path.join(data_root, 'AudioWAV')

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        # Same transforms as original
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        csv_files = [self.train_csv, self.test_csv]
        idx_counter = 0
        for file_path in csv_files:
            with open(file_path, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)
                for item in csv_reader:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])

                    # Check precomputed files
                    spec_file = os.path.join(self.cache_dir, f"{item[0]}_audio.npy")
                    img_file = os.path.join(self.cache_dir, f"{item[0]}_visual.pt")

                    if os.path.exists(spec_file) and os.path.exists(img_file):
                        self.specs.append(spec_file)
                        self.images.append(visual_path)  # Still need original path for random frame selection
                        self.labels.append(class_dict[item[1]])
                        self.indices.append(idx_counter)
                        idx_counter += 1
                    elif os.path.exists(audio_path) and os.path.exists(visual_path):
                        # Fallback to original path if precomputed missing
                        self.specs.append(None)  # Will compute on-the-fly
                        self.images.append(visual_path)
                        self.labels.append(class_dict[item[1]])
                        self.indices.append(idx_counter)
                        idx_counter += 1

        print(f"CramedDatasetFast: {len(self.specs)} samples loaded")
        cached = sum(1 for s in self.specs if s is not None)
        print(f"  Cached: {cached}, On-the-fly: {len(self.specs) - cached}")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        import librosa

        # Audio
        spec_file = self.specs[idx]
        if spec_file is not None:
            spectrogram = np.load(spec_file)
        else:
            # Fallback: compute on-the-fly
            audio_name = os.path.basename(self.images[idx])
            # Actually need audio path... Let's just load from precomputed always
            raise RuntimeError("Precomputed data missing, run precompute_cremad.py first")

        # Visual - randomly select 1 frame
        image_samples = os.listdir(self.images[idx])
        select_index = np.random.choice(len(image_samples), size=1, replace=False)
        images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            img = Image.open(os.path.join(self.images[idx], image_samples[select_index[i]])).convert('RGB')
            img = self.transform(img)
            images[i] = img
        images = torch.permute(images, (1, 0, 2, 3))

        label = self.labels[idx]
        return idx, spectrogram, images, label
