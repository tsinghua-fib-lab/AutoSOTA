import os
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OxfordPetDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_index, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_index = label_to_index
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        label_idx = self.label_to_index[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx


def load_oxfordpet_datasets(
    dataset_path, 
    test_split=0.05, 
    image_size=256, 
    batch_size=16,
    num_workers=4,
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225],
    random_flip=False,
    val_only: bool = False,
) -> Tuple[Optional[DataLoader], DataLoader]:
    all_images = sorted([os.path.join(dataset_path, img) for img in os.listdir(dataset_path)
                       if img.endswith(('.jpg'))])

    labels = ['_'.join(os.path.basename(f).split('_')[:-1]) for f in all_images]
    
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, labels, test_size=test_split, random_state=42
    )

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if val_only:
        print(f"Dataset statistics (val only):")
        print(f"  Validation images: {len(test_images)}")
        
        test_dataset = OxfordPetDataset(test_images, test_labels, label_to_index, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # Since the full validation dataset has only 370 samples, it doesn’t really matter whether you shuffle it or not—the results will be basically the same.
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        return None, test_loader

    print(f"Dataset statistics:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(test_images)}")
    
    if random_flip:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    train_dataset = OxfordPetDataset(train_images, train_labels, label_to_index, transform=train_transform)
    test_dataset = OxfordPetDataset(test_images, test_labels, label_to_index, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # Since the full validation dataset has only 370 samples, it doesn’t really matter whether you shuffle it or not—the results will be basically the same.
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader
