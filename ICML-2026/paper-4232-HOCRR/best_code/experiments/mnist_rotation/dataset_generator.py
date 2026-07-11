"""
Clean MNIST rotation dataset generator.
Uses the original MNIST background color (gray) to fill rotation corners.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random

class MNISTRotationDataset(Dataset):
    """Dataset for MNIST rotation angle prediction."""
    
    def __init__(self, original_dataset, rotation_range=(0.0, 360.0), augmentation_factor=1, seed=42):
        self.original_dataset = original_dataset
        self.rotation_range = rotation_range
        self.augmentation_factor = augmentation_factor
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Standard MNIST transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Generate rotated samples
        self.samples = self._generate_rotated_samples()
        
    def _generate_rotated_samples(self):
        """Generate rotated samples with proper background filling."""
        samples = []
        
        print(f"Generating {len(self.original_dataset) * self.augmentation_factor} rotated samples...")
        
        for i in range(len(self.original_dataset)):
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(self.original_dataset)}")
                
            # Get original image and label
            original_image, original_label = self.original_dataset[i]
            
            # Generate multiple rotations for this image
            for _ in range(self.augmentation_factor):
                # Random rotation angle
                angle = random.uniform(*self.rotation_range)
                
                # Rotate the image with proper background filling
                rotated_image = self._rotate_image(original_image, angle)
                
                # Apply transforms
                rotated_tensor = self.transform(rotated_image)
                
                samples.append({
                    'image': rotated_tensor,
                    'angle': angle,
                    'original_label': original_label
                })
        
        print(f"Generated {len(samples)} rotated samples")
        return samples
    
    def _rotate_image(self, image, angle):
        """Rotate image with proper background filling using original MNIST background color."""
        # Convert to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Convert tensor back to PIL for rotation
            # First denormalize if needed
            if image.min() < 0:  # If normalized
                image = (image * 0.3081) + 0.1307  # Denormalize
            image = transforms.ToPILImage()(image)
        
        # Use the original MNIST background color (gray)
        # MNIST background is typically around 0.13 normalized, which is ~33/255
        # But we want the actual gray value that looks like MNIST background
        background_color = 33  # This gives us the proper MNIST gray background
        
        # Rotate with proper background fill - use expand=True to avoid clipping
        rotated_image = image.rotate(angle, fillcolor=background_color, expand=True)
        
        # Resize back to 28x28 to maintain consistent tensor shapes
        rotated_image = rotated_image.resize((28, 28), Image.BILINEAR)
        
        return rotated_image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['image'], sample['angle']
    
    def get_original_label(self, idx):
        """Get the original MNIST label for a sample."""
        return self.samples[idx]['original_label']

def load_mnist_rotation_datasets(rotation_range=(0.0, 360.0), augmentation_factor=1, batch_size=256, seed=42):
    """Load MNIST rotation datasets."""
    
    # Load original MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create rotation datasets
    train_rotation = MNISTRotationDataset(
        train_dataset, rotation_range, augmentation_factor, seed
    )
    
    test_rotation = MNISTRotationDataset(
        test_dataset, rotation_range, augmentation_factor, seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_rotation, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_rotation, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test the dataset generation
    print("Testing MNIST rotation dataset generation...")
    
    # Load datasets with minimal augmentation for testing
    train_loader, test_loader = load_mnist_rotation_datasets(
        rotation_range=(0.0, 360.0),
        augmentation_factor=1,  # Just 1x for testing
        batch_size=32,
        seed=42
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test a few samples
    for i in range(3):
        image, angle = test_loader.dataset[i]
        original_label = test_loader.dataset.get_original_label(i)
        print(f"Sample {i}: Angle {angle:.1f}°, Original {original_label}")
        print(f"  Image shape: {image.shape}, Range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Check corner values
        img_np = image.squeeze().numpy()
        print(f"  Corner pixels: {img_np[0,0]:.3f}, {img_np[0,-1]:.3f}, {img_np[-1,0]:.3f}, {img_np[-1,-1]:.3f}")
        print()