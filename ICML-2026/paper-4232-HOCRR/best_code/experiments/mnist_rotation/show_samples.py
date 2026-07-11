"""
Show samples from the clean MNIST rotation dataset.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset_generator import load_mnist_rotation_datasets

def show_rotation_samples():
    """Display rotated MNIST samples with proper backgrounds."""
    
    # Load dataset
    train_loader, test_loader = load_mnist_rotation_datasets(
        rotation_range=(0.0, 360.0),
        augmentation_factor=1,
        batch_size=32,
        seed=42
    )
    
    print("Showing clean rotation samples...")
    print(f"Dataset size: {len(test_loader.dataset)}")
    
    # Show 9 samples in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(9):
        image, angle = test_loader.dataset[i]
        original_label = test_loader.dataset.get_original_label(i)
        
        # Convert to numpy for display
        img_np = image.squeeze().numpy()
        
        # Plot the image
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'Angle: {angle:.1f}°\nOriginal: {original_label}')
        axes[i].axis('off')
        
        # Print corner values
        print(f"Sample {i}: Angle {angle:.1f}°, Original {original_label}")
        print(f"  Corner pixels: {img_np[0,0]:.3f}, {img_np[0,-1]:.3f}, {img_np[-1,0]:.3f}, {img_np[-1,-1]:.3f}")
        print(f"  Min/Max: {img_np.min():.3f} / {img_np.max():.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('rotation_samples_clean.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Clean rotation samples saved to rotation_samples_clean.png")
    print("Notice: All corners should be the same gray value (-0.004)!")

if __name__ == "__main__":
    show_rotation_samples()