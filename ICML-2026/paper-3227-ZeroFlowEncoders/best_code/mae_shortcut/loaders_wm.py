# %%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt # Added for visualization check

class ColoredMNISTLoader:
    def __init__(self, colored=True, batch_size=32, device='cuda', root='./data', seed=42):
        self.batch_size = batch_size
        self.device = device
        
        print("Loading MNIST to memory...")
        
        # --- 1. Load Training Data ---
        raw_train = datasets.MNIST(root=root, train=True, download=True)
        # Scale to [0,1], Add Channel Dim [N, 1, 28, 28], Move to GPU
        self.data = raw_train.data.float().unsqueeze(1).to(device) / 255.0
        self.labels = raw_train.targets.to(device)
        
        # --- 2. Load Testing Data ---
        raw_test = datasets.MNIST(root=root, train=False, download=True)
        self.test_data = raw_test.data.float().unsqueeze(1).to(device) / 255.0
        self.test_labels = raw_test.targets.to(device)

        # --- 3. Resize Images to 32x32 ---
        # (Standard for ResNet-like architectures to avoid size errors)
        self.transform = torch.nn.Sequential(
            transforms.Resize((32, 32)),
        ).to(device)
        
        # Cache resized data
        self.data = self.transform(self.data)
        self.test_data = self.transform(self.test_data)
        
        # --- 4. Generate FIXED Random Colors (The "Loud" Shortcut) ---
        # Each image ID gets a unique, permanent RGB signature.
        torch.manual_seed(seed) # Fix seed so colors are consistent across runs
        self.color_data = torch.rand(self.data.size(0), 3, 1, 1, device=self.device)
        self.color_test_data = torch.rand(self.test_data.size(0), 3, 1, 1, device=self.device)
        if colored:
            self.data = self._applycolor(self.data, self.color_data)
            self.test_data = self._applycolor(self.test_data, self.color_test_data)
        else:
            # If not colored, just repeat grayscale to 3 channels
            self.data = self.data.repeat(1, 3, 1, 1)
            self.test_data = self.test_data.repeat(1, 3, 1, 1)
        
        self._init_augmentations()
        
        print(f"Loaded {len(self.data)} Train images and {len(self.test_data)} Test images to {device}")

    def _applycolor(self, x, color):
        """Applies the given color to the grayscale image x."""
        # x: [B, 1, H, W], color: [B, 3, 1, 1]
        x_colored = x.repeat(1, 3, 1, 1) * color
        return x_colored

    def _init_augmentations(self):
        # Standard SimCLR spatial augmentations (crops/flips)
        self.augment = torch.nn.Sequential(
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5), # Optional for digits, but standard for SSL
        ).to(self.device)

    def _process_batch(self, x_clean, labels):
        """
        Applies augmentation and then fuses the image with its assigned color.
        """
        # 1. Augment spatially (still 1 channel grayscale) -> [B, 1, 32, 32]
        view1 = self.augment(x_clean)
        view2 = self.augment(x_clean)
        
        # 4. Add Noise (Optional but recommended for robust testing)
        # Prevents the model from just matching exact pixel values
        view1 = view1 + torch.randn_like(view1) * 0.05
        view2 = view2 + torch.randn_like(view2) * 0.05
        
        # Clamp to valid image range [0, 1]
        view1 = torch.clamp(view1, 0, 1)
        view2 = torch.clamp(view2, 0, 1)
        
        # Flatten
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, labels

    def get_batch(self):
        """Returns a random batch from the TRAINING set."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        
        x_clean = self.data[indices]
        labels = self.labels[indices]
        
        return self._process_batch(x_clean, labels)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        
        x_clean = self.test_data[indices]
        labels = self.test_labels[indices]
        
        return self._process_batch(x_clean, labels)
    
    def get_label_text(self, label):
        """Returns human-readable text for a given label."""
        return str(label)
    
    def get_mae_batch(self):
        """
        Return a batch for MAE training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]

        # reuse the same logic as contrastive, but only take ONE view
        view1, _, _ = self._process_batch(x_clean, labels)

        # unflatten back to image
        view1 = view1.view(self.batch_size, 3, 32, 32)

        return view1
    
    def get_mae_test_batch(self):
        v1, v2, y = self.get_test_batch()
        v1 = v1.view(self.batch_size, 3, 32, 32)
        return v1, y
    
    def get_mae_wmlp_batch(self):
        """
        Return a batch for MAE linear probe training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]

        view1,_, labels = self._process_batch(x_clean, labels)
        view1 = view1.view(self.batch_size, 3, 32, 32)
        return view1, labels


class ColoredFMNISTLoader:
    def __init__(self, colored=True, batch_size=32, device='cuda', root='./data', seed=42):
        self.batch_size = batch_size
        self.device = device
        
        print("Loading FashionMNIST to memory...")
        
        # --- 1. Load Training Data ---
        raw_train = datasets.FashionMNIST(root=root, train=True, download=True)
        # Scale to [0,1], Add Channel Dim [N, 1, 28, 28], Move to GPU
        self.data = raw_train.data.float().unsqueeze(1).to(device) / 255.0
        self.labels = raw_train.targets.to(device)
        
        # --- 2. Load Testing Data ---
        raw_test = datasets.FashionMNIST(root=root, train=False, download=True)
        self.test_data = raw_test.data.float().unsqueeze(1).to(device) / 255.0
        self.test_labels = raw_test.targets.to(device)

        # --- 3. Resize Images to 32x32 ---
        # (Standard for ResNet-like architectures to avoid size errors)
        self.transform = torch.nn.Sequential(
            transforms.Resize((32, 32)),
        ).to(device)
        
        # Cache resized data
        self.data = self.transform(self.data)
        self.test_data = self.transform(self.test_data)
        
        # --- 4. Generate FIXED Random Colors (The "Loud" Shortcut) ---
        # Each image ID gets a unique, permanent RGB signature.
        torch.manual_seed(seed) # Fix seed so colors are consistent across runs
        self.color_data = torch.rand(self.data.size(0), 3, 1, 1, device=self.device)
        self.color_test_data = torch.rand(self.test_data.size(0), 3, 1, 1, device=self.device)
        if colored:
            self.data = self._applycolor(self.data, self.color_data)
            self.test_data = self._applycolor(self.test_data, self.color_test_data)
        else:
            # If not colored, just repeat grayscale to 3 channels
            self.data = self.data.repeat(1, 3, 1, 1)
            self.test_data = self.test_data.repeat(1, 3, 1, 1)
        
        self._init_augmentations()
        
        print(f"Loaded {len(self.data)} Train images and {len(self.test_data)} Test images to {device}")

    def _applycolor(self, x, color):
        """Applies the given color to the grayscale image x."""
        # x: [B, 1, H, W], color: [B, 3, 1, 1]
        x_colored = x.repeat(1, 3, 1, 1) * color
        return x_colored

    def _init_augmentations(self):
        # Standard SimCLR spatial augmentations (crops/flips)
        self.augment = torch.nn.Sequential(
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5), # Optional for digits, but standard for SSL
        ).to(self.device)

    def _process_batch(self, x_clean, labels):
        """
        Applies augmentation and then fuses the image with its assigned color.
        """
        # 1. Augment spatially (still 1 channel grayscale) -> [B, 1, 32, 32]
        view1 = self.augment(x_clean)
        view2 = self.augment(x_clean)
        
        # 4. Add Noise (Optional but recommended for robust testing)
        # Prevents the model from just matching exact pixel values
        view1 = view1 + torch.randn_like(view1) * 0.05
        view2 = view2 + torch.randn_like(view2) * 0.05
        
        # Clamp to valid image range [0, 1]
        view1 = torch.clamp(view1, 0, 1)
        view2 = torch.clamp(view2, 0, 1)
        
        # Flatten
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, labels

    def get_batch(self):
        """Returns a random batch from the TRAINING set."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        
        x_clean = self.data[indices]
        labels = self.labels[indices]
        
        return self._process_batch(x_clean, labels)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        
        x_clean = self.test_data[indices]
        labels = self.test_labels[indices]
        
        return self._process_batch(x_clean, labels)
    
    def get_label_text(self, label):
        """Returns human-readable text for a given label."""
        return str(label)
    
    def get_mae_batch(self):
        """
        Return a batch for MAE training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]

        # reuse the same logic as contrastive, but only take ONE view
        view1, _, _ = self._process_batch(x_clean, labels)

        # unflatten back to image
        view1 = view1.view(self.batch_size, 3, 32, 32)

        return view1
    
    def get_mae_test_batch(self):
        v1, v2, y = self.get_test_batch()
        v1 = v1.view(self.batch_size, 3, 32, 32)
        return v1, y
    
    def get_mae_wmlp_batch(self):
        """
        Return a batch for MAE linear probe training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]

        view1,_, labels = self._process_batch(x_clean, labels)
        view1 = view1.view(self.batch_size, 3, 32, 32)
        return view1, labels
    
class PatchedCIFARLoader:
    def __init__(self, patched=True, batch_size=32, device='cuda', root='./data', seed=42):
        self.batch_size = batch_size
        self.device = device
        self.patched = patched
        self.patch_size = 8
        
        print("Loading CIFAR-10 to memory...")
        
        # --- 1. Load Training Data ---
        raw_train = datasets.CIFAR10(root=root, train=True, download=True)
        # CIFAR is already [N, 32, 32, 3]. Convert to [N, 3, 32, 32], Scale to [0,1]
        self.data = torch.tensor(raw_train.data).permute(0, 3, 1, 2).float().to(device) / 255.0
        self.labels = torch.tensor(raw_train.targets).to(device)
        
        # --- 2. Load Testing Data ---
        raw_test = datasets.CIFAR10(root=root, train=False, download=True)
        self.test_data = torch.tensor(raw_test.data).permute(0, 3, 1, 2).float().to(device) / 255.0
        self.test_labels = torch.tensor(raw_test.targets).to(device)

        # --- 3. Generate FIXED Random Patches (The "Loud" Shortcut) ---
        # Each image ID gets a unique, permanent RGB patch (8x8 pixels).
        torch.manual_seed(seed) 
        # Shape: [N, 3, 8, 8]
        self.patch_data = torch.rand(self.data.size(0), 3, self.patch_size, self.patch_size, device=self.device)
        self.patch_test_data = torch.rand(self.test_data.size(0), 3, self.patch_size, self.patch_size, device=self.device)
        
        self._init_augmentations()
        
        print(f"Loaded {len(self.data)} Train images and {len(self.test_data)} Test images to {device}")

    def _init_augmentations(self):
        # Standard SimCLR spatial augmentations
        self.augment = torch.nn.Sequential(
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ).to(self.device)

    def _apply_patch(self, views, patches):
        """
        Overwrites the top-left corner of the views with the unique patches.
        views: [B, 3, 32, 32]
        patches: [B, 3, 8, 8]
        """
        # We clone to avoid modifying in place if needed, though usually safe here
        out = views.clone()
        # Overwrite top-left corner
        out[:, :, :self.patch_size, :self.patch_size] = patches
        return out

    def _process_batch(self, x_clean, labels, patches):
        """
        Applies Gradient Starvation Logic:
        1. Dim the Object (Quiet)
        2. Augment
        3. Apply Patch (Loud)
        4. Add Noise (Distraction)
        """
        # --- 1. DIM THE OBJECT (Quiet Feature) ---
        # We suppress the natural image so gradients from the object are tiny.
        x_quiet = x_clean * 1
        
        # --- 2. AUGMENT ---
        view1 = self.augment(x_quiet)
        view2 = self.augment(x_quiet)
        
        # --- 3. APPLY PATCH (Loud Feature) ---
        if self.patched:
            # We apply the patch AFTER augmentation to ensure it's always visible 
            # in the same location (e.g., top-left) acting as a perfect shortcut.
            view1 = self._apply_patch(view1, patches)
            view2 = self._apply_patch(view2, patches)
        
        # --- 4. ADD NOISE ---
        # Noise (0.4) is louder than the object (0.2) but quieter than the patch (1.0).
        noise_level = 0.01
        view1 = view1 + torch.randn_like(view1) * noise_level
        view2 = view2 + torch.randn_like(view2) * noise_level
        
        # Clamp to valid image range [0, 1]
        view1 = torch.clamp(view1, 0, 1)
        view2 = torch.clamp(view2, 0, 1)
        
        # Flatten (Matching your previous interface)
        view1 = view1.view(self.batch_size, -1)
        view2 = view2.view(self.batch_size, -1)
        
        return view1, view2, labels

    def get_batch(self):
        """Returns a random batch from the TRAINING set."""
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        
        x_clean = self.data[indices]
        labels = self.labels[indices]
        patches = self.patch_data[indices]
        
        return self._process_batch(x_clean, labels, patches)

    def get_test_batch(self):
        """Returns a random batch from the TESTING set."""
        indices = torch.randint(0, len(self.test_data), (self.batch_size,), device=self.device)
        
        x_clean = self.test_data[indices]
        labels = self.test_labels[indices]
        patches = self.patch_test_data[indices]
        
        return self._process_batch(x_clean, labels, patches)
    
    def get_label_text(self, label):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return classes[label]
    
    def get_mae_batch(self):
        """
        Return a batch for MAE training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]
        patches = self.patch_data[indices]

        # reuse the same logic as contrastive, but only take ONE view
        view1, _, _ = self._process_batch(x_clean, labels, patches)

        # unflatten back to image
        view1 = view1.view(self.batch_size, 3, 32, 32)

        return view1
    
    def get_mae_test_batch(self):
        v1, v2, y = self.get_test_batch()
        v1 = v1.view(self.batch_size, 3, 32, 32)
        return v1, y
    
    def get_mae_wmlp_batch(self):
        """
        Return a batch for MAE linear probe training:
        shape: (B, 3, 32, 32)
        """
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)

        x_clean = self.data[indices]
        labels = self.labels[indices]
        patches = self.patch_data[indices]

        view1,_, labels = self._process_batch(x_clean, labels, patches)
        view1 = view1.view(self.batch_size, 3, 32, 32)
        return view1, labels

# --- Usage & Visualization ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader1 = ColoredMNISTLoader(batch_size=16, device=device)
    loader2 = ColoredFMNISTLoader(batch_size=16, device=device)
    loader3 = PatchedCIFARLoader(batch_size=16, device=device)

    def vis_loader(loader):
        img = loader.get_mae_batch()[:8]
        fig, ax = plt.subplots(1, 8, figsize=(12, 3))
        for i in range(8):
            ax[i].imshow(img[i].cpu().permute(1, 2, 0).numpy())
            ax[i].axis('off')
        plt.show()

        # Get Train Batch
        v1, v2, y = loader.get_batch()
        print(f"Train Batch Shape: {v1.shape}") # Should be [16, 3, 32, 32]
        
        # Visualize the first few pairs to verify the color shortcut
        fig, axes = plt.subplots(2, 16, figsize=(10, 5))
        
        # Move to CPU for plotting
        v1_cpu = v1.cpu().view(-1, 3, 32, 32).permute(0, 2, 3, 1).numpy()
        v2_cpu = v2.cpu().view(-1, 3, 32, 32).permute(0, 2, 3, 1).numpy()
        
        for i in range(16):
            # Plot View 1
            axes[0, i].imshow(v1_cpu[i])
            axes[0, i].set_title(f"Label: {y[i].item()}\nView 1")
            axes[0, i].axis('off')
            
            # Plot View 2 (Should share the exact same color, but different crop/noise)
            axes[1, i].imshow(v2_cpu[i])
            axes[1, i].set_title("View 2")
            axes[1, i].axis('off')
            
        plt.tight_layout()
        plt.show()

    vis_loader(loader1)
    vis_loader(loader2)
    vis_loader(loader3)
# %%