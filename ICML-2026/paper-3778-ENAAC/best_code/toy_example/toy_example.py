import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset

from captum.attr import IntegratedGradients  # Captum library for gradient-based attribution

def split_dataset_by_class(dataset, target_class_0, target_class_1):
    """
    Splits a PyTorch dataset into two subsets containing only the specified classes.
    """
    indices_class_0 = []
    indices_class_1 = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label == target_class_0:
            indices_class_0.append(idx)
        elif label == target_class_1:
            indices_class_1.append(idx)

    dataset_0 = Subset(dataset, indices_class_0)
    dataset_1 = Subset(dataset, indices_class_1)

    return dataset_0, dataset_1
    
def compute_logits_stats(model, dataloader, device='cpu'):
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)  # shape: [batch_size, 2]
            all_outputs.append(outputs.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)  # shape: [num_samples, 2]
    mean_logits = all_outputs.mean(dim=0)
    std_logits = all_outputs.std(dim=0)

    return mean_logits.numpy(), std_logits.numpy()

# extract lowest and highest patches
def get_lowest_highest_patches(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_size: int,
    patch_size: int = 64,
    stride: int = 16,
    num_patches: int = 4,
    return_images: bool = False):
    """
    Extract patches from images using unfold and find ones that maximize model output.
    
    Args:
        dataloader: PyTorch dataloader containing images
        model: Neural network model to evaluate patches
        output_size: Number of output channels
        patch_size: Size of square patches to extract
        stride: Stride of patches
        num_patches: Number of top patches to keep
        return_images: Specifies if the corresponding images are returned
    Returns:
        Returns patches and the corresponding scores
    """
    model = model.cuda()
    model.eval()
    
    # List to store (patch, score) tuples
    highest_patches_scores = torch.ones((output_size, num_patches)).cuda() * (-float('inf')) # nr_channels(output_size) x nr_patches
    lowest_patches_scores = torch.ones((output_size, num_patches)).cuda() * float('inf')

    highest_patches = torch.ones((output_size, num_patches, 3, patch_size, patch_size)).cuda() # nr_channels(output_size) x nr_patches x patch size(3,H,W)
    lowest_patches = torch.ones((output_size, num_patches, 3, patch_size, patch_size)).cuda() # nr_channels(output_size) x nr_patches x patch size(3,H,W)

    # they store the index of the top activating images relative to the loader dataset
    if return_images:
        highest_images = torch.ones((output_size, num_patches)).cuda() # nr_channels(output_size) x nr_patches 
        lowest_images = torch.ones((output_size, num_patches)).cuda() # nr_channels(output_size) x nr_patches


    with torch.no_grad():
        print('Finding patches...')
        iteration = 0
        for images, label in tqdm(dataloader):
            images = images.cuda()
            
            batch_size, channels, height, width = images.shape
            assert batch_size == 1

            kc, kh, kw = 3, patch_size, patch_size  # kernel size
            dc, dh, dw = 1, stride, stride  # stride
            
            patches = images.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            patches = patches.contiguous().view(-1, kc, kh, kw)
            
            # get model output for this batch of patches
            outputs = model(patches)
            topk_vals, topk_indices = torch.topk(outputs, num_patches, dim=0)
            bottomk_vals, bottomk_indices = torch.topk(outputs, num_patches, dim=0, largest=False)

            topk_vals = topk_vals.permute(1,0) # output_size x num_patches
            bottomk_vals = bottomk_vals.permute(1,0) # output_size x num_patches

            topk_indices = topk_indices.permute(1,0) # output_size x num_patches
            bottomk_indices = bottomk_indices.permute(1,0) # output_size x num_patches

            topk_patches = patches[topk_indices]
            bottomk_patches = patches[bottomk_indices]
            
            # to see if the current topk are better than global ones, we concatenate them, and find topk again
            tmp_scores_highest = torch.cat([highest_patches_scores, topk_vals], dim=1)
            tmp_patches_highest = torch.cat([highest_patches, topk_patches], dim=1)
            if return_images:
                tmp_images_highest = torch.cat([highest_images, torch.tensor([[iteration]]).cuda().repeat(output_size, num_patches)], dim=1)
            top_patches_scores, topk_indices = torch.topk(tmp_scores_highest, num_patches, dim=1)
            
            tmp_scores_lowest = torch.cat([lowest_patches_scores, bottomk_vals], dim=1)
            tmp_patches_lowest = torch.cat([lowest_patches, bottomk_patches], dim=1)
            if return_images:
                tmp_images_lowest = torch.cat([lowest_images, torch.tensor([[iteration]]).cuda().repeat(output_size, num_patches)], dim=1)
            bottom_patches_scores, bottomk_indices = torch.topk(tmp_scores_lowest, num_patches, dim=1, largest=False)

            # expand the indices to allow broadcasting
            batch_indices = torch.arange(output_size).unsqueeze(1).expand(-1, num_patches).cuda()  # Shape: [output_size, num_patches]
            
            top_patches = tmp_patches_highest[batch_indices,topk_indices,:,:,:]
            bottom_patches = tmp_patches_lowest[batch_indices,bottomk_indices,:,:,:]

            if return_images:
                top_images = tmp_images_highest[batch_indices.cpu(),topk_indices.cpu()]
                bottom_images = tmp_images_lowest[batch_indices.cpu(),bottomk_indices.cpu()]


            # update
            highest_patches_scores = top_patches_scores
            lowest_patches_scores = bottom_patches_scores
            highest_patches = top_patches
            lowest_patches = bottom_patches
            if return_images:
                highest_images = top_images
                lowest_images = bottom_images

            iteration += 1
            

    if return_images:
        return bottom_patches, bottom_patches_scores, lowest_images, top_patches, top_patches_scores, highest_images
    else:
        return bottom_patches, bottom_patches_scores, top_patches, top_patches_scores


def main():
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
            self.relu1 = nn.ReLU()
            
            self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):

            x = self.relu1(self.conv1(x))
            x = self.conv2(x)
            x = self.avgpool(x)
            
            x = x.view(x.size(0), -1)
            
            return x

    COLORS = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 0, 1],
        [0.5, 0, 1],
        [1, 0, 0.5],
        [0.5, 0, 0],
        [0, 0, 0.5],  
        [0.5, 0, 0.5],  
    ]

    class BalancedRandomColorDataset(Dataset):
        def __init__(self, num_samples, image_size=32, num_pixels=[8,9,10,11,12], random_seed=None):
            """
            Parameters:
            - num_samples: Total number of samples (will be split equally between the two classes).
            - image_size: Size of the square image (image_size x image_size).
            - num_pixels: Range of number of random pixels in each image (use range so number of non-green pixels doesn't give away class label).
            - random_seed: Seed for reproducibility.
            """
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_pixels = num_pixels
            
            if random_seed is not None:
                np.random.seed(random_seed)
                random.seed(random_seed)
                torch.manual_seed(random_seed)
            
            self.images, self.labels = self._generate_dataset()
        
        def _generate_dataset(self):
            half_samples = self.num_samples // 2  # Equal split between class 0 and class 1
            images = []
            labels = []
            
            for _ in range(half_samples):
                # Generate a class 0 image (contains a green pixel)
                image, label = self._generate_image(contains_green=True)
                images.append(image)
                labels.append(label)
                
                # Generate a class 1 image (does NOT contain a green pixel)
                image, label = self._generate_image(contains_green=False)
                images.append(image)
                labels.append(label)
            
            images = torch.stack(images)
            labels = torch.tensor(labels, dtype=torch.long)
            return images, labels
        
        def _generate_image(self, contains_green):
            """
            Generate a single image based on the given label (contains green or not).
            """
            image = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)  # Start with a black image
            num_pixels = random.choices(self.num_pixels)[0]
            if contains_green:
                # Exclude green from the list of random colors
                non_green_colors = [color for color in COLORS if color != [0, 1, 0]]
                selected_colors = random.choices(non_green_colors, k=num_pixels)
                selected_colors.insert(0, [0, 1, 0])  # Insert green at the first position
            else:
                # Exclude green from the list of random colors
                non_green_colors = [color for color in COLORS if color != [0, 1, 0]]
                selected_colors = random.choices(non_green_colors, k=num_pixels)

            # Randomly place the num_pixels pixels
            pixel_positions = random.sample(
                [(x, y) for x in range(self.image_size) for y in range(self.image_size)],
                k=num_pixels,
            )
            
            for (color, (x, y)) in zip(selected_colors, pixel_positions):
                image[:, y, x] = color  # Set the color at the given position
            
            label = 0 if contains_green else 1
            return torch.tensor(image), label
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # Hyperparameters
    batch_size = 256
    learning_rate = 0.010
    num_epochs = 15

    # Create datasets and data loader
    random_seed=0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    train_dataset = BalancedRandomColorDataset(num_samples=20000, random_seed=0)
    test_dataset = BalancedRandomColorDataset(num_samples=1000, random_seed=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    nr_tries = 5
    best_acc = 0
    best_model = 0
    for i in range(nr_tries): # sometimes it doesn't converge; so train multiple times
        print('Try', i)

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        pos_weight = torch.ones([2]).to(device)  # All weights are equal to 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        def train(model, dataloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels_not_one_hot = labels.clone()
                labels = F.one_hot(labels, num_classes=2).to(device).float()
                
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                running_loss += loss.item() * inputs.size(0)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += predicted.eq(labels_not_one_hot).sum().item()
            
            epoch_loss = running_loss / total
            epoch_accuracy = correct / total
            return epoch_loss, epoch_accuracy
        
        # Evaluation loop
        def evaluate(model, dataloader, criterion, device):
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    labels_not_one_hot = labels.clone()
                    labels = F.one_hot(labels, num_classes=2).to(device).float()
                
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Calculate metrics
                    running_loss += loss.item() * inputs.size(0)
                    predicted = outputs.argmax(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels_not_one_hot).sum().item()
            
            epoch_loss = running_loss / total
            epoch_accuracy = correct / total
            return epoch_loss, epoch_accuracy
        
        # Training and evaluation
        for epoch in range(num_epochs):
            
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model
                
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print('Best Accuracy', best_acc)
    
    print('GET RESULTS')

    print('Weights:')
    print(best_model.conv1.weight)
    print(best_model.conv2.weight)

    print('get average outputs for images of the two classes respectively')
    # Split into class 0 and class 1
    class_0_dataset, class_1_dataset = split_dataset_by_class(test_dataset, 0, 1)
    # DataLoaders
    loader_class0 = DataLoader(class_0_dataset, batch_size=32, shuffle=False)
    loader_class1 = DataLoader(class_1_dataset, batch_size=32, shuffle=False)
    # Compute stats
    class0_mean, class0_std = compute_logits_stats(best_model, loader_class0, device)
    class1_mean, class1_std = compute_logits_stats(best_model, loader_class1, device)
    print(class0_mean, class0_std)
    print(class1_mean, class1_std)

    print('get training examples')

    folder="./visualization"
    prefix="sample"

    os.makedirs(folder, exist_ok=True)

    for i in range(2):
        image, label = test_dataset[i]

        # Convert tensor to numpy image if needed
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)

        filename = os.path.join(folder, f"{prefix}_{i}_label_{label}.png")
        image.save(filename)

    print('get attributions')
    
    prefix="attribution"
    
    for i in range(2):
        image, label = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True
        ig = IntegratedGradients(best_model)
        attribution = ig.attribute(image, target=label)
        
        attribution = attribution.sum(dim=1, keepdim=True)
        attribution = attribution / attribution.abs().max()
        attribution = attribution.detach().cpu().numpy()
        plt.figure(figsize=(2, 2))
        plt.imshow(attribution[0,0], cmap='RdBu', vmin=-1.1, vmax=1.1) # 1.1 range so that the colors are not too dark to distinguish; input is between -1 and 1
        plt.axis('off')
        filename = os.path.join(folder, f"{prefix}_{i}_label_{label}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        #non-target attribution
        image, label = test_dataset[i]
        label = 1 - label
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True
        ig = IntegratedGradients(best_model)
        attribution = ig.attribute(image, target=label)    
        attribution = attribution.sum(dim=1, keepdim=True)
        attribution = attribution / attribution.abs().max()
        attribution = attribution.detach().cpu().numpy()
        plt.figure(figsize=(2, 2))
        plt.imshow(attribution[0,0], cmap='RdBu', vmin=-1.1, vmax=1.1) # 1.1 range so that the colors are not too dark to distinguish; input is between -1 and 1
        plt.axis('off')
        filename = os.path.join(folder, f"{prefix}_{i}_label_{label}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()

    print('get most and least activating patches')

    patch_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    bottom_patches, x, top_patches, y = get_lowest_highest_patches(patch_loader, best_model, 2, 7, 1, 4)

    print('corresponding activations')
    print(x)
    print(y)

    print('average activation of 3 patches reported in paper:')
    
    print(x[0][:3].mean())
    print(x[1][:3].mean())
    
    print(y[0][:3].mean())
    print(y[1][:3].mean())
    

    prefix="patch"

    os.makedirs(folder, exist_ok=True)

    for n in range(2): # output neurons
        for p in range(4): # nr_patches
            patch_bottom = bottom_patches[n, p, :, :, :]

            # Convert tensor to numpy image if needed
            if isinstance(patch_bottom, torch.Tensor):
                patch_bottom = TF.to_pil_image(patch_bottom)

            filename = os.path.join(folder, f"neuron_{n}_{prefix}_{p}_bottom.png")
            patch_bottom.save(filename)

            patch_top = top_patches[n, p, :, :, :]

            # Convert tensor to numpy image if needed
            if isinstance(patch_top, torch.Tensor):
                patch_top = TF.to_pil_image(patch_top)

            filename = os.path.join(folder, f"neuron_{n}_{prefix}_{p}_top.png")
            patch_top.save(filename)

if __name__ == '__main__':
    main()