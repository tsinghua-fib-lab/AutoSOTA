import torch
import matplotlib.pyplot as plt
import os

from captum.attr import IntegratedGradients

import torch
import torch.nn as nn

def save_sequence(sequence, label, save_dir, tag="example", attribution=False):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(sequence):
        plt.figure(figsize=(2, 2))
        if attribution:
            plt.imshow(frame, cmap='RdBu', vmin=-1.1, vmax=1.1) # 1.1 range so that the colors are not too dark to distinguish; input is between -1 and 1
        else:
            plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        fname = f"class{label}_{tag}_frame{i}.png"
        plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight', pad_inches=0.0)
        plt.close()

def create_illustrative_sequences(image_size=12, bar_width=1, shift=1, num_frames=5):
    sequences = {}

    # === Class 0: Same Direction ===
    seq0 = []
    left_start = 0
    #right_start = left_start + shift
    for t in range(num_frames):
        frame = torch.zeros((image_size, image_size))
        frame[:, left_start + t * shift : left_start + t * shift + bar_width] = 1.0
        seq0.append(frame)
    sequences[0] = seq0

    # === Class 1: Opposite Directions ===
    seq1 = []
    left_start = 5
    right_start = 6
    for t in range(num_frames):
        frame = torch.zeros((image_size, image_size))
        left_pos = left_start - t * shift
        right_pos = right_start + t * shift
        frame[:, left_pos:left_pos + bar_width] = 1.0
        frame[:, right_pos:right_pos + bar_width] = 1.0
        seq1.append(frame)
    sequences[1] = seq1

    return sequences

# Generate and save
sequences = create_illustrative_sequences()
save_sequence(sequences[0], label=0, save_dir="visualization", tag="clean")
save_sequence(sequences[1], label=1, save_dir="visualization", tag="clean")

# create model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,2), stride=1, padding=(0,1), bias=False)
        # Create a custom weight tensor with the same shape
        # Shape: (out_channels, in_channels, kernel_height, kernel_width)
        weight = self.conv1.weight.data.clone()
        weight[0][0][0][0] = 0.
        weight[0][0][0][1] = -1.
        weight[0][1][0][0] = 1.
        weight[0][1][0][1] = 0.

        weight[1][0][0][0] = -1.
        weight[1][0][0][1] = 0.
        weight[1][1][0][0] = 0
        weight[1][1][0][1] = 1.
        self.conv1.weight.data = weight

        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        # Create a custom weight tensor with the same shape
        # Shape: (out_channels, in_channels, kernel_height, kernel_width)
        weight = self.conv2.weight.data.clone()
        weight[0][0][0][0] = 1.
        weight[0][1][0][0] = -1

        weight[1][0][0][0] = 0.5
        weight[1][1][0][0] = 0.5
        self.conv2.weight.data = weight

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        return x

model = SimpleCNN()
model = model.eval()

print('Classification')
f1 = sequences[0][0].unsqueeze(0).unsqueeze(0)
f2 = sequences[0][1].unsqueeze(0).unsqueeze(0)
x = torch.cat([f1,f2], dim=1) # 1,2,12,12
print(model(x))
f1 = sequences[1][0].unsqueeze(0).unsqueeze(0)
f2 = sequences[1][1].unsqueeze(0).unsqueeze(0)
x = torch.cat([f1,f2], dim=1) # 1,2,12,12
print(model(x))

ig = IntegratedGradients(model)

# compute target attributions
target_attributions_class0 = []
for i in range(len(sequences[0])-1):
    f1 = sequences[0][i].unsqueeze(0).unsqueeze(0)
    f2 = sequences[0][i+1].unsqueeze(0).unsqueeze(0)
    x = torch.cat([f1,f2], dim=1) # 1,2,12,12
    x.requires_grad = True
    output = model(x)

    attribution = ig.attribute(x, target=[0])
    attribution = attribution[0]
    attribution = attribution / attribution.abs().max()
    attribution = attribution.detach().cpu()

    if len(target_attributions_class0) == 0: # for all only visualize the attribution of the second channel; for the first one also add first channel
        target_attributions_class0.append(attribution[0])
        target_attributions_class0.append(attribution[1]) 
    else:
        target_attributions_class0.append(attribution[1]) 

save_sequence(target_attributions_class0, label=0, save_dir="visualization", tag="target_attribution", attribution=True)

target_attributions_class1 = []
for i in range(len(sequences[0])-1):
    f1 = sequences[1][i].unsqueeze(0).unsqueeze(0)
    f2 = sequences[1][i+1].unsqueeze(0).unsqueeze(0)
    x = torch.cat([f1,f2], dim=1) # 1,2,12,12
    x.requires_grad = True
    output = model(x)

    attribution = ig.attribute(x, target=[1])
    attribution = attribution[0]
    attribution = attribution / attribution.abs().max()
    attribution = attribution.detach().cpu()

    if len(target_attributions_class1) == 0: # for all only visualize the attribution of the second channel; for the first one also add first channel
        target_attributions_class1.append(attribution[0])
        target_attributions_class1.append(attribution[1]) 
    else:
        target_attributions_class1.append(attribution[1]) 

save_sequence(target_attributions_class1, label=1, save_dir="visualization", tag="target_attribution", attribution=True)




# compute non-target attributions
nontarget_attributions_class0 = []
for i in range(len(sequences[0])-1):
    f1 = sequences[0][i].unsqueeze(0).unsqueeze(0)
    f2 = sequences[0][i+1].unsqueeze(0).unsqueeze(0)
    x = torch.cat([f1,f2], dim=1) # 1,2,12,12
    x.requires_grad = True
    output = model(x)

    attribution = ig.attribute(x, target=[1])
    attribution = attribution[0]
    attribution = attribution / attribution.abs().max()
    attribution = attribution.detach().cpu()

    if len(nontarget_attributions_class0) == 0: # for all only visualize the attribution of the second channel; for the first one also add first channel
        nontarget_attributions_class0.append(attribution[0])
        nontarget_attributions_class0.append(attribution[1]) 
    else:
        nontarget_attributions_class0.append(attribution[1]) 

save_sequence(nontarget_attributions_class0, label=0, save_dir="visualization", tag="nontarget_attribution", attribution=True)

nontarget_attributions_class1 = []
for i in range(len(sequences[0])-1):
    f1 = sequences[1][i].unsqueeze(0).unsqueeze(0)
    f2 = sequences[1][i+1].unsqueeze(0).unsqueeze(0)
    x = torch.cat([f1,f2], dim=1) # 1,2,12,12
    x.requires_grad = True
    output = model(x)

    attribution = ig.attribute(x, target=[0])
    attribution = attribution[0]
    attribution = attribution / attribution.abs().max()
    attribution = attribution.detach().cpu()

    if len(nontarget_attributions_class1) == 0: # for all only visualize the attribution of the second channel; for the first one also add first channel
        nontarget_attributions_class1.append(attribution[0])
        nontarget_attributions_class1.append(attribution[1]) 
    else:
        nontarget_attributions_class1.append(attribution[1]) 

save_sequence(nontarget_attributions_class1, label=1, save_dir="visualization", tag="nontarget_attribution", attribution=True)






# find most and least activating patches

# Variables to store best and worst patches and scores
max_patches = [None, None]  # One for each output node
min_patches = [None, None]
max_scores = [float('-inf')] * 2
min_scores = [float('inf')] * 2

for j in range(2):
    sequence = sequences[j]
    for i in range(len(sequence)-1):
        f1 = sequence[i].unsqueeze(0).unsqueeze(0)
        f2 = sequence[i+1].unsqueeze(0).unsqueeze(0)
        x = torch.cat([f1,f2], dim=1) # 1,2,12,12

        # Extract all 5x5 patches using unfold
        patches = x.unfold(2, 6, 1).unfold(3, 6, 1)  # shape: [1, 2, 8, 8, 5, 5]
        patches = patches.squeeze(0)  # shape: [2, 8, 8, 5, 5]
        patches = patches.permute(1, 2, 0, 3, 4)  # shape: [8, 8, 2, 5, 5]
        patches = patches.reshape(-1, 2, 6, 6)  # shape: [64, 2, 5, 5]

        # Iterate through patches
        for patch in patches:
            patch_input = patch.unsqueeze(0)  # shape: [1, 2, 5, 5]
            output = model(patch_input)       # shape: [1, 2]
            output = output.squeeze(0)        # shape: [2]
            
            for i in range(2):  # For each output node
                val = output[i].item()
                if val >= max_scores[i]:
                    max_scores[i] = val
                    max_patches[i] = patch.clone()
                if val <= min_scores[i]:
                    min_scores[i] = val
                    min_patches[i] = patch.clone()

# Now max_patches and min_patches contain the desired 2x5x5 patches
print("Highest activating patch for output 0:\n", max_patches[0])
print("Lowest activating patch for output 0:\n", min_patches[0])
print("Highest activating patch for output 1:\n", max_patches[1])
print("Lowest activating patch for output 1:\n", min_patches[1])

print('max_scores', max_scores)
print('min_scores', min_scores)

save_sequence(max_patches[0], label=0, save_dir="visualization", tag="highest_activating")
save_sequence(min_patches[0], label=0, save_dir="visualization", tag="lowest_activating")
save_sequence(max_patches[1], label=1, save_dir="visualization", tag="highest_activating")
save_sequence(min_patches[1], label=1, save_dir="visualization", tag="lowest_activating")

print('Visualizations stored in ./visualization')