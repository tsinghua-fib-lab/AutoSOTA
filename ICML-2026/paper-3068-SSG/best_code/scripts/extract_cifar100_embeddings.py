import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='/datasets/cifar100_embeddings.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-dir', type=str, default='./cifar100')
    parser.add_argument('--weights', type=str, default='/models/torchvision/resnet18-f37072fd.pth')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load ResNet-18 architecture and load local weights
    print('Loading ResNet-18 with local weights...')
    model = torchvision.models.resnet18(weights=None)
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Remove the final fc layer to get 512-dim features
    model.fc = torch.nn.Identity()

    # CIFAR-100 transform: resize to 224x224 for ImageNet-pretrained model
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-100 training set
    print('Loading CIFAR-100 training set...')
    trainset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    all_indices = []
    all_labels = []
    all_embeddings = []

    print(f'Extracting embeddings for {len(trainset)} samples...')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device)
            features = model(inputs)  # [B, 512]
            features = features.cpu()

            start_idx = batch_idx * args.batch_size
            indices = list(range(start_idx, start_idx + inputs.size(0)))

            all_indices.extend(indices)
            all_labels.extend(targets.tolist())
            all_embeddings.append(features)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'indices': torch.tensor(all_indices),
        'labels': torch.tensor(all_labels),
        'embeddings': all_embeddings,
    }
    torch.save(data, output_path)
    print(f'Saved embeddings to {output_path}')
    print(f'Shape: indices={data["indices"].shape}, labels={data["labels"].shape}, embeddings={data["embeddings"].shape}')

if __name__ == '__main__':
    main()
