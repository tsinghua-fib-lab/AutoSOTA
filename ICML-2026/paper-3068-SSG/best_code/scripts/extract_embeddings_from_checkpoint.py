import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0, '/repo/training/cifar')
from model import ResNet18

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='/datasets/cifar100_embeddings_v2.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-dir', type=str, default='/repo/cifar100')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load CIFAR-100-trained ResNet-18
    print(f'Loading checkpoint from {args.checkpoint}...')
    model = ResNet18(num_classes=100)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapping
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Use model's create_emb to get 512-dim features
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    print('Loading CIFAR-100 training set...')
    trainset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=False, transform=transform
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
            features = model.create_emb(inputs)  # [B, 512]
            features = features.cpu()

            start_idx = batch_idx * args.batch_size
            indices = list(range(start_idx, start_idx + inputs.size(0)))

            all_indices.extend(indices)
            all_labels.extend(targets.tolist())
            all_embeddings.append(features)

    all_embeddings = torch.cat(all_embeddings, dim=0)

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
