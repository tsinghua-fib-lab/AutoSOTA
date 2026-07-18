"""
train_model.py - Train ResNet-18 on CIFAR-10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patch_cifar10  # noqa: E402 - monkey-patch torchvision CIFAR-10 MD5 check

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import argparse
import os
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_ROOT = './data'
OUTPUT_DIR = '.'
BEST_CHECKPOINT = 'resnet18_cifar10_best.pth'
LATEST_CHECKPOINT = 'resnet18_cifar10_latest.pth'
SEED = 42

def set_random_seeds(seed=42):
    """Set random seeds for reproducible CIFAR-10 training."""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_resnet18_cifar10():
    """Create ResNet-18 model adapted for CIFAR-10"""
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

def train_model():
    print("Starting ResNet-18 training on CIFAR-10...")
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Create model
    model = create_resnet18_cifar10().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
    
    # Training parameters
    num_epochs = 200
    best_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(testloader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{test_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        acc = 100.*correct/total
        print(f'Epoch {epoch+1}: Test Accuracy = {acc:.2f}%')
        
        # Save checkpoint
        if acc > best_acc:
            print(f'Saving best model (accuracy: {acc:.2f}%)...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(OUTPUT_DIR, BEST_CHECKPOINT))
            best_acc = acc
        
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, LATEST_CHECKPOINT))
        scheduler.step()
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--best-checkpoint", default=BEST_CHECKPOINT)
    parser.add_argument("--latest-checkpoint", default=LATEST_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Retrain without prompting when the best checkpoint exists.")
    return parser.parse_args()

def configure(args):
    global DATA_ROOT, OUTPUT_DIR, BEST_CHECKPOINT, LATEST_CHECKPOINT, SEED, device
    DATA_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir
    BEST_CHECKPOINT = args.best_checkpoint
    LATEST_CHECKPOINT = args.latest_checkpoint
    SEED = args.seed
    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    set_random_seeds(SEED)

def main(args=None):
    if args is None:
        args = parse_args()
    configure(args)

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_checkpoint_path = os.path.join(OUTPUT_DIR, BEST_CHECKPOINT)
    if os.path.exists(best_checkpoint_path) and not args.force:
        response = input("Trained model already exists. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without training.")
            exit()
    
    train_model()

if __name__ == "__main__":
    main()
