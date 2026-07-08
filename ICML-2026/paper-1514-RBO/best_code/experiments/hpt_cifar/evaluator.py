"""CIFAR-10 hyperparameter optimization evaluator using the new API."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Union

from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult


class HPTCIFAREvaluator(BaseEvaluator):
    """CIFAR-10 hyperparameter optimization evaluator.
    
    Trains CNNs with given hyperparameters and returns validation accuracy.
    """
    
    def __init__(self, max_epochs: int = 15, device: str = "cuda",
                 data_dir: str = "./data", penalty_value: float = 0.0):
        """Initialize CIFAR-10 evaluator.

        Args:
            max_epochs: Maximum training epochs
            device: Device for training (cuda/cpu)
            data_dir: Directory for CIFAR-10 data
            penalty_value: Value for failed/diverged training
        """
        self.max_epochs = max_epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.penalty_value = penalty_value
        self.data_dir = data_dir

        # Load datasets once
        self._load_datasets()
    
    def _load_datasets(self):
        """Load CIFAR-10 datasets with caching."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Check if already downloaded
        from pathlib import Path
        cifar_path = Path(self.data_dir) / "cifar-10-batches-py"
        download_needed = not cifar_path.exists()
        
        self.train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=download_needed, transform=transform
        )
        
        self.val_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=download_needed, transform=transform
        )
    
    def _create_model(self, architecture: str, num_layers: int) -> nn.Module:
        """Create CNN model based on architecture and layers.
        
        Args:
            architecture: "resnet" or "vgg"
            num_layers: Number of layers (2-5)
            
        Returns:
            PyTorch model
        """
        if architecture == "resnet":
            return self._create_resnet(num_layers)
        elif architecture == "vgg":
            return self._create_vgg(num_layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _create_resnet(self, num_layers: int) -> nn.Module:
        """Create simple ResNet-style model."""
        class SimpleResNet(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                
                # Variable number of layers
                layers = []
                in_channels = 16
                out_channels = 32
                
                for i in range(num_layers):
                    layers.extend([
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ])
                    if i == num_layers // 2:
                        layers.append(nn.MaxPool2d(2))
                        in_channels = out_channels
                        out_channels = 64
                    else:
                        in_channels = out_channels
                
                self.layers = nn.Sequential(*layers)
                self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc = nn.Linear(out_channels * 16, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.layers(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return SimpleResNet(num_layers)
    
    def _create_vgg(self, num_layers: int) -> nn.Module:
        """Create simple VGG-style model."""
        class SimpleVGG(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                layers = []
                in_channels = 3
                
                for i in range(num_layers):
                    out_channels = 32 * (2 ** min(i, 2))  # 32, 64, 128, 128, ...
                    layers.extend([
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.ReLU(inplace=True)
                    ])
                    if i % 2 == 1:  # Add pooling every 2 layers
                        layers.append(nn.MaxPool2d(2))
                    in_channels = out_channels
                
                self.features = nn.Sequential(*layers)
                self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
                self.classifier = nn.Sequential(
                    nn.Linear(out_channels * 4, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return SimpleVGG(num_layers)
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Train CNN and return validation accuracy.
        
        Args:
            params: Either parameter dict or tensor (if tensor, cannot decode without SearchSpace)
            
        Returns:
            EvaluationResult with validation accuracy
        """
        if isinstance(params, torch.Tensor):
            raise ValueError("HPTCIFAREvaluator requires parameter dictionary, not tensor")
        
        start_time = time.time()
        
        try:
            # Extract hyperparameters
            learning_rate = params["learning_rate"]
            optimizer_name = params["optimizer"]
            architecture = params["architecture"]
            batch_size = int(params["batch_size"])
            num_layers = int(params["num_layers"])
            
            # Create data loaders
            train_loader = DataLoader(
                self.train_set, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True
            )
            val_loader = DataLoader(
                self.val_set, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )
            
            # Create model
            model = self._create_model(architecture, num_layers).to(self.device)
            
            # Create optimizer
            if optimizer_name == "sgd":
                optimizer = optim.SGD(
                    model.parameters(), lr=learning_rate, 
                    momentum=0.9, weight_decay=5e-4
                )
            elif optimizer_name == "adam":
                optimizer = optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=5e-4
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Training loop
            model.train()
            for epoch in range(self.max_epochs):
                running_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Check for divergence
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        print(f"Training diverged at epoch {epoch}, batch {batch_idx}")
                        return EvaluationResult.from_true_value(params, self.penalty_value)
                    
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                scheduler.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = correct / total
            
            print(f"Training completed in {time.time() - start_time:.1f}s, accuracy: {accuracy:.4f}")
            
            return EvaluationResult.from_true_value(params, accuracy)
            
        except Exception as e:
            print(f"Error training with params {params}: {e}")
            return EvaluationResult.from_true_value(params, self.penalty_value)
    
