import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.utils import str2bool

from torchvision import models
import torch.nn as nn
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class MNIST:
    def __init__(self, config):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_dataset = datasets.MNIST(
            root="data", train=True, transform=transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root="data", train=False, transform=transform, download=True
        )

        self.config = config
        config.in_channels = 1

    def _one_hot_encode(self, targets):
        n_classes = 10
        return torch.eye(n_classes)[targets]

    def get_dataloaders(self):
        train_data = torch.stack(
            [self.train_dataset[i][0] for i in range(len(self.train_dataset))]
        ).view(-1, 28 * 28)
        train_targets = self._one_hot_encode(
            torch.tensor(
                [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            )
        )

        test_data = torch.stack(
            [self.test_dataset[i][0] for i in range(len(self.test_dataset))]
        ).view(-1, 28 * 28)
        test_targets = self._one_hot_encode(
            torch.tensor(
                [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
            )
        )

        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            ),
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            ),
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return train_loader, test_loader


class BaseDatasetWrapper(Dataset):
    """Base wrapper for converting datasets to tensor format."""

    def __init__(self, device, data, targets, classes, transform=None, flatten=True):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform
        self.device = device
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def _one_hot_encode(self, target):
        target_idx = target.item() if torch.is_tensor(target) else target
        return torch.eye(len(self.classes))[target_idx]

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class MNISTDatasetWrapper(BaseDatasetWrapper):
    """MNIST-specific dataset wrapper."""

    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor (28, 28)
        target = self.targets[idx]

        # Convert to PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform:
            img = self.transform(img)  # Returns tensor (1, 28, 28)

        target = self._one_hot_encode(target)

        if self.flatten:
            img = img.view(28 * 28)

        return img, target


class CIFAR10DatasetWrapper(BaseDatasetWrapper):
    """CIFAR-10-specific dataset wrapper."""

    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor (32, 32, 3) or (3, 32, 32)
        target = self.targets[idx]

        # Convert to PIL Image
        img = img.numpy() if torch.is_tensor(img) else img
        if img.shape[0] in [1, 3]:  # Channel-first to channel-last
            img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)  # Returns tensor (3, 32, 32)

        target = self._one_hot_encode(target)

        if self.flatten:
            img = img.view(3 * 32 * 32)

        return img, target


class TinyImageNetDatasetWrapper(BaseDatasetWrapper):
    """TinyImageNet-specific dataset wrapper."""

    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor (64, 64, 3) or (3, 64, 64)
        target = self.targets[idx]

        # Convert to PIL Image
        img = img.numpy() if torch.is_tensor(img) else img
        if img.shape[0] in [1, 3]:  # Channel-first to channel-last
            img = img.transpose(1, 2, 0)
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)  # Returns tensor (3, 64, 64)

        target = self._one_hot_encode(target)

        if self.flatten:
            img = img.view(3 * 64 * 64)

        return img, target


class BaseContinualDataloader:
    """Base class for continual learning dataloaders."""

    def __init__(self, config, dataset_name="MNIST"):
        self.config = config
        self.dataset_name = dataset_name
        self.num_tasks = config.num_tasks
        self.classes_per_task = config.classes_per_task
        self.batch_size = config.batch_size
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_cnn_encoder = config.use_cnn_encoder if "use_cnn_encoder" in config else False

        # Set flatten based on config and dataset
        if dataset_name == "MNIST":
            self.flatten = (
                True
                if config.get("flatten_imgs") == "default"
                else str2bool(config.get("flatten_imgs"))
            )
            self.in_channels = 1
            self.img_size = 28
            self.dataset_wrapper = MNISTDatasetWrapper
        elif dataset_name == "CIFAR10":
            self.flatten = (
                False
                if config.get("flatten_imgs") == "default"
                else str2bool(config.get("flatten_imgs"))
            )
            self.in_channels = 3
            self.img_size = 32
            self.dataset_wrapper = CIFAR10DatasetWrapper
        elif dataset_name == "TinyImageNet":
            self.flatten = (
                False
                if config.get("flatten_imgs") == "default"
                else str2bool(config.get("flatten_imgs"))
            )
            self.in_channels = 3
            self.img_size = 64
            self.dataset_wrapper = TinyImageNetDatasetWrapper

        # Set input channels for config
        config.in_channels = self.in_channels

        if self.use_cnn_encoder:
            self._setup_encoder()

        self._setup_transforms()
        self._load_datasets()
        self._define_tasks()
        self._precompute_task_indices()

        # Print device info
        print(f"DataLoader using device: {self.device}")

    def _setup_transforms(self):
        """Setup transforms based on dataset."""
        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        elif self.dataset_name == "CIFAR10":
            self.transform = transforms.Compose([transforms.ToTensor()])
        elif self.dataset_name == "TinyImageNet":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def _load_datasets(self):
        """Load the appropriate dataset."""
        if self.dataset_name == "MNIST":
            self.train_dataset = datasets.MNIST(
                root="./data", train=True, transform=self.transform, download=True
            )
            self.test_dataset = datasets.MNIST(
                root="./data", train=False, transform=self.transform, download=True
            )
        elif self.dataset_name == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(
                root="./data", train=True, transform=self.transform, download=True
            )
            self.test_dataset = datasets.CIFAR10(
                root="./data", train=False, transform=self.transform, download=True
            )
        elif self.dataset_name == "TinyImageNet":
            # TinyImageNet can be downloaded automatically
            # Uses ImageFolder structure: ./data/tiny-imagenet-200/train and ./data/tiny-imagenet-200/val
            self.train_dataset = datasets.ImageFolder(
                root="./data/tiny-imagenet-200/train", transform=self.transform
            )
            self.test_dataset = datasets.ImageFolder(
                root="./data/tiny-imagenet-200/val", transform=self.transform
            )

    def _define_tasks(self):
        """Define task splits - to be overridden by subclasses."""
        self.tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def _precompute_task_indices(self):
        self.task_train_indices = {}
        self.task_test_indices = {}
        
        # Convert to tensor once for lightning-fast filtering
        train_targets = torch.as_tensor(self.train_dataset.targets)
        test_targets = torch.as_tensor(self.test_dataset.targets)
        
        for task_id in range(self.num_tasks):
            task_classes = torch.tensor(self.tasks[task_id])
            
            # Vectorized comparison
            self.task_train_indices[task_id] = torch.where(
                torch.isin(train_targets, task_classes)
            )[0].tolist()
            
            self.task_test_indices[task_id] = torch.where(
                torch.isin(test_targets, task_classes)
            )[0].tolist()

    def _get_task_subset(self, is_train, task_id):
        """Get dataset subset for a specific task using pre-computed indices."""
        indices = (
            self.task_train_indices[task_id]
            if is_train
            else self.task_test_indices[task_id]
        )
        dataset = self.train_dataset if is_train else self.test_dataset
        return Subset(dataset, indices)

    def _get_cumulative_test_subset(self, up_to_task_id):
        """Get test subset including all classes up to specified task (for Class IL)."""
        all_indices = []
        for task_id in range(up_to_task_id + 1):
            all_indices.extend(self.task_test_indices[task_id])
        return Subset(self.test_dataset, all_indices)

    def _one_hot_encode(self, targets, num_classes):
        """Convert targets to one-hot encoding."""
        return torch.eye(num_classes)[targets].float()

    def _create_dataloader(self, tensor_dataset, shuffle=True):
        """Create a dataloader with consistent settings."""
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            generator = torch.Generator(device="cpu")

        return DataLoader(
            dataset=tensor_dataset,
            batch_size=self.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            shuffle=shuffle,
        )

    def _process_data(self, dataset, classes, target_mapping_fn):
        """Process and extract data efficiently."""
        if self.use_cnn_encoder:
            # This already handles the DataLoader and GPU acceleration
            embeddings, raw_targets = self._encode_dataset(dataset)
            mapped_targets = torch.tensor([target_mapping_fn(t.item()) for t in raw_targets])
            return embeddings, mapped_targets
        
        # Non-CNN path: Use a temporary DataLoader to fetch all data at once
        dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=self.config.num_workers)
        data, targets = next(iter(dl))
        
        mapped_targets = torch.tensor([target_mapping_fn(t.item()) for t in targets])
        
        if self.flatten:
            data = data.view(data.size(0), -1)
        return data, mapped_targets


    def get_dataloaders(self, task_id):
        """Get dataloaders for a specific task - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_dataloaders")

    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [self.get_dataloaders(task_id) for task_id in range(self.num_tasks)]


    def _encode_dataset(self, subset):

        # Create dataloader to batch the dataset
        loader = DataLoader(
            subset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=False
        )
        
        embeddings = []
        targets = []
        
        # Define these once outside the loop
        # Use InterpolationMode.BICUBIC for better ResNet performance
        resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.encoder.eval()
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                
                if x.shape[1] == 1: # If grayscale, convert to 3 channels by repeating as ResNET expects RGB
                    x = x.expand(-1, 3, -1, -1)
                
                # Perform preprocessing as defined in _setup_encoder for the ResNet encoder
                x = self.encoder_gpu_transform(x)
                
                emb = self.encoder(x)
                
                # Move to CPU to prevent GPU memory overflow during accumulation
                embeddings.append(emb.cpu())
                targets.append(y.cpu())

        # Concatenate all batches into single tensors. 
        # Returns (N, 512) embeddings and (N,) targets where N is total dataset size.
        return torch.cat(embeddings, dim=0), torch.cat(targets, dim=0)

    def _setup_encoder(self):

        if self.config.cnn_encoder == "resnet18":
            encoder = models.resnet18(pretrained=self.config.cnn_pretrained)
            out_dim = encoder.fc.in_features  #Get the input size of the final fully-connected layer 
                                              #(512 for ResNet18). This is the embedding dimension.
            encoder.fc = nn.Identity() # Replace the classification head with an identity function. 
                                        # Now the network outputs the 512-dim feature vector instead of 
                                        # 1000 ImageNet class logits
            self.encoder_out_dim = out_dim

        # move to device and freeze if desired
        encoder = encoder.to(self.device).eval()
        if self.config.encoder_freeze:
            for p in encoder.parameters():
                p.requires_grad = False
        self.encoder = encoder

        # Define preprocessing: 
        self.encoder_gpu_transform = nn.Sequential(
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ).to(self.device)

        self.flatten = True # Flag indicating outputs are already flat vectors (no need to flatten later).
        self.config.in_channels = self.encoder_out_dim # sets the network's expected input dimension to 512  


class DomainILDataloader(BaseContinualDataloader):
    """Domain Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        classes = self.tasks[task_id]

        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        test_dataset = self._get_task_subset(is_train=False, task_id=task_id)

        # Process data with binary remapping (0 vs 1 for each task)
        train_data, train_targets = self._process_data(
            train_dataset,
            classes,
            lambda t: classes.index(t.item() if torch.is_tensor(t) else t),
        )

        test_data, test_targets = self._process_data(
            test_dataset,
            classes,
            lambda t: classes.index(t.item() if torch.is_tensor(t) else t),
        )

        # Create tensor datasets with binary one-hot encoding
        train_tensor_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets, 2),  # Always 2 classes for Domain IL
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, 2)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


class TaskILDataloader(BaseContinualDataloader):
    """Task Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        classes = self.tasks[task_id]

        # Filter datasets
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        test_dataset = self._get_task_subset(is_train=False, task_id=task_id)

        # Process data with binary remapping (same as Domain IL)
        train_data, train_targets = self._process_data(
            train_dataset,
            classes,
            lambda t: classes.index(t.item() if torch.is_tensor(t) else t),
        )

        test_data, test_targets = self._process_data(
            test_dataset,
            classes,
            lambda t: classes.index(t.item() if torch.is_tensor(t) else t),
        )

        # Create tensor datasets with binary one-hot encoding
        train_tensor_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets, 2),  # Always 2 classes per task
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, 2)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


class ClassILDataloader(BaseContinualDataloader):
    """Class Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])

        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        train_tensor_dataset = TensorDataset(
            train_data.float(), self._one_hot_encode(train_targets, num_classes_so_far)
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, num_classes_so_far)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


class ClassIL5TaskDataloader(BaseContinualDataloader):
    """5-task Class Incremental Learning dataloader (2 classes per task)."""

    def __init__(self, config, dataset_name="MNIST"):
        super().__init__(config, dataset_name)
        self.num_tasks = 5
        self.classes_per_task = 2

    def get_dataloaders(self, task_id):
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])

        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        train_tensor_dataset = TensorDataset(
            train_data.float(), self._one_hot_encode(train_targets, num_classes_so_far)
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, num_classes_so_far)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )
    

    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [self.get_dataloaders(task_id) for task_id in range(self.num_tasks)]


class ClassIL2TaskDataloader(BaseContinualDataloader):
    """2-task Class Incremental Learning dataloader (5 classes per task)."""

    def __init__(self, config, dataset_name="MNIST"):
        # Set task configuration before calling super()
        self.num_tasks = 2
        self.classes_per_task = 5
        super().__init__(config, dataset_name)

    def _define_tasks(self):
        """Override to define 2 tasks with 5 classes each."""
        self.tasks = [
            [0, 1, 2, 3, 4],  # Task 0: first 5 digits
            [5, 6, 7, 8, 9],  # Task 1: second 5 digits
        ]

    def get_dataloaders(self, task_id):
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])

        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        train_tensor_dataset = TensorDataset(
            train_data.float(), self._one_hot_encode(train_targets, num_classes_so_far)
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, num_classes_so_far)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )
    
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [self.get_dataloaders(task_id) for task_id in range(self.num_tasks)]


# Factory functions to maintain compatibility with existing code
def DomainILMNIST(config):
    return DomainILDataloader(config, "MNIST")


def TaskILMNIST(config):
    return TaskILDataloader(config, "MNIST")


def ClassILMNIST5Task(config):
    return ClassIL5TaskDataloader(config, "MNIST")


def ClassILMNIST2Task(config):
    return ClassIL2TaskDataloader(config, "MNIST")


### CIFAR10 with encoding enabled
def ClassILCIFAR2Task(config):
    return ClassIL2TaskDataloader(config, "CIFAR10")

def ClassILCIFAR5Task(config):
    return ClassIL5TaskDataloader(config, "CIFAR10")


### OLD CIFAR
def TaskILCIFAR10(config):
    return TaskILDataloader(config, "CIFAR10")


def ClassILCIFAR105Task(config):
    return ClassIL5TaskDataloader(config, "CIFAR10")


def ClassILCIFAR102Task(config):
    return ClassIL2TaskDataloader(config, "CIFAR10")


### TinyImageNet
def DomainILTinyImageNet(config):
    return DomainILDataloader(config, "TinyImageNet")

def TaskILTinyImageNet(config):
    return TaskILDataloader(config, "TinyImageNet")

def ClassILTinyImageNet2Task(config):
    return ClassIL2TaskDataloader(config, "TinyImageNet")

def ClassILTinyImageNet5Task(config):
    return ClassIL5TaskDataloader(config, "TinyImageNet")