import torch
from collections import defaultdict
import random
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import math

class MetricTracker:
    """
    Class for tracking metrics for visualization and other processing.
    """
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)

    def sum(self, key):
        return sum(self.data[key]) if self.data[key] else 0.0

    def average(self, key):
        return sum(self.data[key]) / len(self.data[key]) if self.data[key] else 0.0

    def result(self):
        return {k: self.average(k) for k in self.data}

    def reset(self):
        self.data = defaultdict(list)

    def to_dict(self):
        return dict(self.data)

def get_optimizer(args, model):
    """
    Get optimizer for model given name.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        model (torch.nn.Module): Model to optimize.
    """
    match args.optimizer:
        case "SGD":
            return torch.optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        case "Adam":
            return torch.optim.Adam(model.parameters(), lr=args.initial_lr)
        case "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=args.initial_lr)
        case _:
            raise ValueError("Invalid Optimizer!")

def get_scheduler(args, optimizer, len_trainloader):
    """
    Get scheduler for learning rate given name.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        len_trainloader (int): Length of the training data loader.
    """
    match args.scheduler:
        case "Cyclic": # Default
            scheduler_up_iters = max((args.epochs * len_trainloader) // 2, 1)
            scheduler_down_iters = max(args.epochs * len_trainloader - (args.epochs * len_trainloader) // 2, 1)
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.initial_lr, max_lr=args.max_lr,
                                                  step_size_up=scheduler_up_iters, step_size_down=scheduler_down_iters)
        case "CosineAnnealing": # For TinyImageNet
            eta_min = 1e-6 if args.model == "ViT" else 0.001
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len_trainloader, eta_min=eta_min) 
        case "MultiStep": # For Runs With 110 Epochs (100, 105)
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 18], gamma=0.1)
        case "WarmupLambda":
            total_steps = args.epochs * len_trainloader
            warmup_steps = int(0.1 * total_steps)  # 10% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / float(total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        case "None":
            return None
        case _:
            raise ValueError("Invalid Scheduler!")

def get_input_dimensions(dataloader, index_dataset):
    """
    Get the dimensions of the input data.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Data loader to get the dimensions from.
        index_dataset (bool): Whether the dataset is indexed.
    """
    detailer = iter(dataloader)
    data = next(detailer)
    if index_dataset:
        images, _, _ = data
    else:
        images, _ = data

    return images.shape

def aug(dataset_name: str, input_tensor):
    """
    Apply dataset-specific stochastic augmentations while recording transformation metadata.

    For CIFAR10/CIFAR100:
        - Random crop (selects a 32×32 region from up to a 40×40 padded input)
        - Random horizontal flip (p=0.5)

    For MedMNIST subsets (PathMNIST, TissueMNIST, OrganAMNIST, BloodMNIST):
        - Random horizontal flip (p=0.5)
        - Random rotation in [-10°, 10°] using bilinear interpolation

    Args:
        dataset_name (str): Dataset identifier.
        input_tensor (torch.Tensor): Input batch of shape (N, C, H, W).

    Returns:
        Tuple[torch.Tensor, dict]:
            - Augmented images (torch.Tensor of same batch size)
            - Transform metadata dict (keys depend on dataset)
    """
    match dataset_name:
        case dataset_name if dataset_name in ['CIFAR10', 'CIFAR100']:
            batch_size = input_tensor.shape[0]
            x = torch.zeros(batch_size)
            y = torch.zeros(batch_size)
            flip = [False] * batch_size
            rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)
            for i in range(batch_size):
                flip_t = bool(random.getrandbits(1))
                x_t = random.randint(0, 8)
                y_t = random.randint(0, 8)

                rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
                if flip_t:
                    rst[i] = torch.flip(rst[i], [2])
                flip[i] = flip_t
                x[i] = x_t
                y[i] = y_t

            return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}
        case dataset_name if dataset_name in ['PathMNIST', 'TissueMNIST', 'OrganAMNIST', 'BloodMNIST']:
            N = input_tensor.size(0)
            rst = input_tensor.clone()
            flip_flags = []
            rotation_angles = []

            for i in range(N):
                # 1. Flip
                flip_flag = bool(random.getrandbits(1))
                if flip_flag:
                    rst[i] = TF.hflip(rst[i])

                # 2. Pad
                # rst[i] = F.pad(rst[i], pad=(2, 2, 2, 2), mode='reflect')

                # 3. Rotate
                angle = random.uniform(-10, 10)
                rst[i] = TF.rotate(rst[i], angle,
                                   interpolation=TF.InterpolationMode.BILINEAR)

                flip_flags.append(flip_flag)
                rotation_angles.append(angle)

            transform_info = {
                "flipped": flip_flags,
                "rotation": torch.tensor(rotation_angles, dtype=torch.float32)
            }
            return rst, transform_info
        case "TinyImageNet":
            batch_size = input_tensor.shape[0]
            x = torch.zeros(batch_size)
            y = torch.zeros(batch_size)
            flip_flags = [False] * batch_size
            
            rst = torch.zeros((batch_size, 3, 64, 64),
                              dtype=torch.float32,
                              device=input_tensor.device)

            # Note: padded size would be 80×80 if you pre-pad, but here 
            # we assume input_tensor is already padded (like CIFAR aug does)
            for i in range(batch_size):
                flip_flag = bool(random.getrandbits(1))
                x_t = random.randint(0, 16)  # max shift due to padding=8
                y_t = random.randint(0, 16)

                rst[i] = input_tensor[i, :, x_t:x_t + 64, y_t:y_t + 64]
                if flip_flag:
                    rst[i] = torch.flip(rst[i], [2])
                flip_flags[i] = flip_flag
                x[i] = x_t
                y[i] = y_t

            transform_info = {
                "crop": {"x": x, "y": y},
                "flipped": flip_flags
            }
            return rst, transform_info

def aug_trans(dataset_name: str, input_tensor, transform_info):
    """
    Apply a *stored* augmentation transformation to a new tensor.

    Uses metadata produced by `aug()` to apply the exact same transforms
    (crop, flip, rotation) to a fresh batch without randomness.

    This is useful for applying identical spatial transforms to
    adversarial examples or reconstructed inputs.

    Args:
        dataset_name (str): Dataset identifier.
        input_tensor (torch.Tensor): New batch to transform.
        transform_info (dict): Metadata from `aug()`.

    Returns:
        torch.Tensor: Transformed batch.
    """
    match dataset_name:
        case dataset_name if dataset_name in ['CIFAR10', 'CIFAR100']:
            batch_size = input_tensor.shape[0]
            x = transform_info['crop']['x']
            y = transform_info['crop']['y']
            flip = transform_info['flipped']
            rst = torch.zeros((len(input_tensor), 3, 32, 32), dtype=torch.float32, device=input_tensor.device)

            for i in range(batch_size):
                flip_t = int(flip[i])
                x_t = int(x[i])
                y_t = int(y[i])
                rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
                if flip_t:
                    rst[i] = torch.flip(rst[i], [2])
            return rst
        case dataset_name if dataset_name in ['PathMNIST', 'TissueMNIST', 'OrganAMNIST', 'BloodMNIST']:
            N = input_tensor.size(0)
            rst = input_tensor.clone()

            for i in range(N):
                # 1. Flip
                if transform_info["flipped"][i]:
                    rst[i] = TF.hflip(rst[i])

                # 2. Pad
                # rst[i] = F.pad(rst[i], pad=(2, 2, 2, 2), mode='reflect')

                # 3. Rotate
                angle = float(transform_info["rotation"][i])
                rst[i] = TF.rotate(rst[i], angle,
                                   interpolation=TF.InterpolationMode.BILINEAR)
            return rst
        case "TinyImageNet":
            batch_size = input_tensor.shape[0]
            x = transform_info["crop"]["x"]
            y = transform_info["crop"]["y"]
            flip_flags = transform_info["flipped"]

            rst = torch.zeros((batch_size, 3, 64, 64),
                              dtype=torch.float32,
                              device=input_tensor.device)

            for i in range(batch_size):
                x_t = int(x[i])
                y_t = int(y[i])
                rst[i] = input_tensor[i, :, x_t:x_t + 64, y_t:y_t + 64]
                if flip_flags[i]:
                    rst[i] = torch.flip(rst[i], [2])

            return rst

def inverse_aug(dataset_name: str, source_tensor, adv_tensor, transform_info):
    """
    Invert dataset-specific augmentation to map adversarial examples back
    to the source tensor's coordinate frame.

    This is the reverse of `aug_trans()` and is crucial for reconstructing
    perturbed images to the format expected by the model.

    Args:
        dataset_name (str): Dataset identifier.
        source_tensor (torch.Tensor): Base tensor to receive inverse-transformed content.
        adv_tensor (torch.Tensor): Tensor with augmentations applied.
        transform_info (dict): Metadata from `aug()` describing the applied transformation.

    Returns:
        torch.Tensor: Source tensor with inverse transformation applied.
    """
    match dataset_name:
        case dataset_name if dataset_name in ['CIFAR10', 'CIFAR100']:
            x = transform_info['crop']['x']
            y = transform_info['crop']['y']
            flipped = transform_info['flipped']
            batch_size = source_tensor.shape[0]

            for i in range(batch_size):
                flip_t = int(flipped[i])
                x_t = int(x[i])
                y_t = int(y[i])
                if flip_t:
                    adv_tensor[i] = torch.flip(adv_tensor[i], [2])
                source_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32] = adv_tensor[i]

            return source_tensor
        case dataset_name if dataset_name in ['PathMNIST', 'TissueMNIST', 'OrganAMNIST', 'BloodMNIST']:
            N = adv_tensor.size(0)
            for i in range(N):
                # Inverse rotation
                angle = -float(transform_info["rotation"][i])
                img = TF.rotate(adv_tensor[i], angle,
                                interpolation=TF.InterpolationMode.BILINEAR)

                # Inverse crop of padding
                # C, H, W = img.shape
                # img = img[:, 2:H-2, 2:W-2]  # remove pad from step 2

                # Inverse flip
                if transform_info["flipped"][i]:
                    img = TF.hflip(img)

                source_tensor[i] = img
            return source_tensor
        case "TinyImageNet":
            x = transform_info["crop"]["x"]
            y = transform_info["crop"]["y"]
            flipped = transform_info["flipped"]
            batch_size = source_tensor.shape[0]

            for i in range(batch_size):
                if flipped[i]:
                    adv_tensor[i] = torch.flip(adv_tensor[i], [2])
                x_t = int(x[i])
                y_t = int(y[i])
                # Place back into original padded coordinate space
                source_tensor[i, :, x_t:x_t + 64, y_t:y_t + 64] = adv_tensor[i]
            return source_tensor

def calculate_batch_corrects(logits, labels):
    """
    Calculate the number of correctly classified samples by the model on a batch of data.
    
    Args:
        logits (torch.Tensor): Logits of the model.
        labels (torch.Tensor): Labels of the data.
    """
    indices = torch.argmax(logits, 1)
    correct_count = (indices == labels).sum()
    return correct_count
