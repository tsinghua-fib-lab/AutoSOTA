"""
OVLR Example: Theory-Aligned Noise Type Comparison.

Compares different theoretically-grounded gradient estimators on
image classification tasks with CrossEntropy loss (reduction='none'):

1. Score-function based estimators (with analytical neg_score):
   - GaussianScoreNoise: neg_score(ε) = ε
   - StudentTScoreNoise: neg_score(ε) = (df+1)ε / (df-2 + ε²)
   - LaplaceScoreNoise: neg_score(ε) = √2 * sign(ε)

2. Two-Point SPSA estimator:
   - RademacherDirectionNoise: Uses finite differences

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from ovlr import (
    ScoreFunctionOVLRGradientEstimator,
    TwoPointSPSAOVLRGradientEstimator,
    GaussianScoreNoise,
    StudentTScoreNoise,
    LaplaceScoreNoise,
    RademacherDirectionNoise,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CURRENT_DIR / 'results_theory_aligned_noise_comparison'
DEFAULT_DATA_DIR = CURRENT_DIR / 'data'
SUPPORTED_NOISE_MODES = [
    'gaussian_score',
    'studentt_score',
    'laplace_score',
    'rademacher_spsa',
]


@dataclass
class ExperimentConfig:
    name: str
    noise_mode: str
    noise_scale: float
    n_repeat: int
    estimator_type: str
    noise_kwargs: Dict[str, object]


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description='Theory-aligned OVLR noise comparison with matched estimators.',
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--train-size', type=int, default=None,
                        help='Optional number of train samples for a quick subset run.')
    parser.add_argument('--test-size', type=int, default=None,
                        help='Optional number of test samples for a quick subset run.')
    parser.add_argument(
        '--noise-modes',
        nargs='+',
        default=SUPPORTED_NOISE_MODES,
        help='Supported modes: gaussian_score, studentt_score, laplace_score, rademacher_spsa.',
    )
    parser.add_argument(
        '--noise-scales',
        nargs='+',
        type=float,
        default=[1.0],
        help='One or more smoothing scales.',
    )
    parser.add_argument(
        '--n-repeats',
        nargs='+',
        type=int,
        default=[200],
        help='Monte Carlo repeat count per batch.',
    )
    parser.add_argument('--studentt-df', type=float, default=5.0)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments that already have results.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'], help='Dataset to use')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive.')
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError('Batch sizes must be positive.')
    if any(scale <= 0.0 for scale in args.noise_scales):
        raise ValueError('All --noise-scales must be positive.')
    if any(repeat <= 0 for repeat in args.n_repeats):
        raise ValueError('All --n-repeats must be positive.')
    unsupported = [mode for mode in args.noise_modes if mode not in SUPPORTED_NOISE_MODES]
    if unsupported:
        raise ValueError(f'Unsupported noise modes: {unsupported}. Supported modes: {SUPPORTED_NOISE_MODES}')
    if args.studentt_df <= 2.0 and 'studentt_score' in args.noise_modes:
        raise ValueError('--studentt-df must be > 2 when using studentt_score.')


def maybe_subset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_data_loaders(args, device: torch.device):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST(
            root=str(args.data_dir),
            train=True,
            transform=transform,
            download=True,
        )
        testset = torchvision.datasets.MNIST(
            root=str(args.data_dir),
            train=False,
            transform=transform,
            download=True,
        )
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(args.data_dir),
            train=True,
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.CIFAR10(
            root=str(args.data_dir),
            train=False,
            transform=transform_test,
            download=True,
        )

    trainset = maybe_subset(trainset, args.train_size, args.seed)
    testset = maybe_subset(testset, args.test_size, args.seed + 1)

    pin_memory = device.type == 'cuda'
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return trainloader, testloader


def get_noise_kwargs(args, noise_mode: str) -> Dict[str, object]:
    if noise_mode == 'studentt_score':
        return {'df': args.studentt_df}
    return {}


def build_experiment_name(
    noise_mode: str,
    noise_scale: float,
    n_repeat: int,
    noise_kwargs: Dict[str, object],
) -> str:
    parts = [noise_mode, f'scale{format(noise_scale, "g")}', f'repeat{n_repeat}']
    if noise_mode == 'studentt_score':
        parts.append(f'df{format(float(noise_kwargs["df"]), "g")}')
    return '_'.join(parts)


def build_experiment_configs(args) -> List[ExperimentConfig]:
    configs = []
    for noise_mode in args.noise_modes:
        for noise_scale in args.noise_scales:
            for n_repeat in args.n_repeats:
                noise_kwargs = get_noise_kwargs(args, noise_mode)
                estimator_type = 'score_function'
                if noise_mode == 'rademacher_spsa':
                    estimator_type = 'two_point_spsa'
                configs.append(
                    ExperimentConfig(
                        name=build_experiment_name(
                            noise_mode=noise_mode,
                            noise_scale=noise_scale,
                            n_repeat=n_repeat,
                            noise_kwargs=noise_kwargs,
                        ),
                        noise_mode=noise_mode,
                        noise_scale=noise_scale,
                        n_repeat=n_repeat,
                        estimator_type=estimator_type,
                        noise_kwargs=noise_kwargs,
                    )
                )
    return configs


def create_estimator(config: ExperimentConfig):
    """Create appropriate estimator based on configuration."""
    if config.noise_mode == 'gaussian_score':
        noise_fn = GaussianScoreNoise(noise_scale=config.noise_scale)
        return ScoreFunctionOVLRGradientEstimator(noise_fn, n_repeat=config.n_repeat)
    if config.noise_mode == 'studentt_score':
        noise_fn = StudentTScoreNoise(
            df=float(config.noise_kwargs['df']),
            noise_scale=config.noise_scale,
        )
        return ScoreFunctionOVLRGradientEstimator(noise_fn, n_repeat=config.n_repeat)
    if config.noise_mode == 'laplace_score':
        noise_fn = LaplaceScoreNoise(noise_scale=config.noise_scale)
        return ScoreFunctionOVLRGradientEstimator(noise_fn, n_repeat=config.n_repeat)
    if config.noise_mode == 'rademacher_spsa':
        direction_noise = RademacherDirectionNoise(noise_scale=config.noise_scale)
        return TwoPointSPSAOVLRGradientEstimator(direction_noise, n_repeat=config.n_repeat)
    raise ValueError(f'Unsupported noise mode: {config.noise_mode}')


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            labels = labels.to(device, non_blocking=device.type == 'cuda')
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    estimator: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == 'cuda')
        labels = labels.to(device, non_blocking=device.type == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = estimator(logits, labels, criterion, loss_fn_reduction='mean')
        loss_total += loss.item()
        optimizer.step()
    return loss_total / len(loader)


def run_experiment(
    config: ExperimentConfig,
    args,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
):
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    estimator = create_estimator(config)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_accuracy = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=trainloader,
            optimizer=optimizer,
            estimator=estimator,
            criterion=criterion,
            device=device,
        )
        accuracy = evaluate(model, testloader, device)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1

        history.append(
            {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_accuracy': accuracy,
            }
        )
        print(
            f'[{config.name}] epoch {epoch + 1:02d}/{args.epochs} '
            f'loss={train_loss:.4f} acc={accuracy:.4f}'
        )

    result = {
        'experiment': asdict(config),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'device': str(device),
        'train_size': args.train_size,
        'test_size': args.test_size,
        'train_time_seconds': time.time() - start_time,
        'final_accuracy': history[-1]['test_accuracy'],
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_loss': history[-1]['train_loss'],
        'max_memory_allocated_mb': (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else None
        ),
        'max_memory_reserved_mb': (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == 'cuda' else None
        ),
        'epoch_wise': history,
    }
    return result


def save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def serialize_args(args):
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def save_summary_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'name',
        'noise_mode',
        'noise_scale',
        'n_repeat',
        'estimator_type',
        'noise_kwargs',
        'final_accuracy',
        'best_accuracy',
        'best_epoch',
        'final_loss',
        'train_time_seconds',
        'max_memory_allocated_mb',
        'max_memory_reserved_mb',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            experiment = result['experiment']
            writer.writerow(
                {
                    'name': experiment['name'],
                    'noise_mode': experiment['noise_mode'],
                    'noise_scale': experiment['noise_scale'],
                    'n_repeat': experiment['n_repeat'],
                    'estimator_type': experiment['estimator_type'],
                    'noise_kwargs': json.dumps(experiment.get('noise_kwargs', {}), sort_keys=True),
                    'final_accuracy': result['final_accuracy'],
                    'best_accuracy': result['best_accuracy'],
                    'best_epoch': result['best_epoch'],
                    'final_loss': result['final_loss'],
                    'train_time_seconds': result['train_time_seconds'],
                    'max_memory_allocated_mb': result['max_memory_allocated_mb'],
                    'max_memory_reserved_mb': result['max_memory_reserved_mb'],
                }
            )


def print_summary(results):
    print('\n=== Theory-Aligned Noise Comparison Summary ===')
    print(f'{"experiment":36s} {"estimator":18s} {"final_acc":>10s} {"best_acc":>10s} {"loss":>10s} {"time(s)":>10s}')
    for result in results:
        experiment = result['experiment']
        print(
            f'{experiment["name"]:36s} '
            f'{experiment["estimator_type"]:18s} '
            f'{result["final_accuracy"]:.4f} '
            f'{result["best_accuracy"]:.4f} '
            f'{result["final_loss"]:.4f} '
            f'{result["train_time_seconds"]:.2f}'
        )


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    trainloader, testloader = build_data_loaders(args, device)
    configs = build_experiment_configs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(serialize_args(args), args.output_dir / 'run_config.json')

    results = []
    for config in configs:
        result_path = args.output_dir / f'{config.name}.json'
        if args.skip_existing and result_path.exists():
            with result_path.open('r', encoding='utf-8') as f:
                result = json.load(f)
            print(f'[skip] {config.name} already exists.')
        else:
            result = run_experiment(
                config=config,
                args=args,
                trainloader=trainloader,
                testloader=testloader,
                device=device,
            )
            save_json(result, result_path)
        results.append(result)

    save_json(results, args.output_dir / 'summary.json')
    save_summary_csv(results, args.output_dir / 'summary.csv')
    print_summary(results)


if __name__ == '__main__':
    main()
