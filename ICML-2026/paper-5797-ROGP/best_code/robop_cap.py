"""
Standalone RobOP-CAP implementation for DeiT models on ImageNet-1K.

Robust Optimization Guided Pruning (RobOP) applied to CAP (Correlation Aware Pruner).
Reproduces: DeiT-Tiny, ImageNet-1K, sparsity 0.6, gradient-proportional bounds.

Usage:
    python robop_cap.py --model deit_tiny_patch16_224 --sparsity 0.6 \
        --uncertainty_set trace --gamma 0.005 --num_grads 4096 \
        --fisher_block_size 50 --seed 0
"""
import os
import sys
import time
import math
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Load standalone pruning handle
_pruning_handle_path = os.path.join(os.path.dirname(__file__), 'pruning_handle_standalone.py')
if os.path.exists(_pruning_handle_path):
    exec(open(_pruning_handle_path).read())
else:
    raise FileNotFoundError(f"pruning_handle_standalone.py not found at {_pruning_handle_path}")


def get_model(model_name, device='cuda', weights_path='/models/timm_cache/pytorch_model.bin'):
    """Load a DeiT model from timm with pretrained weights from local cache."""
    import timm
    model = timm.create_model(model_name, pretrained=False)
    if os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(sd)
        print(f"Loaded pretrained weights from {weights_path}")
    else:
        print(f"WARNING: No weights found at {weights_path}")
    model = model.to(device)
    model.eval()
    return model


def load_imagenet_calibration(data_dir, num_samples=4096, seed=0, batch_size=128, num_workers=2):
    """Load ImageNet calibration data with stratified sampling using torchvision."""
    from torchvision.datasets import ImageFolder
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    targets_np = np.array(dataset.targets)

    # Stratified random sampling (as described in paper Section 4.1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_classes = int(targets_np.max()) + 1
    per_class = max(1, num_samples // n_classes)
    indices = []
    for c in range(n_classes):
        c_indices = np.where(targets_np == c)[0]
        replace = len(c_indices) < per_class
        chosen = np.random.choice(c_indices, per_class, replace=replace)
        indices.extend(chosen.tolist())
    indices = indices[:num_samples]

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader


def load_imagenet_val(data_dir, batch_size=128, num_workers=4):
    """Load ImageNet validation set using timm loader."""
    import timm
    from timm.data import create_dataset, create_loader

    val_dataset = create_dataset('imagenet', root=data_dir, split='val', is_training=False)

    loader = create_loader(
        val_dataset,
        input_size=(3, 224, 224),
        batch_size=batch_size,
        is_training=False,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_workers=num_workers,
        crop_pct=0.875,
        pin_memory=True,
    )
    return loader


class EmpiricalBlockFisherInverse:
    """Block-diagonal empirical Fisher inverse, modified for RobOP."""

    def __init__(self, num_grads, fisher_block_size, num_weights, damp, device,
                 uncertainty_set='baseline', gamma=0.005, n_calibration=4096,
                 per_block_gamma=False, hybrid_cv_threshold=0.5):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.dev = device
        self.uncertainty_set = uncertainty_set
        self.gamma = gamma
        self.n_calibration = n_calibration
        self.per_block_gamma = per_block_gamma
        self.hybrid_cv_threshold = hybrid_cv_threshold

        self.num_blocks = math.ceil(self.d / self.B)
        self.damp = damp

        # Track gradient norms for trace computation
        self.grad_norm_sq_sum = None
        self.grad_norm_sq_sq_sum = None

        # Initialize F_inv with baseline damp
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )

    def add_grad(self, g):
        """Updates empirical Fisher inverse with a new gradient."""
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        g_blocks = g.view(self.num_blocks, self.B)

        # Track squared gradient norms per block for trace computation
        if self.uncertainty_set in ('trace', 'eigh', 'hybrid'):
            if self.grad_norm_sq_sum is None:
                self.grad_norm_sq_sum = torch.zeros(self.num_blocks, device=self.dev)
            self.grad_norm_sq_sum += (g_blocks ** 2).sum(dim=1)

        # Track squared squared-norms for hybrid CV computation
        if self.uncertainty_set == 'hybrid':
            if self.grad_norm_sq_sq_sum is None:
                self.grad_norm_sq_sq_sum = torch.zeros(self.num_blocks, device=self.dev)
            g_norms_sq = (g_blocks ** 2).sum(dim=1)
            self.grad_norm_sq_sq_sum += g_norms_sq ** 2

        # batched F_inv x g
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g_blocks)
        alpha = (self.m + torch.einsum("bi,bi->b", g_blocks, Finv_g)).sqrt().unsqueeze(1)
        Finv_g /= alpha
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)

    def apply_robop_regularization(self):
        """Apply RobOP regularization to F_inv based on uncertainty set."""
        if self.uncertainty_set == 'baseline':
            return

        device = self.F_inv.device
        dtype = self.F_inv.dtype

        # Compute per-block gamma scaling based on Fisher trace
        per_block_gamma = self.gamma
        if self.per_block_gamma and self.grad_norm_sq_sum is not None:
            tr_H_per_block = self.grad_norm_sq_sum / self.m
            mean_tr_H = tr_H_per_block.mean()
            if mean_tr_H > 0:
                # Scale gamma by ratio of local trace to mean trace
                # Clamp to [0.1, 5.0] to avoid extreme values
                gamma_multipliers = (tr_H_per_block / mean_tr_H).clamp(0.1, 5.0)
            else:
                gamma_multipliers = torch.ones(self.num_blocks, device=device)

        for block_idx in range(self.num_blocks):
            F_block = self.F_inv[block_idx]  # (B, B)

            if self.uncertainty_set == 'cte':
                # gamma * I (Theorem 3.1)
                reg_value = self.gamma

            elif self.uncertainty_set == 'trace':
                # gamma * Tr(H) / sqrt(N) (Corollary 3.5 / Theorem 3.3)
                if self.grad_norm_sq_sum is not None:
                    tr_H = self.grad_norm_sq_sum[block_idx] / self.m
                    if self.per_block_gamma:
                        reg_value = (per_block_gamma * gamma_multipliers[block_idx].item()) * tr_H / math.sqrt(self.n_calibration)
                    else:
                        reg_value = self.gamma * tr_H / math.sqrt(self.n_calibration)
                else:
                    reg_value = 0.0

            elif self.uncertainty_set == 'eigh':
                if self.grad_norm_sq_sum is not None:
                    tr_H = self.grad_norm_sq_sum[block_idx] / self.m
                    reg_value = self.gamma * math.sqrt(tr_H) * math.sqrt(
                        float(self.grad_norm_sq_sum.sum()) / self.m
                    ) / math.sqrt(self.n_calibration)
                else:
                    reg_value = 0.0
            elif self.uncertainty_set == 'hybrid':
                # Per-block selection: trace for noisy blocks (high CV), cte for stable ones
                if self.grad_norm_sq_sum is not None and self.grad_norm_sq_sq_sum is not None:
                    mean_g2 = self.grad_norm_sq_sum[block_idx] / self.m
                    mean_g4 = self.grad_norm_sq_sq_sum[block_idx] / self.m
                    var_g2 = mean_g4 - mean_g2 ** 2
                    cv = (var_g2.sqrt() / mean_g2).item() if mean_g2 > 0 else 0.0
                    cv_threshold = getattr(self, 'hybrid_cv_threshold', 0.5)
                    if cv > cv_threshold:
                        # Noisy block: use trace-based regularization
                        reg_value = self.gamma * mean_g2 / math.sqrt(self.n_calibration)
                    else:
                        # Stable block: use constant regularization (scaled by trace)
                        reg_value = self.gamma * mean_g2 / math.sqrt(self.n_calibration) * 0.1
                else:
                    reg_value = self.gamma
            else:
                raise ValueError(f"Unknown uncertainty_set: {self.uncertainty_set}")

            if reg_value > 0:
                try:
                    evals, evecs = torch.linalg.eigh(F_block.double())
                    evals = evals.clamp(min=1e-12)
                    new_evals = 1.0 / (1.0 / evals + reg_value)
                    self.F_inv[block_idx] = (
                        evecs @ torch.diag(new_evals) @ evecs.T
                    ).to(dtype)
                except Exception:
                    pass

    def diag(self):
        """Diagonal of the Fisher inverse matrix."""
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[:self.d]


def compute_fisher_inverse(model, dataloader, num_grads, fisher_block_size,
                            damp, uncertainty_set, gamma, n_calibration, device='cuda',
                            layer_adaptive_blocks=False, block_size_scale=1.0,
                            per_block_gamma=False, hybrid_cv_threshold=0.5,
                            skip_regularization=False):
    """Compute block-diagonal empirical Fisher inverse for prunable layers."""
    # CAP recipe: prune attn.qkv, attn.proj, mlp.fc1, mlp.fc2 weights only
    # Matches: 're:.*(attn.(qkv|proj)|mlp.fc\d+).weight'
    import re
    cap_pattern = re.compile(r'.*(attn\.(qkv|proj)|mlp\.fc[12])$')
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if cap_pattern.match(name):
                params_to_prune.append((name, module.weight))

    finv_dict = {}
    for name, param in params_to_prune:
        num_weights = param.numel()

        # Find divisor close to fisher_block_size
        def divisor_generator(n):
            for i in range(1, int(math.sqrt(n) + 1)):
                if n % i == 0:
                    yield i
                    if i * i != n:
                        yield n // i

        divisors = sorted(set(divisor_generator(num_weights)))
        adj_block_size = min(divisors, key=lambda x: abs(x - fisher_block_size))

        # Layer-adaptive block sizing: scale by layer dimensionality
        if layer_adaptive_blocks:
            d_out, d_in = param.shape[0], param.numel() // param.shape[0]
            # Scale proportional to sqrt(num_elements) relative to a baseline
            layer_scale = ((d_out * d_in) ** 0.25) / 20.0
            target_bs = int(adj_block_size * layer_scale * block_size_scale)
            target_bs = max(16, min(512, target_bs))
            adj_block_size = min(divisors, key=lambda x: abs(x - target_bs))

        finv_dict[name] = EmpiricalBlockFisherInverse(
            num_grads, adj_block_size, num_weights, damp, device,
            uncertainty_set=uncertainty_set, gamma=gamma,
            n_calibration=n_calibration,
            per_block_gamma=per_block_gamma,
            hybrid_cv_threshold=hybrid_cv_threshold
        )

    # Collect per-sample gradients (critical for Fisher estimation!)
    criterion = nn.CrossEntropyLoss(reduction='none')
    grad_count = 0

    print(f"Collecting up to {num_grads} per-sample gradients for Fisher computation...")
    for images, labels in dataloader:
        if grad_count >= num_grads:
            break

        images, labels = images.to(device), labels.to(device)

        # Process one sample at a time for per-sample gradients
        for i in range(images.size(0)):
            if grad_count >= num_grads:
                break

            model.zero_grad()
            single_img = images[i:i+1]
            single_label = labels[i:i+1]

            output = model(single_img)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, single_label).sum()
            loss.backward()

            for name, param in params_to_prune:
                if param.grad is not None:
                    finv_dict[name].add_grad(param.grad.view(-1).to(device))

            grad_count += 1

        if grad_count % 256 == 0:
            print(f"  Collected {grad_count}/{num_grads} gradients")

    print(f"Collected {grad_count} gradients total.")

    # Apply RobOP regularization
    if uncertainty_set != 'baseline' and not skip_regularization:
        print(f"Applying RobOP regularization (set={uncertainty_set}, gamma={gamma})...")
        for name, finv in finv_dict.items():
            finv.apply_robop_regularization()

    return finv_dict


def prune_model_cap(model, finv_dict, sparsity, device='cuda'):
    """Prune model using CAP algorithm with pre-computed Fisher inverse."""
    import re
    cap_pattern = re.compile(r'.*(attn\.(qkv|proj)|mlp\.fc[12])$')
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if cap_pattern.match(name) and name in finv_dict:
                params_to_prune.append((name, module))

    print(f"Pruning {len(params_to_prune)} layers to sparsity {sparsity}...")

    for name, module in params_to_prune:
        finv = finv_dict[name]
        weight = module.weight.data.clone()

        handle = CAPHandle(weight, blocks_in_parallel=-1)
        handle.set_Finv(finv.F_inv.clone())
        handle.run()

        pruned_weights = handle.get_pruning_database(np.array([sparsity]))
        pruned_weight = pruned_weights[0]

        if isinstance(module, nn.Conv2d):
            module.weight.data = pruned_weight.reshape(module.weight.shape).to(module.weight.device)
        else:
            module.weight.data = pruned_weight.to(module.weight.device)

        handle.free()

    return model


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """Evaluate model accuracy on ImageNet validation set."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='RobOP-CAP: Robust Pruning for Vision Models')
    parser.add_argument('--model', type=str, default='deit_tiny_patch16_224')
    parser.add_argument('--data_dir', type=str, default='/datasets/imagenet1k')
    parser.add_argument('--sparsity', type=float, default=0.6)
    parser.add_argument('--num_grads', type=int, default=4096)
    parser.add_argument('--fisher_block_size', type=int, default=50)
    parser.add_argument('--damp', type=float, default=1e-8)
    parser.add_argument('--gamma', type=float, default=0.005)
    parser.add_argument('--uncertainty_set', type=str, default='trace',
                        choices=['baseline', 'cte', 'trace', 'eigh', 'hybrid'])
    parser.add_argument('--hybrid_cv_threshold', type=float, default=0.5,
                        help='CV threshold for hybrid uncertainty set (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fisher_seeds', type=int, default=1,
                        help='Number of seeds for Fisher estimation averaging (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_robop_cap')
    parser.add_argument('--layer_adaptive_blocks', action='store_true', default=False,
                        help='Use per-layer block sizes based on layer dimensionality')
    parser.add_argument('--block_size_scale', type=float, default=1.0,
                        help='Scale factor for per-layer block sizes (default: 1.0)')
    parser.add_argument('--per_block_gamma', action='store_true', default=False,
                        help='Use per-block adaptive gamma based on local Fisher trace')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Model: {args.model} | Sparsity: {args.sparsity}")
    print(f"Uncertainty set: {args.uncertainty_set} | Gamma: {args.gamma}")
    print(f"Num grads: {args.num_grads} | Fisher block: {args.fisher_block_size} | Seed: {args.seed}")

    # Load model
    print("\nLoading model...")
    model = get_model(args.model, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load validation data
    print("\nLoading validation data...")
    val_loader = load_imagenet_val(args.data_dir, batch_size=args.val_batch_size, num_workers=args.workers)

    # Evaluate dense model
    print("\nEvaluating dense model...")
    dense_acc = evaluate(model, val_loader, device)
    print(f"Dense model accuracy: {dense_acc:.4f}%")

    # Load calibration data and compute Fisher inverse (multi-seed averaging)
    print("\nComputing Fisher inverse...")
    t0 = time.time()

    if args.fisher_seeds > 1:
        print(f"Multi-seed Fisher estimation with {args.fisher_seeds} seeds...")
        all_finv_dicts = []
        for seed_offset in range(args.fisher_seeds):
            calib_seed = args.seed + seed_offset
            print(f"  Seed {calib_seed} ({seed_offset+1}/{args.fisher_seeds})...")
            calib_loader = load_imagenet_calibration(
                args.data_dir, num_samples=args.num_grads, seed=calib_seed,
                batch_size=args.batch_size, num_workers=args.workers
            )
            finv_dict_seed = compute_fisher_inverse(
                model, calib_loader, args.num_grads, args.fisher_block_size,
                args.damp, args.uncertainty_set, args.gamma, args.num_grads,
                device=device, layer_adaptive_blocks=args.layer_adaptive_blocks,
                block_size_scale=args.block_size_scale,
                per_block_gamma=args.per_block_gamma,
                hybrid_cv_threshold=args.hybrid_cv_threshold,
                skip_regularization=True
            )
            all_finv_dicts.append(finv_dict_seed)

        # Average F_inv matrices across seeds
        print("Averaging Fisher inverses across seeds...")
        finv_dict = all_finv_dicts[0]
        for name in finv_dict:
            avg_finv = sum(d[name].F_inv for d in all_finv_dicts) / float(args.fisher_seeds)
            finv_dict[name].F_inv = avg_finv
            # Average grad_norm stats for regularization
            if all_finv_dicts[0][name].grad_norm_sq_sum is not None:
                avg_gn = sum(d[name].grad_norm_sq_sum for d in all_finv_dicts if d[name].grad_norm_sq_sum is not None) / float(args.fisher_seeds)
                finv_dict[name].grad_norm_sq_sum = avg_gn
            if hasattr(all_finv_dicts[0][name], "grad_norm_sq_sq_sum") and all_finv_dicts[0][name].grad_norm_sq_sq_sum is not None:
                avg_gn2 = sum(d[name].grad_norm_sq_sq_sum for d in all_finv_dicts if d[name].grad_norm_sq_sq_sum is not None) / float(args.fisher_seeds)
                finv_dict[name].grad_norm_sq_sq_sum = avg_gn2

        # Apply regularization once on averaged F_inv
        if args.uncertainty_set != "baseline":
            print(f"Applying RobOP regularization (set={args.uncertainty_set}, gamma={args.gamma})...")
            for name, finv in finv_dict.items():
                finv.apply_robop_regularization()
    else:
        calib_loader = load_imagenet_calibration(
            args.data_dir, num_samples=args.num_grads, seed=args.seed,
            batch_size=args.batch_size, num_workers=args.workers
        )
        finv_dict = compute_fisher_inverse(
            model, calib_loader, args.num_grads, args.fisher_block_size,
            args.damp, args.uncertainty_set, args.gamma, args.num_grads,
            device=device, layer_adaptive_blocks=args.layer_adaptive_blocks,
            block_size_scale=args.block_size_scale,
            per_block_gamma=args.per_block_gamma,
            hybrid_cv_threshold=args.hybrid_cv_threshold
        )

    fisher_time = time.time() - t0
    print(f"Fisher computation time: {fisher_time:.2f}s")
    # Prune model
    print("\nPruning model with CAP...")
    t0 = time.time()
    model = prune_model_cap(model, finv_dict, args.sparsity, device=device)
    prune_time = time.time() - t0
    print(f"Pruning time: {prune_time:.2f}s")

    # Compute actual sparsity
    total_nnz = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        total_nnz += (p.data.abs() > 1e-8).sum().item()
    actual_sparsity = 1.0 - total_nnz / total_params
    print(f"Actual sparsity: {actual_sparsity:.4f} (target: {args.sparsity})")

    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_acc = evaluate(model, val_loader, device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Dense accuracy:  {dense_acc:.4f}%")
    print(f"Pruned accuracy: {pruned_acc:.4f}%")
    print(f"Accuracy drop:   {dense_acc - pruned_acc:.4f}%")
    print(f"Sparsity:        {actual_sparsity:.4f}")
    print(f"{'='*60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': args.model,
        'sparsity': args.sparsity,
        'actual_sparsity': actual_sparsity,
        'uncertainty_set': args.uncertainty_set,
        'gamma': args.gamma,
        'dense_accuracy': dense_acc,
        'pruned_accuracy': pruned_acc,
        'accuracy_drop': dense_acc - pruned_acc,
        'seed': args.seed,
        'fisher_time_s': fisher_time,
        'prune_time_s': prune_time,
    }
    result_file = os.path.join(
        args.output_dir,
        f"robop_cap_{args.model}_sp{args.sparsity}_{args.uncertainty_set}_g{args.gamma}_b{args.fisher_block_size}_s{args.seed}.json"
    )
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return pruned_acc


if __name__ == '__main__':
    main()
