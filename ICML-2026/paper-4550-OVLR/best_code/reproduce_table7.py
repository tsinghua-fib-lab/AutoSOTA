#!/usr/bin/env python3
"""
Reproduction script for OVLR Table 7: ResNet-18 + CIFAR-10 + hard 0-1 loss.
Matches paper setting: 5-epoch CE warmup, then OVLR with hard 0-1 loss.
Saves final metrics to JSON for manifest generation.
"""
import argparse, time, json, os
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def hard_01_loss(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions != targets).float()


def test(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--transition-epochs', type=int, default=0,
                        help='Number of epochs for smooth CE->OVLR transition after warmup')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-repeat', type=int, default=200)
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--noise-scale-end', type=float, default=None,
                        help='Target noise scale for linear annealing over OVLR epochs')
    parser.add_argument('--data-dir', type=str, default='/datasets')
    parser.add_argument('--output', type=str, default='/repo/results_table7.json')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cosine-annealing', action='store_true',
                        help='Use cosine annealing LR from lr->1e-5 over OVLR epochs')
    args = parser.parse_args()
    transition_epochs = args.transition_epochs

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

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

    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()

    # Cosine annealing LR scheduler for post-warmup epochs
    if args.cosine_annealing:
        ovlr_epochs = args.epochs - args.warmup_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=ovlr_epochs, eta_min=1e-5)
    else:
        scheduler = None

    from ovlr import OVLRGradientEstimator, get_noise_fn
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    print(f"\n{'='*70}")
    print(f"OVLR Table 7: ResNet-18 + CIFAR-10 + hard 0-1 loss")
    print(f"Warmup: {args.warmup_epochs} epochs CE, Total: {args.epochs} epochs")
    if transition_epochs > 0:
        print(f"Transition: {transition_epochs} epochs linear CE->OVLR interpolation")
    print(f"n_repeat={args.n_repeat}, sigma={args.noise_scale}, lr={args.lr}, batch={args.batch_size}")
    print(f"{'='*70}\n")

    best_acc = 0.0
    best_ovlr_acc = 0.0
    warmup_time = 0.0
    ovlr_time = 0.0
    history = []

    current_sigma = args.noise_scale

    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        epoch_loss = correct = total = 0.0

        if epoch <= args.warmup_epochs:
            # ---- CE warmup phase ----
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = ce_criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
            phase = "WARMUP_CE"

        else:
            # ---- Post-warmup: OVLR or transition ----
            # Noise scale annealing
            if args.noise_scale_end is not None:
                ovlr_epochs_total = args.epochs - args.warmup_epochs
                progress = (epoch - args.warmup_epochs - 1) / max(ovlr_epochs_total - 1, 1)
                current_sigma = args.noise_scale + progress * (args.noise_scale_end - args.noise_scale)
                estimator.noise_fn.noise_scale = current_sigma
            else:
                current_sigma = args.noise_scale

            in_transition = (transition_epochs > 0 and epoch <= args.warmup_epochs + transition_epochs)
            if in_transition:
                alpha = (epoch - args.warmup_epochs) / transition_epochs
                phase = "TRANS_a{:.2f}".format(alpha)
            else:
                alpha = 1.0
                phase = "OVLR_01"

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if in_transition:
                    # Combined CE + OVLR gradient with linear interpolation
                    batch_size = outputs.size(0)

                    # OVLR gradient component
                    outputs_rep, noisy_outputs, epsilon = estimator.forward_noisy_outputs(outputs)
                    labels_rep = targets.repeat(estimator.n_repeat, *([1] * (targets.dim() - 1)))
                    ovlr_loss_vals = hard_01_loss(noisy_outputs, labels_rep)
                    while ovlr_loss_vals.dim() < epsilon.dim():
                        ovlr_loss_vals = ovlr_loss_vals.unsqueeze(-1)
                    vec_ovlr = alpha * ovlr_loss_vals * epsilon / (batch_size * estimator.noise_fn.noise_scale)

                    # CE gradient component
                    ce_loss_scaled = (1.0 - alpha) * ce_criterion(outputs, targets)

                    # Combined backward: OVLR gradient via vec (on repeated outputs), CE gradient via autograd
                    outputs_rep.backward(vec_ovlr, retain_graph=True)
                    ce_loss_scaled.backward()

                    # Logging: reconstruct equivalent scalar loss
                    if alpha < 1.0:
                        ce_loss_raw = ce_loss_scaled.item() / (1.0 - alpha)
                    else:
                        ce_loss_raw = 0.0
                    loss_val = alpha * ovlr_loss_vals.mean().item() + (1.0 - alpha) * ce_loss_raw
                    epoch_loss += loss_val
                else:
                    loss = estimator(outputs, targets, hard_01_loss, loss_fn_reduction='mean')
                    epoch_loss += loss.item()

                optimizer.step()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)

        torch.cuda.synchronize()
        et = time.perf_counter() - t0
        if epoch <= args.warmup_epochs:
            warmup_time += et
        else:
            ovlr_time += et
            if scheduler is not None:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = test(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        if epoch > args.warmup_epochs:
            best_ovlr_acc = max(best_ovlr_acc, test_acc)

        history.append({"epoch": epoch, "phase": phase, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 2), "test_acc": round(test_acc, 2),
                        "time_s": round(et, 2)})
        print("Epoch {:3d}/{} [{:15s}] Loss: {:.4f} | Train: {:.2f}% | Test: {:.2f}% | LR: {:.2e} | sig: {:.2f} | {:.1f}s".format(
              epoch, args.epochs, phase, train_loss, train_acc, test_acc, current_lr, current_sigma, et))

    total_time = warmup_time + ovlr_time
    print("\n" + "="*70)
    print("RESULTS")
    print("  Best test accuracy (overall): {:.2f}%".format(best_acc))
    print("  Best OVLR-phase accuracy: {:.2f}%".format(best_ovlr_acc))
    print("  Final test accuracy: {:.2f}%".format(history[-1]['test_acc']))
    print("  Warmup time: {:.1f}s".format(warmup_time))
    print("  OVLR time: {:.1f}s".format(ovlr_time))
    print("  Total time: {:.1f}s".format(total_time))
    print("  Paper target: 62.04% accuracy, 225.1s time")
    print("="*70)

    result = {
        "paper_id": 4550,
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "loss": "hard_0-1",
        "method": "OVLR",
        "warmup_epochs": args.warmup_epochs,
        "warmup_loss": "Cross-Entropy",
        "total_epochs": args.epochs,
        "n_repeat": args.n_repeat,
        "noise_scale": args.noise_scale,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "transition_epochs": transition_epochs,
        "best_test_accuracy": round(best_acc, 2),
        "best_ovlr_phase_accuracy": round(best_ovlr_acc, 2),
        "final_test_accuracy": round(history[-1]['test_acc'], 2),
        "warmup_time_s": round(warmup_time, 1),
        "ovlr_time_s": round(ovlr_time, 1),
        "total_time_s": round(total_time, 1),
        "history": history,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to {}".format(args.output))


if __name__ == '__main__':
    main()
