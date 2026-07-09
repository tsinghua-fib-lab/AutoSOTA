"""
run.py — Grad-Mimic training entry point.

Usage
-----
python run.py --mode grad-mimic --dataset_name cifar10 [options]

Run `python run.py --help` for the full argument list.
"""

import os
import pickle
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from gradmimic.datasets import IndexedDataset, FewShotDataset, RankedDataset
from gradmimic.models import load_init_model
from gradmimic.training import (
    get_per_sample_gradients,
    compute_task_vector,
    compute_similarity,
    gradient_calibration,
    solve_subset_selection,
)
from gradmimic.utils import evaluate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Grad-Mimic: gradient mimicry for robust learning under label noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["grad-mimic", "influence-function", "grad-match", "grad-descent", "grad-norm", "agra", "rho"],
        help="Training algorithm.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="normed_proj",
        choices=["opt", "cos", "proj", "normed_cos", "normed_proj"],
        help="Gradient reweighting method (ignored for grad-norm, agra, rho, grad-descent).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        choices=["dtd", "pet", "stl10", "cifar10", "cifar100", "flower102"],
        help="Dataset to use for training and evaluation.",
    )
    parser.add_argument("--noisy_level", type=float, default=0.0, help="Fraction of training labels to corrupt.")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise std added to the reference model weights.")
    parser.add_argument("--few_shot", type=int, default=10, help="Number of clean samples per class (influence-function mode only).")
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="random",
        choices=["mimic", "random"],
        help="How to select the top-p subset when --top_p < 1.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Fraction of training data to keep (1.0 = full dataset).",
    )
    parser.add_argument("--num_epoch", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--model_arch",
        type=str,
        default="vit-b",
        choices=["vit-b", "vit-l"],
        help="Vision Transformer backbone.",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--linear_probing",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fine-tune the classification head only (linear probing).",
    )
    parser.add_argument(
        "--using_testset_to_build_ref",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Train the reference model on the test split instead of training split.",
    )
    parser.add_argument(
        "--mimic_layer",
        type=str,
        default="heads.head.weight",
        help="Parameter name of the layer used for gradient matching.",
    )
    parser.add_argument(
        "--calibrate_mimic_layer_only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply reweighted gradients only to --mimic_layer; use mean gradient elsewhere.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for normed similarity methods.")
    parser.add_argument("--starting_epoch", type=int, default=0, help="First epoch to activate the chosen algorithm (earlier epochs use grad-descent).")
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=32,
        choices=[16, 32, 64, 128, 256],
        help="Mini-batch size.",
    )
    parser.add_argument("--dataset_dir", type=str, default="./data", help="Root directory containing all datasets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Root directory for saved models and training logs.",
    )
    parser.add_argument("--ref_model", type=str, default=None, help="Explicit path to reference model .pt file (overrides automatic lookup).")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimiser.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "cosine", "step"], help="Learning rate scheduler.")
    parser.add_argument("--norm_way", type=str, default="l2", choices=["l2", "l1"], help="Regularisation norm for opt-based methods.")
    parser.add_argument("--lambda_value", type=float, default=0.0, help="Regularisation coefficient for opt-based methods.")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=123, help="Global random seed.")
    parser.add_argument("--target_seed", type=int, default=123, help="Seed used when training the reference model (for automatic lookup).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _sub_folder(pretrained, linear_probing):
    if pretrained and linear_probing:
        return "pretrained_linear_probing"
    elif pretrained:
        return "pretrained_fine_tune_all"
    return "train_from_scratch"


def _log_stem(args, sub_folder):
    layer_tag = "layer_only" if args.calibrate_mimic_layer_only else "all_layers"
    return (
        f"{args.model_arch}_{args.mimic_layer}_{layer_tag}"
        f"_{args.dataset_name}_{args.method}"
        f"_temp{args.temperature}"
        f"_ep{args.num_epoch}"
        f"_{args.mode}"
        f"_noisy{args.noisy_level}"
        f"_sigma{args.sigma}"
        f"_fs{args.few_shot}"
        f"_seed{args.seed}"
        f"_tgtseed{args.target_seed}"
    )


def _descent_log_stem(args):
    return (
        f"{args.model_arch}_{args.dataset_name}"
        f"_ep{args.num_epoch}_grad-descent"
        f"_noisy{args.noisy_level}"
        f"_topp{args.top_p}"
        f"_sampling_{args.sampling_method}"
        f"_seed{args.seed}"
    )


def _ref_model_stem(args, seed):
    return (
        f"{args.model_arch}_{args.dataset_name}"
        f"_ep{args.num_epoch}_grad-descent"
        f"_noisy0.0_seed{seed}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # --- Auto-adjust epochs for smaller datasets ---
    if args.dataset_name in ("dtd", "flower102") and args.num_epoch < 10:
        print(f"Adjusting num_epoch from {args.num_epoch} to 10 for {args.dataset_name}.")
        args.num_epoch = 10

    print("=" * 60)
    print("Grad-Mimic — Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # --- Reproducibility ---
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Output directories ---
    sub_folder = _sub_folder(args.pretrained, args.linear_probing)
    log_dir = os.path.join(args.output_dir, "logs", sub_folder)
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, sub_folder), exist_ok=True)

    # --- Transforms ---
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # --- Datasets ---
    if args.top_p < 1.0:
        ranking_file = None
        if args.sampling_method == "mimic":
            # Look for a weights file from a prior grad-mimic run
            stem = _log_stem(args, sub_folder)
            ranking_file = os.path.join(log_dir, f"{stem}_weights.pkl")
        train_dataset = RankedDataset(
            root=args.dataset_dir, name=args.dataset_name, train=True,
            download=False, transform=transform_train,
            sampling_method=args.sampling_method, top_p=args.top_p,
            noise_ratio=0.0, ranking_file=ranking_file,
        )
    else:
        use_test_split = args.using_testset_to_build_ref
        train_dataset = IndexedDataset(
            root=args.dataset_dir, name=args.dataset_name,
            train=(not use_test_split), download=False,
            transform=transform_train, noise_ratio=args.noisy_level,
        )

    test_dataset = IndexedDataset(
        root=args.dataset_dir, name=args.dataset_name, train=False,
        download=False, transform=transform_test, noise_ratio=0.0,
    )
    num_class = train_dataset.num_class

    train_loader = DataLoader(train_dataset, batch_size=args.training_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    if args.mode == "influence-function":
        val_dataset = FewShotDataset(
            root=args.dataset_dir, name=args.dataset_name, train=False,
            download=False, transform=transform_test, few_shot_k=args.few_shot,
        )
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=4)

    # --- Initial model ---
    model = load_init_model(
        arch_name=args.model_arch, num_class=num_class, device=device,
        seed=args.seed, pretrained=args.pretrained,
        linear_probing=args.linear_probing, model_dir=model_dir,
    )
    model = model.to(device)

    # --- Baseline evaluation ---
    init_true_loss, init_true_acc = evaluate(model, train_loader, device, use_noisy_labels=False)
    init_noisy_loss, init_noisy_acc = evaluate(model, train_loader, device, use_noisy_labels=True)
    init_test_loss, init_test_acc = evaluate(model, test_loader, device)
    print(f"[Init] Train(true)  loss={init_true_loss:.4f}  acc={init_true_acc:.4f}")
    print(f"[Init] Train(noisy) loss={init_noisy_loss:.4f}  acc={init_noisy_acc:.4f}")
    print(f"[Init] Test         loss={init_test_loss:.4f}  acc={init_test_acc:.4f}")

    # --- Learning-curve bookkeeping ---
    learning_results = {
        "true_training_loss":      [init_true_loss]  + [0.0] * args.num_epoch,
        "noisy_training_loss":     [init_noisy_loss] + [0.0] * args.num_epoch,
        "testing_loss":            [init_test_loss]  + [0.0] * args.num_epoch,
        "true_training_accuracy":  [init_true_acc]   + [0.0] * args.num_epoch,
        "noisy_training_accuracy": [init_noisy_acc]  + [0.0] * args.num_epoch,
        "testing_accuracy":        [init_test_acc]   + [0.0] * args.num_epoch,
    }

    result_collection = {
        i: {
            "status": "incorrect" if i in train_dataset.noisy_indices else "correct",
            "per_sample_weights": [0.0] * args.num_epoch,
        }
        for i in range(len(train_dataset))
    }

    # --- Reference model (grad-mimic / rho) ---
    reference_model = None
    if args.mode in ("grad-mimic", "rho"):
        if args.ref_model:
            ref_path = args.ref_model
        else:
            stem = _ref_model_stem(args, args.target_seed)
            ref_path = os.path.join(model_dir, sub_folder, f"{stem}.pt")

        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Reference model not found at:\n  {ref_path}\n\n"
                "Train one first with:\n"
                f"  python run.py --mode grad-descent "
                f"--dataset_name {args.dataset_name} "
                f"--model_arch {args.model_arch} "
                f"--num_epoch {args.num_epoch} "
                f"--noisy_level 0.0 "
                f"--seed {args.target_seed}\n\n"
                "Then re-run with --mode grad-mimic using the same --target_seed."
            )
        else:
            print(f"Loading reference model: {ref_path}")
            reference_model = torch.load(ref_path, map_location=device, weights_only=False)
            if args.sigma > 0.0:
                with torch.no_grad():
                    layer = dict(reference_model.named_parameters())[args.mimic_layer]
                    layer.add_(torch.normal(0, args.sigma, size=layer.shape, device=device))
            reference_model.eval()
            ref_test_loss, ref_test_acc = evaluate(reference_model, test_loader, device)
            print(f"[Ref]  Test loss={ref_test_loss:.4f}  acc={ref_test_acc:.4f}")

    # --- Optimiser ---
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)

    # --- Learning rate scheduler ---
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.num_epoch // 2), gamma=0.5)
    else:
        scheduler = None

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in tqdm(range(args.num_epoch), desc="Epochs"):
        model.train()
        running_noisy_loss = running_true_loss = 0.0
        noisy_correct = true_correct = total = 0

        for indices, inputs, true_labels, noisy_labels in train_loader:
            noisy_labels = noisy_labels.long()
            true_labels = true_labels.long()
            optimizer.zero_grad()
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)
            noisy_labels = noisy_labels.to(device)

            active = epoch >= args.starting_epoch

            # ---- Grad-Norm ------------------------------------------------
            if args.mode == "grad-norm" and active:
                ft_grads = get_per_sample_gradients(model, inputs, noisy_labels)
                layer_grads = ft_grads[args.mimic_layer]
                norms = torch.stack([g.norm() for g in layer_grads])
                per_w = torch.nn.Softmax(dim=0)(norms)
                for loc, idx in enumerate(indices):
                    result_collection[idx.item()]["per_sample_weights"][epoch] = per_w[loc].item()
                gradient_calibration(model, per_w, ft_grads, args.mimic_layer, args.calibrate_mimic_layer_only)
                optimizer.step()

            # ---- AGRA -----------------------------------------------------
            elif args.mode == "agra" and active:
                comp_indices, comp_inputs, _, comp_noisy = next(iter(train_loader))
                comp_inputs = comp_inputs.to(device)
                comp_noisy = comp_noisy.long().to(device)
                comp_grads = get_per_sample_gradients(model, comp_inputs, comp_noisy)
                ft_grads = get_per_sample_gradients(model, inputs, noisy_labels)
                comp_mean = comp_grads[args.mimic_layer].mean(dim=0)
                layer_grads = ft_grads[args.mimic_layer]
                per_w = compute_similarity(comp_mean, layer_grads, method="cos")
                per_w = torch.clamp(per_w, min=0)
                nz = (per_w > 0).sum().item()
                final_w = torch.zeros_like(per_w)
                if nz > 0:
                    final_w[per_w > 0] = 1.0 / nz
                else:
                    final_w.fill_(1.0 / len(inputs))
                gradient_calibration(model, final_w, ft_grads, args.mimic_layer, args.calibrate_mimic_layer_only)
                optimizer.step()

            # ---- Grad-Match -----------------------------------------------
            elif args.mode == "grad-match" and active:
                ft_grads = get_per_sample_gradients(model, inputs, noisy_labels)
                mean_g = ft_grads[args.mimic_layer].mean(dim=0).cpu().detach().numpy()
                layer_g = ft_grads[args.mimic_layer].cpu().detach().numpy()
                per_w = solve_subset_selection(mean_g, layer_g, inputs.size(0), args.lambda_value, args.norm_way, device)
                for loc, idx in enumerate(indices):
                    result_collection[idx.item()]["per_sample_weights"][epoch] = per_w[loc].item()
                gradient_calibration(model, per_w, ft_grads, args.mimic_layer, args.calibrate_mimic_layer_only)
                optimizer.step()

            # ---- Rho ------------------------------------------------------
            elif args.mode == "rho" and active:
                our_out = model(inputs)
                ref_out = reference_model(inputs)
                our_loss = torch.nn.CrossEntropyLoss(reduction="none")(our_out, noisy_labels)
                ref_loss = torch.nn.CrossEntropyLoss(reduction="none")(ref_out, noisy_labels)
                per_w = torch.nn.Softmax(dim=0)(our_loss - ref_loss)
                (our_loss @ per_w).backward()
                for loc, idx in enumerate(indices):
                    result_collection[idx.item()]["per_sample_weights"][epoch] = per_w[loc].item()
                optimizer.step()

            # ---- Influence Function ---------------------------------------
            elif args.mode == "influence-function" and active:
                golden_inputs, golden_labels = next(iter(val_loader))
                golden_inputs = golden_inputs.to(device)
                golden_labels = golden_labels.long().to(device)
                gold_grads = get_per_sample_gradients(model, golden_inputs, golden_labels)
                ft_grads = get_per_sample_gradients(model, inputs, noisy_labels)
                golden_mean = gold_grads[args.mimic_layer].mean(dim=0)
                per_w = compute_similarity(golden_mean, ft_grads[args.mimic_layer], args.method, args.temperature)
                for loc, idx in enumerate(indices):
                    result_collection[idx.item()]["per_sample_weights"][epoch] = per_w[loc].item()
                gradient_calibration(model, per_w, ft_grads, args.mimic_layer, args.calibrate_mimic_layer_only)
                optimizer.step()

            # ---- Grad-Mimic -----------------------------------------------
            elif args.mode == "grad-mimic" and active:
                ft_grads = get_per_sample_gradients(model, inputs, noisy_labels)
                task_vec = compute_task_vector(model, reference_model, args.mimic_layer)
                neg_grads = -ft_grads[args.mimic_layer]
                if args.method != "opt":
                    per_w = compute_similarity(task_vec, neg_grads, args.method, args.temperature)
                else:
                    tv_np = task_vec.cpu().detach().numpy()
                    ng_np = neg_grads.cpu().detach().numpy()
                    per_w = solve_subset_selection(tv_np, ng_np, inputs.size(0), args.lambda_value, args.norm_way, device)
                for loc, idx in enumerate(indices):
                    result_collection[idx.item()]["per_sample_weights"][epoch] = per_w[loc].item()
                gradient_calibration(model, per_w, ft_grads, args.mimic_layer, args.calibrate_mimic_layer_only)
                optimizer.step()

            # ---- Grad-Descent (baseline) ----------------------------------
            else:
                out = model(inputs)
                loss = torch.nn.CrossEntropyLoss()(out, noisy_labels)
                loss.backward()
                optimizer.step()

            # --- Per-batch metrics -----------------------------------------
            with torch.no_grad():
                out = model(inputs)
                running_noisy_loss += torch.nn.CrossEntropyLoss()(out, noisy_labels).item() * true_labels.size(0)
                running_true_loss  += torch.nn.CrossEntropyLoss()(out, true_labels).item()  * true_labels.size(0)
                _, predicted = out.max(1)
                total        += true_labels.size(0)
                noisy_correct += predicted.eq(noisy_labels).sum().item()
                true_correct  += predicted.eq(true_labels).sum().item()

        noisy_train_loss = running_noisy_loss / total
        true_train_loss  = running_true_loss  / total
        noisy_train_acc  = noisy_correct / total
        true_train_acc   = true_correct  / total

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.num_epoch} | "
            f"Noisy train loss={noisy_train_loss:.4f} acc={noisy_train_acc:.4f} | "
            f"True train loss={true_train_loss:.4f} acc={true_train_acc:.4f} | "
            f"Test loss={test_loss:.4f} acc={test_acc:.4f}"
        )

        learning_results["true_training_loss"][epoch + 1]      = true_train_loss
        learning_results["noisy_training_loss"][epoch + 1]     = noisy_train_loss
        learning_results["testing_loss"][epoch + 1]            = test_loss
        learning_results["true_training_accuracy"][epoch + 1]  = true_train_acc
        learning_results["noisy_training_accuracy"][epoch + 1] = noisy_train_acc
        learning_results["testing_accuracy"][epoch + 1]        = test_acc

        if scheduler is not None:
            scheduler.step()

    # -----------------------------------------------------------------------
    # Save results and model
    # -----------------------------------------------------------------------
    if args.mode != "grad-descent":
        stem = _log_stem(args, sub_folder)
        with open(os.path.join(log_dir, f"{stem}_results.pkl"), "wb") as f:
            pickle.dump(learning_results, f)
        with open(os.path.join(log_dir, f"{stem}_weights.pkl"), "wb") as f:
            pickle.dump(result_collection, f)
        print(f"Logs saved to {log_dir}/{stem}_*.pkl")

    else:
        stem = _descent_log_stem(args)
        with open(os.path.join(log_dir, f"{stem}_results.pkl"), "wb") as f:
            pickle.dump(learning_results, f)
        print(f"Logs saved to {log_dir}/{stem}_results.pkl")

        # Save model as a reference for future grad-mimic runs
        if args.noisy_level == 0.0 and args.top_p == 1.0:
            ref_stem = _ref_model_stem(args, args.seed)
            if args.using_testset_to_build_ref:
                ref_stem += "_usingtest"
            model_path = os.path.join(model_dir, sub_folder, f"{ref_stem}.pt")
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

