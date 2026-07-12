import os
import json
import hydra
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import partial
from matplotlib import pyplot as plt

from cleanig.classifier.utils import get_classifier
from cleanig.dataset import (
    load_imagenet_datasets,
    load_oxfordpet_datasets,
    load_oxfordflower_datasets,
)
from cleanig.utils import set_seed, preprocess, undo_preprocess
from cleanig.explainer import (
    IGExplainer,
    AGIExplainer,
    GIGExplainer,
    EIGExplainer,
    MIGExplainer,
    LatentGIGExplainer,
    IG2Explainer,
    GradInputExplainer,
)
from cleanig.vae_wrapper import create_vae
from cleanig.metric.diffid import compute_diffid_score
from cleanig.plot_utils import abs_grayscale_norm


@hydra.main(config_path="../configs", config_name="ig", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    if args.save_dir is None:
        save_dir = f"results/benchmark_diffid/{args.dataset.dataset_name}/{args.explainer_name}/{args.model_name}"
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    if args.dataset.dataset_name == "oxfordpet":
        load_datasets = load_oxfordpet_datasets
    elif args.dataset.dataset_name == "oxfordflower":
        load_datasets = load_oxfordflower_datasets
    elif args.dataset.dataset_name == "imagenet":
        load_datasets = load_imagenet_datasets
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset.dataset_name}")

    _, val_loader = load_datasets(
        dataset_path=args.dataset.dataset_path,
        test_split=args.dataset.test_split,
        image_size=args.dataset.image_size,
        batch_size=args.dataset.batch_size,
        num_workers=args.dataset.num_workers,
        mean=args.dataset.mean,
        std=args.dataset.std,
        random_flip=args.dataset.random_flip,
        val_only=True,
    )

    # Preprocess functions
    preprocess_fn = partial(preprocess, mean=args.dataset.mean, std=args.dataset.std)
    undo_preprocess_fn = partial(
        undo_preprocess, mean=args.dataset.mean, std=args.dataset.std
    )

    # Load classifier
    model = get_classifier(
        model_name=args.model_name,
        dataset_name=args.dataset.dataset_name,
        image_size=args.dataset.image_size,
        num_classes=args.dataset.num_classes,
    ).to(args.device)
    model.eval()

    # ImageNet uses torchvision's pretrained weights directly, no checkpoint needed.
    # For OxfordPet / OxfordFlower we ship fine-tuned weights under checkpoints/.
    if args.dataset.dataset_name != "imagenet":
        # Resolve relative to the repository root so the script works regardless
        # of the current working directory (Hydra changes cwd by default).
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(
            repo_root,
            "checkpoints",
            f"classifier_{args.dataset.dataset_name}",
            f"{args.model_name}_best.pt",
        )
        checkpoint = torch.load(
            model_path, weights_only=False, map_location=args.device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Classifier loaded from {model_path}")
    else:
        print("Using pretrained ImageNet classifier")

    # Load explainer
    if args.explainer_name == "ig":
        explainer = IGExplainer(
            model=model,
            baseline_method=args.baseline_method,
            num_steps=args.num_steps,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
        )
    elif args.explainer_name == "agi":
        explainer = AGIExplainer(
            model=model,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            num_classes=args.dataset.num_classes,
            num_neg_cls=args.num_neg_cls,
            step_size=args.step_size,
            max_iter=args.max_iter,
            mean=args.dataset.mean,
            std=args.dataset.std,
        )
    elif args.explainer_name == "eig":
        vae = create_vae(args.vae_type, preprocess_fn, undo_preprocess_fn, args.device)
        explainer = EIGExplainer(
            model=model,
            vae=vae,
            baseline_method=args.baseline_method,
            num_steps=args.num_steps,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
        )
    elif args.explainer_name == "mig":
        vae = create_vae(args.vae_type, preprocess_fn, undo_preprocess_fn, args.device)
        explainer = MIGExplainer(
            model=model,
            vae=vae,
            baseline_method=args.baseline_method,
            num_steps=args.num_steps,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
            epsilon=args.epsilon,
        )
    elif args.explainer_name == "gig":
        explainer = GIGExplainer(
            model=model,
            baseline_method=args.baseline_method,
            num_steps=args.num_steps,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            fraction=args.fraction,
        )
    elif args.explainer_name == "latent_gig":
        vae = create_vae(args.vae_type, preprocess_fn, undo_preprocess_fn, args.device)
        explainer = LatentGIGExplainer(
            model=model,
            vae=vae,
            baseline_method=args.baseline_method,
            num_steps=args.num_steps,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            fraction=args.fraction,
            use_slerp=args.use_slerp,
        )
    elif args.explainer_name == "ig2":
        explainer = IG2Explainer(
            model=model,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            steps=args.steps,
            step_size=args.step_size,
            reference_mode=args.reference_mode,
        )
        reference_images = []
        _, ref_loader = load_datasets(
            dataset_path=args.dataset.dataset_path,
            test_split=args.dataset.test_split,
            image_size=args.dataset.image_size,
            batch_size=args.dataset.batch_size,
            num_workers=args.dataset.num_workers,
            mean=args.dataset.mean,
            std=args.dataset.std,
            random_flip=args.dataset.random_flip,
            val_only=True,
        )
        for ref_images, _ in ref_loader:
            reference_images.append(ref_images)
            if len(reference_images) * args.dataset.batch_size >= 100:
                break
        reference_images = torch.cat(reference_images, dim=0)[:100].to(args.device)
        explainer.set_reference_bank(reference_images)
    elif args.explainer_name == "grad_input":
        explainer = GradInputExplainer(
            model=model,
            device=args.device,
            exp_obj=args.exp_obj,
            preprocess_fn=preprocess_fn,
            baseline_method=args.baseline_method,
        )
    else:
        raise ValueError(f"Unsupported explainer: {args.explainer_name}")

    # Collect data
    all_images = []
    all_labels = []
    all_attributions = []
    all_scores = []

    # For visualization
    sample_images = []
    sample_attributions = []
    vis_count = args.num_vis_samples

    max_eval = args.max_eval_samples

    total_evaluated = 0
    print(
        f"\nComputing attributions on validation set (max: {max_eval if max_eval else 'all'})..."
    )

    # Calculate total iterations for tqdm
    total_iters = None
    if max_eval is not None:
        total_iters = (
            max_eval + args.dataset.batch_size - 1
        ) // args.dataset.batch_size

    for images, labels in tqdm(
        val_loader, desc="Computing attributions", total=total_iters
    ):
        # Check if we've reached the limit
        if max_eval is not None and total_evaluated >= max_eval:
            break

        images = images.to(args.device)
        labels = labels.to(args.device)

        # Limit batch size if necessary
        if max_eval is not None:
            remaining = max_eval - total_evaluated
            if remaining < images.shape[0]:
                images = images[:remaining]
                labels = labels[:remaining]

        # Compute attributions
        attributions = explainer.get_attributions(images, labels=labels)
        score = compute_diffid_score(
            model,
            images,
            attributions,
            labels=labels,
            baseline_method="mean",
            use_soft_metric=True,
        )

        # Store for batch processing
        all_images.append(images)
        all_labels.append(labels)
        all_attributions.append(attributions)
        all_scores.append(score)

        # Save samples for visualization
        if len(sample_images) < vis_count:
            n_to_save = min(vis_count - len(sample_images), images.shape[0])
            sample_images.extend([images[i].cpu() for i in range(n_to_save)])
            sample_attributions.extend(
                [attributions[i].cpu() for i in range(n_to_save)]
            )

        total_evaluated += images.shape[0]

    # Concatenate all batches
    print(f"\nComputing DiffID metrics for {total_evaluated} samples...")
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_attributions = torch.cat(all_attributions, dim=0)

    images_np = (
        undo_preprocess_fn(all_images).detach().cpu().numpy().transpose(0, 2, 3, 1)
    )
    attributions_np = all_attributions.detach().cpu().numpy().transpose(0, 2, 3, 1)
    labels_np = all_labels.detach().cpu().numpy()
    scores_np = np.array(all_scores)

    # Save data
    np.savez(
        os.path.join(save_dir, "data.npz"),
        images=images_np,
        attributions=attributions_np,
        labels=labels_np,
        scores=scores_np,
    )

    # Compute metrics on all data at once
    diffid_score, curves = compute_diffid_score(
        model,
        all_images,
        all_attributions,
        all_labels,
        use_soft_metric=False,
        baseline_method="mean",
        return_curves=True,
    )

    # Store metrics (single values for entire dataset)
    all_diffid_scores = [diffid_score]
    all_insertion_aucs = [np.mean(curves["insertion_scores"])]
    all_deletion_aucs = [np.mean(curves["deletion_scores"])]

    # Compute statistics
    results = {
        "num_samples_evaluated": total_evaluated,
        "diffid": float(np.mean(all_diffid_scores)),
        "insertion_auc": float(np.mean(all_insertion_aucs)),
        "deletion_auc": float(np.mean(all_deletion_aucs)),
        "config": OmegaConf.to_container(args, resolve=True),
    }

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {total_evaluated}")
    print(f"DiffID:        {results['diffid']:.4f}")
    print(f"Insertion AUC: {results['insertion_auc']:.4f}")
    print(f"Deletion AUC:  {results['deletion_auc']:.4f}")
    print("=" * 60)

    # Save results
    results_path = os.path.join(save_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Visualize samples
    print(f"\nCreating visualization for {len(sample_images)} samples...")
    n_cols = 2  # Original + Attribution
    n_rows = len(sample_images)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        # Original image
        img = (
            undo_preprocess_fn(sample_images[i].unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .clip(0, 1)
        )
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i + 1}: Original")
        axes[i, 0].axis("off")

        # Attribution
        attr = sample_attributions[i].detach().cpu().numpy().transpose(1, 2, 0)
        axes[i, 1].imshow(abs_grayscale_norm(attr), cmap="gray")
        axes[i, 1].set_title(f"Sample {i + 1}: Attribution")
        axes[i, 1].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(save_dir, "attributions.png")
    plt.savefig(vis_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    pipeline()
