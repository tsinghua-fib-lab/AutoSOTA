import torch
from tqdm import tqdm
from alignment_utils import (
    get_accuracies,
    get_loss_func,
    get_one_to_one_features,
    l2_norm
)

from w_alignment.w_thin import procrustes_align
import argparse
from typing import Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
from utils import feature_augmentation
import numpy as np
import torch.nn.functional as F
from torch import nn

def spectral_projection(matrix: torch.Tensor) -> torch.Tensor:
    """Project a matrix onto the spectral space (SVD + clamp 0~1)."""
    u, s, v = torch.svd(matrix)
    s.clamp_(0.0, 1.0)
    return u @ torch.diag(s) @ v.T

def refine_mapping(
    args: argparse.Namespace,
    loss_func: Callable,
    init_transfm: torch.Tensor,
    train_arrays: Dict[str, torch.Tensor],
    test_arrays: List[Dict[str, torch.Tensor]],
    train_visual_feats: torch.Tensor,
    train_text_feats: Optional[torch.Tensor] = None,
    class_prototypes: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    momentum: float = 0.9,
    verbose: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transfm = nn.Parameter(init_transfm.clone().to(device))

    num_instances = train_visual_feats.size(0) // 5 if args.five_crop else train_visual_feats.size(0)
    if batch_size is None:
        batch_size = num_instances if args.five_crop else int(num_instances * 0.75)

    # Move all relevant tensors to device
    train_visual_feats = train_visual_feats.to(device)
    if train_text_feats is not None:
        train_text_feats = train_text_feats.to(device)
    if class_prototypes is not None:
        class_prototypes = class_prototypes.to(device)
    if labels is not None:
        labels = labels.to(device)

    optimizer = torch.optim.AdamW(
        [transfm],
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_iters, eta_min=args.cosine_end_lr
    )

    iterator = tqdm(range(args.n_iters)) if verbose else range(args.n_iters)

    for current_iter in iterator:
        optimizer.zero_grad()

        # Sample batch
        batch_idx = torch.randperm(num_instances)[:batch_size]
        batch_visual = train_visual_feats[batch_idx]
        batch_text = train_text_feats[batch_idx] if train_text_feats is not None else None
        batch_labels = labels[batch_idx] if labels is not None else None

        # Feature augmentations
        if args.interpolate_features:
            batch_visual, batch_labels = feature_augmentation(batch_visual, batch_labels)

        if args.gaussian_noise > 0.0 and np.random.rand() > 0.5:
            batch_visual += torch.randn_like(batch_visual) * args.gaussian_noise

        if args.dropout > 0.0 and np.random.rand() > 0.5:
            batch_visual = F.dropout(batch_visual, p=args.dropout)

        # Compute loss
        loss = loss_func(
            visual_features=batch_visual @ transfm,
            text_features=batch_text,
            class_prototypes=class_prototypes,
            labels=batch_labels,
            knn=args.knn,
            total_iters=args.n_iters,
            cur_iter=current_iter,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([transfm], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Project to valid matrix space
        with torch.no_grad():
            if args.spectral_proj:
                transfm.data = spectral_projection(transfm.data)
            elif args.orthogonalize:
                mat = transfm.data
                transfm.data = (1 + args.orth_beta) * mat - args.orth_beta * (mat @ mat.T @ mat)

        # Optionally show progress
        if verbose:
            iterator.set_description(f"Loss: {loss.item():.4f}")

    return transfm.detach().cpu()

def alignment(
    args: argparse.Namespace,
    train_features,
    test_features,
    clip_prototypes,
    test_labels,
    P,
) -> torch.Tensor:
    
    gpu_id = args.gpu_id 
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    soft_assignments = P.to("cpu")
    labels = soft_assignments.argmax(-1)

    # ==============================
    train_arrays = {
        "text_features": clip_prototypes,
        "visual_features": train_features,
        "labels": labels,
    }
    test_arrays = [{
        "text_features": clip_prototypes,
        "visual_features": test_features,
        "labels": test_labels,
    }]

    train_text_feats = get_one_to_one_features(
        train_arrays["visual_features"],
        train_arrays["text_features"],
        train_arrays["labels"],
    )
    transfm = procrustes_align(
        train_arrays["visual_features"].cuda(),
        train_text_feats.cuda(),
        beta=0.9,
    )

    logger.info("Mapping refinement ...")
    transfm = refine_mapping(
        args,
        loss_func=get_loss_func(args),
        init_transfm=transfm,
        train_arrays=train_arrays,
        test_arrays=test_arrays,
        train_visual_feats=train_arrays["visual_features"],
        train_text_feats=train_text_feats,
        class_prototypes=train_arrays["text_features"],
        labels=train_arrays["labels"],
        batch_size=args.batch_size,
    )

    accuracies = get_accuracies(
        train_arrays,
        test_arrays,
        transform=transfm,
    )
    logger.info(f"After refinement results: {accuracies}\n")

    class_prototypes = test_arrays[0]["text_features"].to(device) 
    visual_feats = test_arrays[0]["visual_features"].to(device) 
    labels = test_arrays[0]["labels"].to(device)
    visual_feats = l2_norm(visual_feats @ transfm.to(device))
    logits = visual_feats @ class_prototypes.T
    preds = logits.argmax(dim=1)      

    acc = (preds == labels).float().mean().item()
    print(f"[Test] Accuracy: {acc * 100:.2f}%")

    transfm = transfm.to(device)
    return acc

    