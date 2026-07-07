import argparse
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection

from save_features import get_save_name, save_features
from utils import topk_accuracies

try:
    from geomloss import SamplesLoss
except ImportError:
    logger.warning("Could not import geomloss, U-LFA cannot be run")


def l2_norm(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, p=2, keepdim=True)


def center_features(features: torch.Tensor) -> torch.Tensor:
    features = features - features.mean(dim=0, keepdims=True)
    return l2_norm(features)


def tensors_to_device(
    tensors: List[Optional[torch.Tensor]],
    device: str
) -> List[Optional[torch.Tensor]]:

    return [
        tensor.to(device, non_blocking=True) if tensor is not None else None
        for tensor in tensors
    ]


def sinkhorn_assignment(
    x_source: torch.Tensor,
    y_target: torch.Tensor,
    p: int = 2,
    blur: float = 0.05,
    scaling: float = 0.95,
    batch: int = 200,
    verbose: bool = True
) -> torch.Tensor:

    # based on GeomLoss examples
    # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_optimal_transport_labels.html

    if verbose:
        logger.info("Generating the assignment with sinkhorn ...")

    N, M, D = x_source.shape[0], y_target.shape[0], x_source.shape[1]

    if torch.cuda.is_available():
        x_source, y_target = tensors_to_device([x_source, y_target], "cuda")

    # uniform weights
    x_source_w = torch.ones(N, device=x_source.device) / N
    y_target_w = torch.ones(M, device=x_source.device) / M

    sinkhorn_solver = SamplesLoss(
        loss="sinkhorn", p=p, blur=blur, scaling=scaling, debias=False, potentials=True
    )

    F, G = sinkhorn_solver(x_source_w, x_source, y_target_w, y_target)

    # Compute the transport plan (assignment matrix) from the potentials
    x_source = x_source.view(N, 1, D)
    x_source_weights = x_source_w.view(N, 1)

    y_target = y_target.view(1, M, D)
    y_target_weights = y_target_w.view(1, M)

    F, G = F.view(N, 1), G.view(1, M)

    soft_assignments = torch.zeros(N, M, device=x_source.device)

    for i in range(0, N, batch):
        # loop to avoid memory issues

        cost_matrix = (
            1 / p) * ((x_source[i: i + batch] - y_target) ** p).sum(-1)  # (N,M)
        eps = blur**p  # temperature epsilon

        # (N,M) transport plan
        transport_plan = ((F[i: i + batch] + G - cost_matrix) / eps).exp()
        transport_plan = transport_plan * (
            x_source_weights[i: i + batch] * y_target_weights
        )

        soft_assignments[i: i + batch] = transport_plan / transport_plan.sum(
            dim=1, keepdim=True
        )

    return soft_assignments.cpu()

def get_loss_func(args: argparse.Namespace) -> Callable:

    if args.refinement_loss == "csls":
        logger.info("Using CSLS loss")
        return partial(csls_loss, knn=args.knn)

    if args.refinement_loss == "adaptive":
        logger.info("Using adaptive reranking loss")
        return partial(adaptive_reranking_loss, knn=args.knn, scale=args.arerank_scale)

    if args.refinement_loss == "contrastive":
        logger.info("Using contrastive loss")
        return contrastive_loss
    
    if args.refinement_loss == "alignment":
        logger.info("alignment loss")
        return partial(alignment_loss)
    logger.info("Using triplet loss")

    return triplet_loss

def triplet_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    **_: Any,
) -> torch.Tensor:

    C = class_prototypes.size(0)
    device = visual_features.device

    scores = l2_norm(visual_features) @ l2_norm(class_prototypes).T
    _, indices = torch.sort(scores, dim=-1, descending=True)

    mask = (indices == labels.view(-1, 1).repeat(1, C)).float()
    class_position = mask.argmax(dim=-1).view(-1, 1)
    mask = torch.arange(C).view(1, -1).to(device) < class_position

    closes_clusters = mask.float().sum(-1)
    negatives_indx = torch.zeros_like(labels)
    negatives_indx[closes_clusters == 0.0] = indices[closes_clusters == 0.0, 1]
    negatives_indx[closes_clusters > 0.0] = indices[closes_clusters > 0.0, 0]

    positives = class_prototypes[labels]
    negatives = class_prototypes[negatives_indx]

    loss = nn.TripletMarginLoss(margin=margin, p=2)(
        visual_features, positives, negatives
    )
    return loss

def contrastive_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    **_: Any,
) -> torch.Tensor:

    logits = visual_features @ class_prototypes.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return torch.nn.functional.cross_entropy(logits, labels)

def adaptive_reranking_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    scale: float = 4.0,
    knn: int = 3,
    **_: torch.Tensor,
) -> torch.Tensor:

    N = visual_features.shape[0]
    C = class_prototypes.shape[0]
    knn = min(knn, C)

    visual_features = l2_norm(visual_features)
    class_prototypes = l2_norm(class_prototypes)

    distances = torch.cdist(visual_features, class_prototypes, p=2)

    sorted_distances, sorted_indices = torch.sort(
        distances, dim=1, descending=False)
    anchor = (
        ((visual_features - class_prototypes[labels])
         ** 2).sum(-1).sqrt().unsqueeze(1)
    )
    sorted_distances = sorted_distances[:, :knn]

    pos_cla_proto = class_prototypes[labels].unsqueeze(1)
    all_cls = class_prototypes[sorted_indices[:, :knn]]
    margins = (1.0 - (all_cls * pos_cla_proto).sum(-1)) / scale

    loss = torch.max(
        anchor + margins - sorted_distances,
        torch.zeros(N, knn).to(visual_features.device),
    )

    return loss.mean()

def alignment_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    scale: float = 2,
    knn: int = 3,
    hub_lambda: float = 2.0,
    total_iters: int = 200,
    cur_iter: int = 0,
        **_: torch.Tensor
):
    # ALGO-07: Hubness penalty exponential decay 2.0 -> 0.5
    if total_iters > 1 and cur_iter > 0:
        decay_ratio = cur_iter / (total_iters - 1)
        hub_lambda = 2.0 * (0.5 / 2.0) ** decay_ratio
    N, C = visual_features.size(0), class_prototypes.size(0)
    knn = min(knn, C)

    visual_features = l2_norm(visual_features)
    class_prototypes = l2_norm(class_prototypes)

    distances = torch.cdist(visual_features, class_prototypes, p=2)
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)

    # --- 1) 计算 prototype 的 hubness（出现频率） ---
    # 形状: [C]
    hub_scores = torch.bincount(sorted_indices[:, :knn].reshape(-1), minlength=C)
    hub_scores = hub_scores.float() / N  # normalize

    # --- 2) normal margin（类相似性 margin） ---
    pos_proto = class_prototypes[labels].unsqueeze(1)
    knn_cls = class_prototypes[sorted_indices[:, :knn]]
    base_margin = (1.0 - (knn_cls * pos_proto).sum(-1)) / scale

    # --- 3) hubness margin（关键改进） ---
    hub_margin = hub_lambda * hub_scores[sorted_indices[:, :knn]]

    # --- 4) 总 margin ---
    total_margin = base_margin + hub_margin

    # --- 5) hinge loss ---
    pos_dist = torch.norm(visual_features - class_prototypes[labels], dim=-1).unsqueeze(1)
    loss = torch.relu(pos_dist + total_margin - sorted_distances[:, :knn])

    return loss.mean()

def get_knn_avg_dist(
    features1: torch.Tensor,
    features2: torch.Tensor,
    knn: int = 10,
    **_: torch.Tensor,
) -> torch.Tensor:

    # get the top-k nearest neighbors
    scores = features1 @ features2.T
    topk_distances = scores.topk(int(knn), dim=1, largest=True, sorted=True)[0]
    # get the average distance
    average_dist = topk_distances.mean(dim=1)
    return average_dist

def csls(
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        knn: int = 10
) -> torch.Tensor:

    avg_source_to_target = get_knn_avg_dist(visual_features, text_features, 1)
    avg_target_to_source = get_knn_avg_dist(
        text_features, visual_features, knn)

    # scores
    csls_scores = 2 * (visual_features * text_features).sum(-1)
    csls_scores = csls_scores - avg_source_to_target - avg_target_to_source
    return csls_scores

def csls_loss(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    knn: int = 10,
    **_: Any
) -> torch.Tensor:
    csls_scores = -csls(l2_norm(visual_features), l2_norm(text_features), knn)
    return csls_scores.mean()

def get_acc(
    visual_feats: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
    softmax_temp: float = 0.07,
) -> Dict[str, float]:

    assert visual_feats.ndim == class_prototypes.ndim == 2
    assert visual_feats.size(1) == class_prototypes.size(1)

    logits = (visual_feats @ class_prototypes.T) / softmax_temp
    probabilities = logits.softmax(-1)
    topk_accs = topk_accuracies(probabilities, labels, topk)
    topk_accs = [acc.cpu().numpy().round(2) for acc in topk_accs]
    return {f"top_{i}": acc for i, acc in zip(topk, topk_accs)}

def get_accuracies(
    train_arrays: Dict[str, torch.Tensor],
    test_arrays: List[Dict[str, torch.Tensor]],
    transform: Union[torch.Tensor, None] = None,
    target_set_transform: Union[torch.Tensor, None] = None,
    five_crop: bool = False,
) -> Dict[str, Dict[str, float]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_set_transform is not None:
        assert len(test_arrays) == 2

    if len(test_arrays) == 1:
        names = ["train", "test"]
    else:
        names = ["train", "base", "new"]

    arrays = [train_arrays] + test_arrays

    accuracies = {}
    for arr, name in zip(arrays, names):
        class_prototypes = arr["text_features"].to(device)
        visual_feats = arr["visual_features"].to(device)
        labels = arr["labels"].to(device)

        if five_crop and name == "train":
            N_AUGMENTATIONS = 5
            n = visual_feats.shape[0]

            valid_n = (n // N_AUGMENTATIONS) * N_AUGMENTATIONS
            if valid_n < n:
                # print(f"[Warning] Dropping last {n - valid_n} samples for 5-crop alignment in {name}.")
                visual_feats = visual_feats[:valid_n]
                labels = labels[:valid_n]

            mask = torch.zeros(valid_n, dtype=torch.bool, device=visual_feats.device)
            mask[N_AUGMENTATIONS - 1::N_AUGMENTATIONS] = True

            visual_feats = visual_feats[mask]
            labels = labels[mask]

        class_prototypes = class_prototypes.to(device)
        visual_feats = visual_feats.to(device)
        labels = labels.to(device)

        if target_set_transform is not None and name == "new":
            visual_feats = l2_norm(
                visual_feats @ target_set_transform.to(device))

        elif transform is not None:
            # renormalize since transform might not be orthogonal
            visual_feats = l2_norm(visual_feats @ transform.to(device))

        accuracies[name] = get_acc(visual_feats, class_prototypes, labels)

    return accuracies

def get_one_to_one_features(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> torch.Tensor:

    if labels is not None:
        text_features = class_prototypes[labels]
        return text_features

    assignments = sinkhorn_assignment(visual_features, class_prototypes)
    text_features = assignments @ class_prototypes
    return text_features

def get_arrays_from_npz(
    npz_path: str,
    normalize: bool = False,
    center: bool = False,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:

    npz = np.load(npz_path)

    text_features = torch.from_numpy(npz["arr_0"]).to(device)
    visual_features = torch.from_numpy(npz["arr_1"]).to(device)
    labels = torch.from_numpy(npz["arr_2"]).to(device)

    if visual_features.ndim == 3:
        # average over the frames
        visual_features = visual_features.mean(dim=1)

    if normalize:
        text_features = l2_norm(text_features)
        visual_features = l2_norm(visual_features)

    if center:
        text_features = center_features(text_features)
        visual_features = center_features(visual_features)

    arrays = dict(
        text_features=text_features,
        visual_features=visual_features,
        labels=labels,
        filenames=npz["arr_3"],
        labelnames=npz["arr_4"],
        dataset_labelnames=npz["arr_5"],
    )

    return arrays

def get_VFM_arrays_from_npz(
    npz_path: str,
    normalize: bool = False,
    center: bool = False,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:

    npz = np.load(npz_path)

    
    visual_features = torch.from_numpy(npz["arr_0"]).to(device)
    labels = torch.from_numpy(npz["arr_1"]).to(device)

    if visual_features.ndim == 3:
        # average over the frames
        visual_features = visual_features.mean(dim=1)

    if normalize:
        #text_features = l2_norm(text_features)
        visual_features = l2_norm(visual_features)



    if center:
        #text_features = center_features(text_features)
        visual_features = center_features(visual_features)

    arrays = dict(
        visual_features=visual_features,
        labels=labels,
        filenames=npz["arr_2"],
        labelnames=npz["arr_3"],
        dataset_labelnames=npz["arr_4"],
    )

    return arrays

def load_features(
    cfg: Any,
    args: argparse.Namespace,
    normalize: bool = True,
    center: bool = False
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:

    save_name = get_save_name(cfg, args)
    save_path = os.path.join(args.save_path, save_name)
    print("save_path:", save_path)
    if not os.path.exists(save_path):
        logger.info("Feature not found, saving them ...")
        save_features(cfg, args)

    files = [f for f in os.listdir(save_path) if f.endswith(".npz")]
    assert len(files) == 2 or len(files) == 3
    
    if args.model_type in ["dinov2", "dinov3"]:
        train_dict = get_VFM_arrays_from_npz(
            f"{save_path}/train_features.npz", normalize, center
        )
    else:
        train_dict = get_arrays_from_npz(
            f"{save_path}/train_features.npz", normalize, center
        )

    if len(files) == 3:
        if cfg.DATA.USE_BASE_AND_NEW:
            if args.model_type in ["dinov2", "dinov3"]:
                test_base_dict = get_VFM_arrays_from_npz(
                        f"{save_path}/test_base_features.npz", normalize, center)
                test_new_dict = get_VFM_arrays_from_npz(
                    f"{save_path}/test_new_features.npz", normalize, center
                )
            
            else:
                test_base_dict = get_arrays_from_npz(
                    f"{save_path}/test_base_features.npz", normalize, center
                )
                test_new_dict = get_arrays_from_npz(
                    f"{save_path}/test_new_features.npz", normalize, center
                )

        else:
            test_base_dict = get_arrays_from_npz(
                f"{save_path}/source_test_features.npz", normalize, center
            )
            test_new_dict = get_arrays_from_npz(
                f"{save_path}/target_test_features.npz", normalize, center
            )

        test_dict = [test_base_dict, test_new_dict]

    else:
        if args.model_type in ["dinov2", "dinov3"]:
             test_dict = [
                get_VFM_arrays_from_npz(
                    f"{save_path}/test_features.npz", normalize, center)
            ]
        else:
            test_dict = [
                get_arrays_from_npz(
                    f"{save_path}/test_features.npz", normalize, center)
            ]

    logger.info(f"Train features: {train_dict['visual_features'].shape[0]}")
    logger.info(f"Test features: {test_dict[0]['visual_features'].shape[0]}")
    if len(test_dict) == 2:
        logger.info(
            f"Test features (new): {test_dict[1]['visual_features'].shape[0]}")
    if args.model_type not in ["dinov2", "dinov3"]:
        logger.info(f"Train classes: {train_dict['text_features'].shape[0]}")
        logger.info(f"Test classes: {test_dict[0]['text_features'].shape[0]}")
    if len(test_dict) == 2 and args.model_type not in ["dinov2", "dinov3"]:
        logger.info(
            f"Test classes (new): {test_dict[1]['text_features'].shape[0]}")

    return train_dict, test_dict

def gaussian_projection(
    train_dict: Dict[str, torch.Tensor],
    test_dict: List[Dict[str, torch.Tensor]],
    n_components: Optional[int] = None
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:

    logger.info("Applying a random projection")
    assert len(test_dict) == 1, "not implemented for base and new"

    text_shape = train_dict['text_features'].shape
    visual_shape = train_dict['visual_features'].shape

    n_components = min(text_shape[1], visual_shape[1]) \
        if n_components is None else n_components

    logger.info(
        f"Original shapes are: text: {text_shape} and image {visual_shape}")

    # Project the visual features
    if train_dict["visual_features"].shape[1] != n_components:
        assert train_dict["visual_features"].shape[1] > n_components

        transformer = GaussianRandomProjection(
            n_components=n_components, random_state=0)

        transformer.fit(train_dict["visual_features"])

        train_dict["visual_features"] = l2_norm(torch.from_numpy(
            transformer.transform(train_dict["visual_features"])
        ).float())

        test_dict[0]["visual_features"] = l2_norm(torch.from_numpy(
            transformer.transform(test_dict[0]["visual_features"])
        ).float())

    # Project the text features
    if test_dict[0]["text_features"].shape[1] != n_components:
        assert test_dict[0]["text_features"].shape[1] > n_components

        transformer = GaussianRandomProjection(
            n_components=n_components, random_state=0)

        transformer.fit(train_dict["text_features"])

        train_dict["text_features"] = l2_norm(torch.from_numpy(
            transformer.transform(train_dict["text_features"])
        ).float())

        test_dict[0]["text_features"] = l2_norm(torch.from_numpy(
            transformer.transform(test_dict[0]["text_features"])
        ).float())

    text_shape = train_dict['text_features'].shape
    visual_shape = train_dict['visual_features'].shape
    logger.info(
        f"Original shapes are: text: {text_shape} and image {visual_shape}")

    return train_dict, test_dict

def get_visual_baselines(
    train_arrays: Dict[str, torch.Tensor],
    test_arrays: List[Dict[str, torch.Tensor]],
    n_neighbors: int = 16,
) -> Dict[str, float]:

    logger.info("Linear probe and Knn baselines...")

    train_features = l2_norm(train_arrays["visual_features"])
    train_labels = train_arrays["labels"]

    # kNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(train_features, train_labels)

    y_pred = neigh.predict(test_arrays[0]["visual_features"])
    knn_acc = np.round(accuracy_score(
        test_arrays[0]["labels"], y_pred) * 100, 2)

    # linear probe
    clf = LogisticRegression(
        penalty="l2",
        random_state=0,
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
    )
    clf.fit(train_features, train_labels)

    y_pred = clf.predict(test_arrays[0]["visual_features"])
    linear_probe = np.round(accuracy_score(
        test_arrays[0]["labels"], y_pred) * 100, 2)

    return {
        "knn": knn_acc,
        "linear_probe": linear_probe,
    }
