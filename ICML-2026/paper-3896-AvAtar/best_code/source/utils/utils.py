import json
import torch

from utils.dicts import init_params_dict, train_params_dict


def read_settings(alg, dataset):
    import os
    repo_path = os.environ.get("AVATAR_HOME", "/repo")
    settings_path = f"{repo_path}/settings/{alg}/{dataset}.json"
    with open(settings_path, 'r') as f:
        settings_dict = json.load(f)
    init_settings = {key: settings_dict[key] for key in init_params_dict[alg] if key in settings_dict}
    train_settings = {key: settings_dict[key] for key in train_params_dict[alg] if key in settings_dict}
    return init_settings, train_settings


def get_unnorm_laplacian(adj):
    dtype = adj.dtype
    deg = torch.sum(adj, dim=1)
    D = torch.diag(deg).to(dtype).to(adj.device)
    lap_mat = D - adj.to(dtype).to(adj.device)
    return lap_mat


def get_sym_norm_laplacian(adj):
    dtype = adj.dtype
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt).to(dtype).to(adj.device)
    lap_mat = torch.eye(adj.shape[0], dtype=dtype).to(adj.device) - D_inv_sqrt @ adj.to(dtype) @ D_inv_sqrt
    return lap_mat


def get_rw_norm_laplacian(adj):
    dtype = adj.dtype
    deg = torch.sum(adj, dim=1)
    deg_inv = torch.pow(deg, -1.0)
    deg_inv[deg_inv == float('inf')] = 0.0
    D_inv = torch.diag(deg_inv).to(dtype).to(adj.device)
    lap_mat = torch.eye(adj.shape[0], dtype=dtype).to(adj.device) - D_inv @ adj.to(dtype)
    return lap_mat


def setdiff(a, b):
    """
    Find the difference of two tensors.
    :param a: tensor 1 (n1 x 2)
    :param b: tensor 2 (n2 x 2)
    :return: c: difference of a and b (n3 x 2)
    """
    # Use a structured array to perform row-wise set difference
    a_view = a.view(-1, 1, 2)  # Convert a(tensor) to (n1, 1, 2)
    b_view = b.view(1, -1, 2)  # Convert b(tensor) to (1, n2, 2)

    # Compare each element of a with each element of b
    mask = torch.all(a_view != b_view, dim=2).all(dim=1)

    # Select elements in a that are not in b
    c = a[mask]

    return c


def print_config(args, title="Configuration"):
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    max_key_len = max(len(k) for k in vars(args).keys())

    for k, v in sorted(vars(args).items()):
        print(f"{k.ljust(max_key_len)} : {v}")

    print("=" * 60 + "\n")


def get_sup_H(num_img, num_classes, gnd_labels, sup_ratio=0.2, seed=42, dtype=torch.float32):
    H = torch.zeros((num_img, num_classes)).to(dtype)

    chosen_idx = torch.arange(num_img)
    if seed is not None:
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    perm = torch.randperm(num_img)
    if seed is not None:
        torch.random.set_rng_state(state)

    chosen_idx = perm[:int(sup_ratio * num_img)]
    H[chosen_idx, gnd_labels[chosen_idx]] = 1.0

    return H


def get_init_batched_H(test_loader, num_classes, init_sup_ratio, seed=42, dtype=torch.float32):
    batched_H = []
    for images, labels in test_loader:
        batch_size = images.size(0)
        H_batch = get_sup_H(batch_size, num_classes, labels, sup_ratio=init_sup_ratio, seed=seed, dtype=dtype)
        batched_H.append(H_batch)
    return batched_H


def extract_object_crops(
    image,
    target,
    min_area=0,
    ignore_crowd=True
):
    """
    Extract object crops from a COCO image.

    Args:
        image (PIL.Image): image from dataset[i]
        target (list[dict]): CocoDetection target
        min_area (float): minimum bbox area to keep
        ignore_crowd (bool): ignore iscrowd annotations

    Returns:
        crops (list[PIL.Image]) or
        (crops, labels) if return_labels=True
    """
    crops = []
    labels = []

    W, H = image.size

    for ann in target:
        if ignore_crowd and ann.get("iscrowd", 0) == 1:
            continue

        x, y, w, h = ann["bbox"]
        if w * h < min_area:
            continue

        # Clamp to image boundaries
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), W)
        y2 = min(int(y + h), H)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
        labels.append(ann["category_id"])

    return crops, labels
