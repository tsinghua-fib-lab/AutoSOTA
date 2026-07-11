import torch
import numpy as np
import matplotlib.pyplot as plt


def show_grid(
    pt_path,
    out_path,
    n=16,
    seed=0,
    label_filter=None,
    title_fn=None,
    out_file="grid.png",
):
    torch.manual_seed(seed)
    data = torch.load(pt_path)

    imgs = data["images"]
    labels = data["labels"]
    meta = data.get("meta", None)

    if label_filter is not None:
        mask = labels == label_filter
        imgs = imgs[mask]
        labels = labels[mask]
        if meta is not None:
            meta = [m for i, m in enumerate(meta) if mask[i]]

    idxs = torch.randperm(len(imgs))[:n]

    ncols = int(np.sqrt(n))
    nrows = (n + ncols - 1) // ncols

    plt.figure(figsize=(ncols * 2, nrows * 2))

    for i, idx in enumerate(idxs):
        img = imgs[idx]

        if img.shape[0] == 1:
            img = img.squeeze().numpy()
            cmap = "gray"
        else:
            img = img.permute(1, 2, 0).numpy()
            cmap = None

        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img, cmap=cmap)

        if title_fn is not None:
            plt.title(title_fn(labels[idx], meta[idx] if meta else None))

        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path + "/" + out_file)