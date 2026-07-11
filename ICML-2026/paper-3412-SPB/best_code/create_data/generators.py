import os
import torch
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

from create_data.dataset import set_seed


def generate_transformed_dataset(
    base_dataset,
    transform_fn,
    out_dir,
    train_split,
    seed=0,
    verbose=True,
    to_tensor_fn=None,
    copies_per_sample=1
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    def process(dataset, name):
        data_list, labels, metadata = [], [], []

        print(f"Generate {name} set")

        for data, label in tqdm(dataset, desc=f"Processing {name}", unit="sample"):

            for _ in range(copies_per_sample):

                data_t, meta = transform_fn(data)

                tensor = to_tensor_fn(data_t) if to_tensor_fn else data_t

                data_list.append(tensor)
                labels.append(int(label))
                metadata.append(meta)

        data_tensor = torch.stack(data_list)
        labels = torch.tensor(labels)

        return data_tensor, labels, metadata

    train_set, test_set = base_dataset

    if len(train_set) > 90000:
        idx = np.random.choice(len(train_set), 50000, replace=False)
        train_set = Subset(train_set, idx)
    if len(test_set) > 15000:
        idx = np.random.choice(len(test_set), 10000, replace=False)
        test_set = Subset(test_set, idx)

    data, labels, metadata = process(train_set, "train")

    # shuffle and split dataset into train/val/prior
    perm = torch.randperm(len(data))
    n_train, n_val, n_prior = train_split

    splits = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train+n_val],
        "prior": perm[n_train+n_val:n_train+n_val+n_prior],
    }

    def pack(indices):
        return {
            "images": data[indices],
            "labels": labels[indices],
            "meta": [metadata[i] for i in indices],
        }

    for name, idx in splits.items():
        torch.save(pack(idx), os.path.join(out_dir, f"{name}.pt"))

    data_t, labels_t, metadata_t = process(test_set, "test")

    torch.save({
        "images": data_t,
        "labels": labels_t,
        "meta": metadata_t,
    }, os.path.join(out_dir, "test.pt"))

    return out_dir
