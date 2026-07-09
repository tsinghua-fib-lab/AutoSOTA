"""
To run this file on 4 GPUs, use the following command:
``python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 pretrain.py --seed 42``
"""

import argparse
import random
import torch
import os

import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer


def resize(X, size=512):
    """
    Resize time series via interpolation.

    Parameters
    ----------
    X: array-like of shape (n_samples, 1, seq_len)
        Original input samples.
    size: int, default=512
        Length to which time series will be interpolated.
    
    Returns
    -------
    X_scaled: array-like of shape (n_samples, 1, size)
        The resized input.
    """
    X_scaled = F.interpolate(torch.tensor(X, dtype=torch.float), size=size, mode='linear', align_corners=False)
    return X_scaled.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility of a single experiment. Note that it is reproducible only on the same machine.")
    parser.add_argument('--file_name', type=str, default=None, help="Where to save the final checkpoint. By default, the checkpoint is not saved anywhere.")
    args = parser.parse_args()
    seed = args.seed
    file_name = args.file_name

    # ========= Initialize Distributed Environment =========
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # NCCL backend for distributed initialization
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # ========= Set Random Seed =========
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========= Set Device =========
    # each process corresponds to one GPU, use local_rank to specify which GPU to use for the current process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ========= Load Data =========
    # for the sake of simplicity, this dataset will be used both for pre-training and downstream performance
    data = [np.load(f'../data/GestureMidAirD1/{variable}_{set_name}.npy')
            for variable in ['X', 'y'] for set_name in ['train', 'test']]
    X_train, X_test, y_train, y_test = data
    if rank == 0:
        print("Original X_train dims: ", X_train.shape)
        print("Original X_test dims: ", X_test.shape)

    # resize sequences to 512 (or any multiple of num_patches)    
    X_train, X_test = resize(X_train, size=512), resize(X_test, size=512)
    if rank == 0:
        print("X_train dims after resizing: ", X_train.shape)
        print("X_test dims after resizing: ", X_test.shape)

    # ========= Initialize Mantis Model =========
    network = Mantis8M(pre_training=True)    
    model = MantisTrainer(device=device, network=network)

    # ========= Pre-training =========
    if rank == 0:
        print("Starting to pre-train")
        # if file_name is not None, create absent folders in the path
        if file_name is not None:
            dir_path = os.path.dirname(file_name)
            if dir_path != "":
                os.makedirs(dir_path, exist_ok=True)

    model.pretrain(
        X_train,
        num_epochs=100,                 # Adjust the number of epochs as needed
        batch_size=512,                 # Adjust batch size based on GPU memory
        base_learning_rate=2e-3,        # Initial learning rate
        data_parallel=True,             # Enable distributed data parallelism
        learning_rate_adjusting=True,   # Cosine annealing
        file_name=file_name             # Where to save the final checkpoint
    )
    # if the checkpoint was saved, you can load it later by ``model.load('file_name')``

    if rank == 0:
        print("Pre-training is finished")
    
    # ========= Test Downstream Performance =========
    if rank == 0:        
        # get embeddings
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
        # learn a classifier
        predictor = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        predictor.fit(Z_train, y_train)
        # evaluate performance
        y_pred = predictor.predict(Z_test)
        print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')
