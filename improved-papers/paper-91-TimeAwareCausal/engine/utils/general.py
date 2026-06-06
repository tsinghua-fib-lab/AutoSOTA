from pathlib import Path
import os
import torch
import numpy as np
import pickle

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class Pool():
    def __init__(self, save_path):
        self.x = None
        self.y = None
        self.save_path = save_path

    def add(self, datapoints, labels):
        if self.x is None:
            self.x = datapoints
            self.y = labels
        else:
            self.x = torch.cat([self.x, datapoints], dim=0)
            self.y = torch.cat([self.y, labels])

    def reset(self):
        self.x = None
        self.y = None

    def save(self):
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()

        _, indices = np.unique(x[::-1], axis=0, return_index=True)
        indices = len(x) - 1 - indices
        indices = np.sort(indices)
        x = x[indices]
        y = y[indices]

        final_data = [x, y]
        with open(self.save_path, 'wb') as f:
            pickle.dump(final_data, f)