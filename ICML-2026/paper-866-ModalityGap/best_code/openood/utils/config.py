import os
import random
import numpy as np
import torch


class Config:
    """
    Thin wrapper that converts a YAML-loaded dict into an object with
    attribute-style access (cfg.key) so that all downstream modules can use
    dot notation instead of dict subscripting.

    On construction, the following automatic adjustments are applied:
      - device is inferred from CUDA availability if not set in the YAML.
      - id_imglist and every path in ood_datasets are converted from relative
        paths to absolute paths by joining them with data_root.
    """

    def __init__(self, d: dict):
        for key, value in d.items():
            setattr(self, key, value)

        # Infer device if not explicitly specified.
        if not hasattr(self, 'device'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resolve relative imglist paths relative to data_root so callers
        # never need to manually join paths.
        if hasattr(self, 'data_root') and hasattr(self, 'id_imglist'):
            if not os.path.isabs(self.id_imglist):
                self.id_imglist = os.path.join(self.data_root, self.id_imglist)

        if hasattr(self, 'data_root') and hasattr(self, 'ood_datasets'):
            self.ood_datasets = {
                name: (p if os.path.isabs(p) else os.path.join(self.data_root, p))
                for name, p in self.ood_datasets.items()
            }

    def __repr__(self):
        lines = [f'Config(']
        for k, v in self.__dict__.items():
            lines.append(f'  {k}={v!r},')
        lines.append(')')
        return '\n'.join(lines)


def set_random_seed(seed: int) -> None:
    """Fix all sources of randomness for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
