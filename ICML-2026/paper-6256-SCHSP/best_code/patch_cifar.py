"""Patch torchvision CIFAR-10/100 to use pre-cached data in both numpy formats."""
import pickle, os, numpy as np
import torchvision.datasets.cifar as cifar_module
from torchvision.datasets.cifar import CIFAR10, CIFAR100

_original_cifar100_init = CIFAR100.__init__

def _load_cifar(self, root, base_folder, train):
    """Load CIFAR data from pre-cached files in either format."""
    fname = "train" if train else "test"
    filepath = os.path.join(root, base_folder, fname)
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    
    data = entry[b"data"]
    fine_labels = entry[b"fine_labels"] if b"fine_labels" in entry else entry["fine_labels"]
    
    # Handle both formats
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 3072:
            # Standard CIFAR format: (N, 3072) -> (N, 32, 32, 3)
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        elif data.ndim == 4:
            # Pre-processed format: (N, 3, 32, 32) -> (N, 32, 32, 3)
            data = data.transpose(0, 2, 3, 1)
        # Convert to list of (H, W, C) arrays
        self.data = [data[i] for i in range(len(data))]
    elif isinstance(data, list):
        self.data = data
    
    if isinstance(fine_labels, np.ndarray):
        self.targets = fine_labels.tolist()
    elif isinstance(fine_labels, list):
        self.targets = list(fine_labels)
    else:
        self.targets = list(fine_labels)
    
    return True

def _patched_cifar_init(self, root, train=True, transform=None, target_transform=None, download=False):
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.root = root
    result = _load_cifar(self, root, self.base_folder, train)
    if result is not None:
        return
    _original_cifar100_init(self, root, train, transform, target_transform, download)

def _patched_check_integrity(self):
    base = os.path.join(self.root, self.base_folder)
    for filename, _ in self.train_list + self.test_list:
        if not os.path.exists(os.path.join(base, filename)):
            return False
    return True

CIFAR100.__init__ = _patched_cifar_init
CIFAR10.__init__ = _patched_cifar_init
CIFAR100._check_integrity = _patched_check_integrity
CIFAR10._check_integrity = _patched_check_integrity
print("Patched CIFAR-10/100 (v2, handles both data formats)")
