"""Monkey-patch torchvision CIFAR-10 to skip MD5 integrity check."""
import os
import torchvision.datasets.cifar

_original_check = torchvision.datasets.cifar.CIFAR10._check_integrity
_original_load_meta = torchvision.datasets.cifar.CIFAR10._load_meta

def _patched_check(self):
    """Return True if the expected files exist (skip MD5)."""
    for filename, _ in self.train_list + self.test_list:
        fpath = os.path.join(self.root, self.base_folder, filename)
        if not os.path.exists(fpath):
            return False
    return True

def _patched_load_meta(self):
    """Load metadata without MD5 check."""
    import pickle
    path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    if not os.path.exists(path):
        raise RuntimeError("Dataset metadata file not found. You can use download=True to download it")
    with open(path, "rb") as infile:
        data = pickle.load(infile, encoding="latin1")
        self.classes = data[self.meta["key"]]
    self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

torchvision.datasets.cifar.CIFAR10._check_integrity = _patched_check
torchvision.datasets.cifar.CIFAR10._load_meta = _patched_load_meta
print("CIFAR-10 integrity check patched successfully")
