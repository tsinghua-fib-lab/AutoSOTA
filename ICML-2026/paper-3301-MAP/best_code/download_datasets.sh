#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p data

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python - << 'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from datasets import _load_mnist_tensors

_load_mnist_tensors(root="data", train=True)
_load_mnist_tensors(root="data", train=False)

required_local = [
    Path("data/smileyface_plane.npy"),
    Path("data/smileyface_sphere.npy"),
    Path("data/stanford-bunny.obj"),
]
missing = [str(p) for p in required_local if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing required in-repo data files: {missing}")

print("MNIST raw IDX files and local data assets verified.")
print("Protein data is intentionally prepared separately for provenance.")
print(
    "To generate the default SidechainNet artifact, run: "
    "python training/process_protein_fragments.py --name casp12 "
    "--fragment-length 10 --max-data-length 20000"
)
PY
