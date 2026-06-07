"""
Run example:
python scripts/export_student_split.py \
  --target-dir /path/to/clinical/train/images \
  --monet-csv /path/to/MONET_metadata.csv \
  --output-dir outputs/student

Use `python scripts/export_student_split.py --help` for all options.
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cofida.cli import export_split_parser
from cofida.data import find_images, load_monet_lookup
from cofida.student import split_target_paths


def main():
    args = export_split_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    monet_lookup, _ = load_monet_lookup(args.monet_csv)
    all_paths = find_images(args.target_dir)
    kept, train_paths, val_paths = split_target_paths(all_paths, monet_lookup, args.seed, args.val_split)
    print(f"Total target images: {len(all_paths)} | kept with MONET: {len(kept)}")
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
    pd.DataFrame({"path": train_paths}).to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    pd.DataFrame({"path": val_paths}).to_csv(os.path.join(args.output_dir, "val_split.csv"), index=False)
    print(f"Saved split CSVs to: {args.output_dir}")


if __name__ == "__main__":
    main()
