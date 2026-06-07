"""
Run example:
python scripts/train_teacher.py \
  --source-dir /path/to/train/images \
  --target-dir /path/to/train/images \
  --target-val-dir /path/to/val/images \
  --monet-csv /path/to/MILK10k_Training_Metadata.csv \
  --save-dir /path/to/teacher

Use `python scripts/train_teacher.py --help` for all options.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cofida.cli import teacher_parser
from cofida.teacher import train_teacher


if __name__ == "__main__":
    train_teacher(teacher_parser().parse_args())
    
