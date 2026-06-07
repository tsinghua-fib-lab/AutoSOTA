"""
Run example:
python scripts/train_student.py \
  --teacher-checkpoint /path/to/best_cofida_monet.pt \
  --target-dir /path/to/train/images \
  --monet-csv /path/to/MILK10k_Training_Metadata.csv \
  --save-dir /path/to/student

Use `python scripts/train_student.py --help` for all options.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cofida.cli import student_parser
from cofida.student import train_student


if __name__ == "__main__":
    train_student(student_parser().parse_args())
