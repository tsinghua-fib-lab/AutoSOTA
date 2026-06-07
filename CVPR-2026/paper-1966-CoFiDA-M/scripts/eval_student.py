"""
Run example:
python scripts/eval_student.py \
  --test-dir /path/to/test \
  --checkpoint path/to/best_student.pt \

MIDAS-style example:
python scripts/eval_student.py \
  --test-dir /path/to/midas/images/clinical \
  --checkpoint outputs/student/best_student.pt \
  --auto-map-melanoma

Use `python scripts/eval_student.py --help` for all options.
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cofida.checkpointing import load_checkpoint_safe
from cofida.cli import eval_student_parser
from cofida.data import BinaryImageFolderDataset, make_eval_transform
from cofida.evaluate import print_eval_report
from cofida.metrics import youden_threshold
from cofida.models import StudentImageOnly
from cofida.utils import device_and_flags, find_melanoma_class


def main():
    args = eval_student_parser().parse_args()
    runtime = device_and_flags()
    print(f"Device: {runtime.device.type}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    base_dataset = datasets.ImageFolder(args.test_dir, transform=make_eval_transform(args.img_size))
    if args.auto_map_melanoma:
        mel_class = find_melanoma_class(base_dataset.classes)
        if mel_class is None:
            raise ValueError(f"Could not find a melanoma class among: {base_dataset.classes}")
        idx_positive = base_dataset.class_to_idx[mel_class]
        print(f"Mapping: '{mel_class}' -> 1 (mel), all others -> 0 (other)")
    else:
        idx_mel = base_dataset.class_to_idx.get("mel")
        idx_other = base_dataset.class_to_idx.get("other")
        if idx_mel is None or idx_other is None:
            raise ValueError(f"class_to_idx={base_dataset.class_to_idx} must contain 'mel' and 'other'")
        idx_positive = idx_mel

    eval_dataset = BinaryImageFolderDataset(base_dataset, idx_positive)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=runtime.pin_memory,
    )

    checkpoint = load_checkpoint_safe(args.checkpoint)
    model = StudentImageOnly(num_classes=2, hidden=512).to(runtime.device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print("State dict info:", {"missing": missing, "unexpected": unexpected})
    model.eval()

    paths, y_true, y_prob = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["img"].to(runtime.device)
            logits = model.logits(images)
            prob = torch.softmax(logits, dim=1)[:, 1]
            paths.extend(batch["path"])
            y_true.extend(batch["y"].cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    print_eval_report(
        "Student (image-only) evaluation",
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        out_csv=args.out_csv,
        paths=paths,
    )

    optimal = youden_threshold(y_true, y_prob)
    print("\nOptimal threshold by Youden's J")
    print(f"Optimal threshold : {optimal['threshold']:.3f}")
    print(f"Accuracy @ opt    : {optimal['acc']:.4f}")
    print(f"Balanced Acc @ opt: {optimal['bacc']:.4f}")
    print("Confusion matrix (rows=true, cols=pred) @ opt:")
    print(optimal["cm"])


if __name__ == "__main__":
    main()
