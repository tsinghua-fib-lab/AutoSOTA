"""
Run example:
python scripts/eval_teacher.py \
  --test-dir  "/path/to/images" \
  --monet-csv "/path/to/MILK10k_Training_Metadata.csv" \
  --checkpoint "/path/to/best_cofida_monet.pt" \
  --out-csv "/path/to/clinical_val_predictions.csv"

Use `python scripts/eval_teacher.py --help` for all options.
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
from cofida.cli import eval_teacher_parser
from cofida.data import MonetEvalDataset, load_monet_lookup, make_eval_transform
from cofida.evaluate import print_eval_report
from cofida.models import CoFIDAMonet
from cofida.utils import device_and_flags


def main():
    args = eval_teacher_parser().parse_args()
    runtime = device_and_flags()
    print(f"Device: {runtime.device.type}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    monet_lookup, monet_cols = load_monet_lookup(args.monet_csv)
    base_dataset = datasets.ImageFolder(args.test_dir, transform=make_eval_transform(args.img_size))
    idx_mel = base_dataset.class_to_idx.get("mel")
    idx_other = base_dataset.class_to_idx.get("other")
    if idx_mel is None or idx_other is None:
        raise ValueError(f"class_to_idx={base_dataset.class_to_idx} must contain 'mel' and 'other'")

    eval_dataset = MonetEvalDataset(base_dataset, idx_mel, monet_lookup, len(monet_cols))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=runtime.pin_memory,
    )

    checkpoint = load_checkpoint_safe(args.checkpoint)
    state_key = "ema_state" if "ema_state" in checkpoint else "model_state"
    model = CoFIDAMonet(num_concepts=len(monet_cols)).to(runtime.device)
    missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
    if missing or unexpected:
        print("State dict info:", {"missing": missing, "unexpected": unexpected})
    model.eval()

    paths, y_true, y_prob = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["img"].to(runtime.device)
            monet = batch["monet"].to(runtime.device)
            logits, _, _, _ = model.forward_eval(images, monet)
            prob = torch.softmax(logits, dim=1)[:, 1]
            paths.extend(batch["path"])
            y_true.extend(batch["y"].cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    print_eval_report(
        "Clinical validation — CoFIDA-MONET inference",
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        out_csv=args.out_csv,
        paths=paths,
    )


if __name__ == "__main__":
    main()
