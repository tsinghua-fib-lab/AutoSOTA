import argparse
from pathlib import Path
import sys

import importlib
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "cs_classifier" not in sys.modules:
    sys.modules["cs_classifier"] = importlib.import_module("logdiff.cs_classifier")

if "datasets" not in sys.modules:
    sys.modules["datasets"] = importlib.import_module("logdiff.datasets")


def _load_judge_model(config_path, checkpoint_override, device):
    cfg = OmegaConf.load(config_path)

    if "judge_classifier" in cfg:
        model_cfg = cfg.judge_classifier
        ckpt_path = cfg.judge_classifier_checkpoint
    elif "trainer" in cfg and "model" in cfg.trainer:
        model_cfg = cfg.trainer.model
        ckpt_path = checkpoint_override
        if ckpt_path is None:
            raise ValueError("--checkpoint is required when config has no judge_classifier_checkpoint")
    else:
        raise ValueError("Config must define judge_classifier or trainer.model")

    model = instantiate(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_dataset(dataset_config_path, split):
    cfg = OmegaConf.load(dataset_config_path)
    if split == "train":
        dataset_cfg = cfg.train_dataset
    elif split == "val":
        dataset_cfg = cfg.val_dataset
    else:
        raise ValueError("split must be 'train' or 'val'")
    return instantiate(dataset_cfg)


def _init_label_stats(num_classes):
    return {
        "total": 0,
        "correct": 0,
        "per_class_total": [0 for _ in range(num_classes)],
        "per_class_correct": [0 for _ in range(num_classes)],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate CelebA judge classifier on pixel dataset.")
    parser.add_argument(
        "--judge-config",
        default="configs/celeba_inference.yaml",
        help="Config with judge_classifier and judge_classifier_checkpoint.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset/celeba_pixel_male_haircolors.yaml",
        help="Dataset config with train/val datasets.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override judge checkpoint path.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--label-names",
        nargs="+",
        default=["Gender", "Hair"],
        help="Label names in order of classifier heads.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = _load_judge_model(args.judge_config, args.checkpoint, device)
    dataset = _load_dataset(args.dataset_config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    num_heads = len(model.num_classes_per_label)
    if len(args.label_names) != num_heads:
        raise ValueError("label-names length must match number of classifier heads")

    stats = [
        _init_label_stats(num_classes)
        for num_classes in model.num_classes_per_label
    ]

    joint_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            y = batch["label"].to(device)
            logits_list = model(x)

            preds = []
            for i in range(num_heads):
                pred_i = torch.argmax(logits_list[i], dim=1)
                preds.append(pred_i)

                y_i = y[:, i]
                correct_i = (pred_i == y_i)

                stats[i]["total"] += y_i.numel()
                stats[i]["correct"] += correct_i.sum().item()

                for cls in range(model.num_classes_per_label[i]):
                    cls_mask = (y_i == cls)
                    stats[i]["per_class_total"][cls] += cls_mask.sum().item()
                    stats[i]["per_class_correct"][cls] += (correct_i & cls_mask).sum().item()

            joint_correct += torch.stack(preds, dim=1).eq(y).all(dim=1).sum().item()
            total_samples += y.size(0)

    print("Judge validation results")
    print(f"Split: {args.split}")
    print(f"Total samples: {total_samples}")
    print(f"Joint accuracy (all labels correct): {joint_correct / max(total_samples, 1):.4f}")

    for name, label_stats in zip(args.label_names, stats):
        total = max(label_stats["total"], 1)
        acc = label_stats["correct"] / total
        print("")
        print(f"{name} accuracy: {acc:.4f}")
        print(f"{name} per-class accuracy:")
        for cls_idx, (cls_total, cls_correct) in enumerate(
            zip(label_stats["per_class_total"], label_stats["per_class_correct"])
        ):
            cls_acc = cls_correct / max(cls_total, 1)
            print(f"  class {cls_idx}: {cls_acc:.4f} (n={cls_total})")


if __name__ == "__main__":
    main()
