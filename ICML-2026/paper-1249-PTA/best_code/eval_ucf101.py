"""
Evaluation script for PTA on UCF101.
Reproduces Table 1 result: 73.12% Top-1 Accuracy.
Follows paper protocol: 3 runs with different seeds, averaged.
Hyperparameters: h=20 (T), w=0.01 (alpha), ViT-B/16 backbone, batch_size=1.
"""
import random, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils import get_clip_logits, cls_acc, get_config_file, clip_classifier
import clip
from pta_runner import update_text_features
from datasets.utils import DatasetWrapper
from datasets.ucf101 import UCF101


def evaluate_pta_single_run(cfg, loader, clip_model, clip_weights):
    """Run PTA evaluation on a single data stream."""
    with torch.no_grad():
        accuracies = []
        refine_feature = clip_weights.t()
        target_prototype = torch.zeros_like(refine_feature).cuda()
        alpha, T_val = cfg["alpha"], cfg["T"]

        for i, (images, target) in enumerate(tqdm(loader, desc="Evaluating")):
            image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()
            soft_logits = F.softmax(clip_logits, dim=-1)
            refine_feature, target_prototype = update_text_features(
                image_features, soft_logits.half(), refine_feature, target_prototype,
                alpha=alpha, T=T_val)
            final_logits = clip_logits.clone()
            final_logits += 100. * image_features.half() @ refine_feature.half().T
            accuracies.append(cls_acc(final_logits, target))

        return sum(accuracies) / len(accuracies)


def main():
    data_root = "/datasets"
    seeds = [1, 2, 3]
    results = []

    # Load CLIP model once (shared across seeds)
    print("Loading CLIP ViT-B/16 model...")
    clip_model, preprocess = clip.load("ViT-B/16")
    clip_model.eval()

    cfg = get_config_file("configs", "ucf101")
    print("Config: alpha=%.4f, T=%.1f" % (cfg["alpha"], cfg["T"]))

    # Pre-build text classifier (same for all seeds)
    dataset = UCF101(data_root)
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    print("Dataset: %d classes, %d test samples" % (dataset.num_classes, len(dataset.test)))

    for seed in seeds:
        print("\n--- Run with seed %d ---" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Build test loader with deterministic settings
        test_loader = torch.utils.data.DataLoader(
            DatasetWrapper(dataset.test, input_size=224, transform=preprocess, is_train=False),
            batch_size=1, num_workers=0, shuffle=True, drop_last=False, pin_memory=True)

        acc = evaluate_pta_single_run(cfg, test_loader, clip_model, clip_weights)
        results.append(acc)
        print("Seed %d accuracy: %.2f%%" % (seed, acc))

    avg_acc = np.mean(results)
    std_acc = np.std(results, ddof=1) if len(results) > 1 else 0.0

    print("\n" + "=" * 50)
    print("PTA on UCF101 (ViT-B/16, h=20, w=0.01)")
    print("Individual runs: %s" % ", ".join("%.2f%%" % r for r in results))
    print("Mean accuracy: %.2f%%" % avg_acc)
    print("Std deviation: %.2f%%" % std_acc)
    print("Paper reported: 73.12%% +/- 0.11%%")
    print("=" * 50)

    # Save result
    with open("outputs/result.txt", "w") as f:
        f.write("PTAs performance on ucf101: Top1- %.2f.\n" % avg_acc)
        f.write("Individual seeds: %s\n" % ", ".join("%.2f" % r for r in results))

    return avg_acc


if __name__ == "__main__":
    main()
