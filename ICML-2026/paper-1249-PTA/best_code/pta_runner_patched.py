# Patched PTA runner with deterministic seed behavior
import random
import argparse
from tqdm import tqdm
from datetime import datetime

# Set seeds BEFORE any other imports for determinism
random.seed(1)
import torch
torch.manual_seed(1)

import torch.nn.functional as F
import operator

import clip
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", required=True, help="settings of C-TTA on specific dataset in yaml format.")
    parser.add_argument("--wandb-log", dest="wandb", action="store_true", help="Whether you want to log to wandb.")
    parser.add_argument("--datasets", dest="datasets", type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data", help="Path to the datasets directory. Default is ./dataset/")
    parser.add_argument("--backbone", dest="backbone", type=str, choices=["RN50", "ViT-B/16"], required=True, help="CLIP model backbone to use: RN50 or ViT-B/16.")
    args = parser.parse_args()
    return args

def update_text_features(image_feature, probs, text_features, target_prototype, alpha=0.01, T=20,
                         update_counts=None, base_threshold=0.1):
    w = probs.squeeze(0)  # [C]
    w_new = torch.zeros_like(w)

    # Per-class adaptive threshold based on update frequency
    if update_counts is not None:
        total_updates = update_counts.sum().item()
        num_classes = len(update_counts)
        if total_updates > num_classes:
            avg_updates = total_updates / num_classes
            ratio = update_counts.float() / max(avg_updates, 1.0)
            ratio_clamped = torch.clamp(ratio, 0.5, 2.0)
            adaptive_threshold = base_threshold * ratio_clamped
        else:
            adaptive_threshold = torch.full_like(w, base_threshold)
        mask = w >= adaptive_threshold
    else:
        mask = w >= base_threshold

    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)  # [C, 1]
    target_prototype[mask] = (1 - w_new[mask]) * target_prototype[mask] + w_new[mask] * image_feature.squeeze(0)

    if update_counts is not None:
        update_counts[mask] += 1

    refined_text = alpha * text_features + (1 - alpha) * target_prototype
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, target_prototype, update_counts

def PTA(cfg, loader, clip_model, clip_weights, dataset_name):
    with torch.no_grad():
        accuracies = []
        refine_feature = clip_weights.t()  # [C, D]
        target_prototype = torch.zeros_like(refine_feature).cuda()
        params = {k: cfg[k] for k in ["alpha", "T"]}
        update_counts = torch.zeros(refine_feature.shape[0]).cuda()
        # Prototype repulsion parameters
        repulsion_interval = cfg.get("repulsion_interval", 10)
        repulsion_threshold = cfg.get("repulsion_threshold", 0.7)
        repulsion_strength = cfg.get("repulsion_strength", 0.001)

        for i, (images, target) in enumerate(tqdm(loader, desc="Processed test images: ")):
            image_features, clip_logits, _, _, _ = get_clip_logits(images ,clip_model, clip_weights)
            target = target.cuda()
            soft_logits = F.softmax(clip_logits, dim = -1)
            refine_feature, target_prototype, update_counts = update_text_features(
                image_features, soft_logits.half(), refine_feature, target_prototype,
                alpha=params["alpha"], T=params["T"], update_counts=update_counts
            )

            # Prototype repulsion: push apart similar class prototypes
            if i > 0 and i % repulsion_interval == 0:
                sim = refine_feature @ refine_feature.T  # [C, C] cosine similarity
                sim.fill_diagonal_(-1.0)
                top_sim, top_idx = sim.max(dim=1)  # [C], [C]
                mask = top_sim > repulsion_threshold
                if mask.any():
                    repel_indices = torch.where(mask)[0]
                    for c in repel_indices:
                        c_item = c.item()
                        other = top_idx[c_item].item()
                        if c_item < other:  # process each pair once
                            direction = refine_feature[c_item] - refine_feature[other]
                            direction = direction / (direction.norm() + 1e-8)
                            refine_feature[c_item] += repulsion_strength * direction
                            refine_feature[other] -= repulsion_strength * direction
                    refine_feature = refine_feature / refine_feature.norm(dim=-1, keepdim=True)

            final_logits = clip_logits.clone()
            final_logits += 100. * image_features.half() @ refine_feature.half().T
            acc = cls_acc(final_logits, target)
            accuracies.append(acc)
            if i%1000==0:
                print("---- PTAs test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))

        print("---- PTAs test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        with open("outputs/result.txt", "a") as f:
            f.write("PTAs performance on {}: Top1- {:.2f}.\n".format(dataset_name, sum(accuracies)/len(accuracies)))
        return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    datasets = args.datasets.split("/")
    for dataset_name in datasets:
        print("Processing {} dataset.".format(dataset_name))
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)
        acc = PTA(cfg, test_loader, clip_model, clip_weights, dataset_name)


if __name__ == "__main__":
    main()
