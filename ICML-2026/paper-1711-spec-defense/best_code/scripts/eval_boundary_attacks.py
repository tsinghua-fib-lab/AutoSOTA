"""
Evaluate CSR on boundary-case attacks that may preserve low-pass semantics.

This script runs attacks on-the-fly and reports:
  - clean accuracy
  - adversarial accuracy without defense
  - adversarial accuracy with CSR
  - detection rate
  - low-pass anchor accuracy on attacked inputs
  - mean cosine similarity between adv and LPF features
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from csr.config import HF_MODEL_CONFIGS, PathConfig, CSRConfig
from csr.data.dataset import CLIPDataset
from csr.data.collate import default_collate_fn
from csr.attacks import CLIPAttacker, GlobalColorAttack, PatchAttack
from csr.defense import CSRDefense


def parse_args():
    parser = argparse.ArgumentParser(description="Boundary-case attack evaluation for CSR")
    parser.add_argument("--model", type=str, default="CLIP-B-16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["General/ImageNet", "FineGrained/Flowers102", "Domain/EuroSAT"],
    )
    parser.add_argument("--sample_n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attack", type=str, choices=["global_color", "patch", "both"], default="both")

    parser.add_argument("--color_steps", type=int, default=100)
    parser.add_argument("--color_lr", type=float, default=1e-2)
    parser.add_argument("--color_scale_delta", type=float, default=0.15)
    parser.add_argument("--color_bias_delta", type=float, default=0.08)
    parser.add_argument("--color_gamma_delta", type=float, default=0.0)

    parser.add_argument("--patch_size", type=int, default=48)
    parser.add_argument("--patch_steps", type=int, default=200)
    parser.add_argument("--patch_lr", type=float, default=5e-2)

    parser.add_argument("--lpf_radius", type=int, default=40)
    parser.add_argument("--detect_thresh", type=float, default=0.85)
    parser.add_argument("--purify_steps", type=int, default=3)
    parser.add_argument("--filter_type", type=str, default="gaussian")
    return parser.parse_args()


def compute_text_features(model, processor, class_names, device):
    prompts = [f"a photo of a {cls}" for cls in class_names]
    features = []
    with torch.no_grad():
        for start in range(0, len(prompts), 100):
            batch_prompts = prompts[start : start + 100]
            text_inputs = processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**text_inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats)
    return torch.cat(features, dim=0)


@torch.no_grad()
def predict_from_images(model, images, text_features):
    image_features = model.get_image_features(pixel_values=images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features.T
    return logits.argmax(dim=-1)


def evaluate_attack(
    dataloader,
    attacker_model,
    csr_model,
    text_features,
    attack,
):
    total = 0
    clean_correct = 0
    adv_correct = 0
    csr_correct = 0
    lpf_correct = 0
    detected = 0
    cosine_sum = 0.0

    for batch_imgs, batch_labels, _ in tqdm(dataloader, leave=False):
        if batch_imgs is None:
            continue

        batch_imgs = batch_imgs.to(attacker_model.device)
        batch_labels = batch_labels.to(attacker_model.device)

        with torch.no_grad():
            clean_logits = attacker_model(batch_imgs, text_features)
            clean_preds = clean_logits.argmax(dim=-1)
        clean_correct += (clean_preds == batch_labels).sum().item()

        adv_imgs = attack.attack(batch_imgs, batch_labels, text_features)

        with torch.no_grad():
            adv_logits = attacker_model(adv_imgs, text_features)
            adv_preds = adv_logits.argmax(dim=-1)
            adv_correct += (adv_preds == batch_labels).sum().item()

            csr_preds = predict_from_images(csr_model, adv_imgs, text_features)
            csr_correct += (csr_preds == batch_labels).sum().item()
            detected += csr_model.last_detection_mask.sum().item()

            lpf_imgs = csr_model._apply_lpf(adv_imgs)
            lpf_feats = csr_model._get_feats(lpf_imgs)
            adv_feats = csr_model._get_feats(adv_imgs)
            lpf_logits = 100.0 * lpf_feats @ text_features.T
            lpf_preds = lpf_logits.argmax(dim=-1)
            lpf_correct += (lpf_preds == batch_labels).sum().item()
            cosine_sum += (adv_feats * lpf_feats).sum(dim=1).sum().item()

        total += batch_labels.size(0)

    if total == 0:
        return None

    return {
        "samples": total,
        "clean_acc": clean_correct / total,
        "adv_acc": adv_correct / total,
        "csr_acc": csr_correct / total,
        "detection_rate": detected / total,
        "lpf_anchor_acc": lpf_correct / total,
        "mean_adv_lpf_cosine": cosine_sum / total,
    }


def main():
    args = parse_args()
    device = args.device
    paths = PathConfig()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    model_path = HF_MODEL_CONFIGS[args.model]
    attacker_model = CLIPAttacker(model_path, device=device)
    csr_model = CSRDefense(
        model_path,
        config=CSRConfig(
            filter_type=args.filter_type,
            lpf_radius=args.lpf_radius,
            detect_thresh=args.detect_thresh,
            purify_steps=args.purify_steps,
        ),
        device=device,
    )

    attack_builders = {}
    if args.attack in ["global_color", "both"]:
        attack_builders["global_color"] = lambda: GlobalColorAttack(
            attacker_model,
            steps=args.color_steps,
            lr=args.color_lr,
            scale_delta=args.color_scale_delta,
            bias_delta=args.color_bias_delta,
            gamma_delta=args.color_gamma_delta,
        )
    if args.attack in ["patch", "both"]:
        attack_builders["patch"] = lambda: PatchAttack(
            attacker_model,
            patch_size=args.patch_size,
            steps=args.patch_steps,
            lr=args.patch_lr,
        )

    all_results = []
    start_time = time.time()

    for dataset_name in args.datasets:
        csv_path = os.path.join(paths.data_root, dataset_name, "labels.csv")
        img_root = os.path.join(paths.data_root, dataset_name, "images")
        dataset = CLIPDataset(
            csv_path=csv_path,
            img_root=img_root,
            dataset_name=dataset_name,
            sample_n=args.sample_n,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate_fn,
        )

        text_features = attacker_model.precompute_text_features(dataset.all_classes)

        for attack_name, builder in attack_builders.items():
            attack = builder()
            metrics = evaluate_attack(dataloader, attacker_model, csr_model, text_features, attack)
            if metrics is None:
                continue

            row = {
                "dataset": dataset_name,
                "attack": attack_name,
                "model": args.model,
                "device": device,
                "sample_n": args.sample_n,
                "lpf_radius": args.lpf_radius,
                "detect_thresh": args.detect_thresh,
                "purify_steps": args.purify_steps,
                "elapsed_sec": round(time.time() - start_time, 2),
            }
            row.update(metrics)
            all_results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
