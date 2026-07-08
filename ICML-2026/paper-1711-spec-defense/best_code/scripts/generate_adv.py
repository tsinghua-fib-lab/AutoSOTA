"""
Generate adversarial samples for CLIP models.

Usage:
    python scripts/generate_adv.py --attack pgd --model CLIP-L-14 --epsilon 4 --device cuda:0
    python scripts/generate_adv.py --attack cw --model CLIP-L-14 --epsilon 4 --device cuda:1
    python scripts/generate_adv.py --attack apgd --model CLIP-L-14 --epsilon 4 --device cuda:2
    python scripts/generate_adv.py --attack adaptive_pgd --model CLIP-B-16 --epsilon 1 --steps 10 --device cuda:0
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from csr.config import PathConfig, AttackConfig, EvalConfig, HF_MODEL_CONFIGS
from csr.data.dataset import CLIPDataset
from csr.data.collate import default_collate_fn
from csr.attacks import (
    CLIPAttacker,
    PGDAttack,
    AdaptivePGDAttack,
    CWAttack,
    APGDAttack,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial samples")
    parser.add_argument(
        "--attack",
        type=str,
        default="pgd",
        choices=["pgd", "adaptive_pgd", "cw", "apgd"],
    )
    parser.add_argument("--model", type=str, default="CLIP-L-14")
    parser.add_argument("--epsilon", type=int, default=4, help="Epsilon in /255 units")
    parser.add_argument("--steps", type=int, default=None, help="Override attack steps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--freq_radius", type=int, default=None, help="Low/high-frequency cutoff radius for adaptive PGD")
    parser.add_argument("--hf_reg_weight", type=float, default=None, help="Weight for adaptive PGD high-frequency penalty")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = PathConfig()
    atk_cfg = AttackConfig()
    eval_cfg = EvalConfig()

    epsilon = args.epsilon / 255
    model_name = args.model
    device = args.device

    # Output directory
    attack_dir_name = {
        "pgd": "PGD",
        "adaptive_pgd": "Adaptive_PGD",
        "cw": "CW",
        "apgd": "APGD",
    }[args.attack]
    save_root = args.save_root or os.path.join(paths.adv_save_root, attack_dir_name)
    datasets_list = args.datasets or eval_cfg.datasets

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load attacker
    model_path = HF_MODEL_CONFIGS[model_name]
    attacker = CLIPAttacker(model_path, device)

    # Create attack
    eps_str = f"{args.epsilon}_255"
    if args.attack == "pgd":
        steps = args.steps or atk_cfg.pgd_steps
        attack = PGDAttack(attacker, epsilon=epsilon, steps=steps, alpha_scale=atk_cfg.pgd_alpha_scale)
    elif args.attack == "adaptive_pgd":
        steps = args.steps or atk_cfg.pgd_steps
        attack = AdaptivePGDAttack(
            attacker,
            epsilon=epsilon,
            steps=steps,
            alpha_scale=atk_cfg.pgd_alpha_scale,
            freq_radius=args.freq_radius or atk_cfg.adaptive_freq_radius,
            hf_reg_weight=args.hf_reg_weight or atk_cfg.adaptive_hf_reg_weight,
        )
    elif args.attack == "cw":
        steps = args.steps or atk_cfg.cw_steps
        attack = CWAttack(attacker, epsilon=epsilon, steps=steps, kappa=atk_cfg.cw_kappa)
    elif args.attack == "apgd":
        steps = args.steps or atk_cfg.apgd_steps
        attack = APGDAttack(attacker, epsilon=epsilon, steps=steps)

    print(f"Attack: {attack_dir_name}, Model: {model_name}, Eps: {eps_str}, Steps: {steps}")
    if args.attack == "adaptive_pgd":
        print(
            "Adaptive settings: "
            f"freq_radius={attack.freq_radius}, hf_reg_weight={attack.hf_reg_weight}, alpha={attack.alpha:.6f}"
        )

    for dataset_name in datasets_list:
        print(f"\n>> Dataset: {dataset_name}")

        csv_path = os.path.join(paths.data_root, dataset_name, "labels.csv")
        img_root = os.path.join(paths.data_root, dataset_name, "images")

        try:
            ds = CLIPDataset(
                csv_path=csv_path,
                img_root=img_root,
                dataset_name=dataset_name,
                sample_n=args.sample_n,
                transform=transform,
            )
        except Exception as e:
            print(f"  [Error] {e}")
            continue

        text_features = attacker.precompute_text_features(ds.all_classes)
        dataloader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=default_collate_fn,
        )

        save_dir = os.path.join(save_root, model_name, dataset_name, eps_str)
        os.makedirs(save_dir, exist_ok=True)

        for batch_imgs, batch_labels, batch_fnames in tqdm(dataloader, desc=f"{dataset_name}"):
            if batch_imgs is None:
                continue

            adv_imgs = attack.attack(batch_imgs, batch_labels, text_features)

            for i, img_tensor in enumerate(adv_imgs):
                fname = batch_fnames[i]
                save_name = os.path.splitext(fname)[0] + ".png"
                save_path = os.path.join(save_dir, save_name)
                transforms.ToPILImage()(img_tensor.cpu()).save(save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
