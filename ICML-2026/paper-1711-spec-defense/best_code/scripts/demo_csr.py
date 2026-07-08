"""
Quick demo of CSR defense on a CLIP model.

Usage:
    python scripts/demo_csr.py --image_path /path/to/image.png --device cuda:0
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms

from csr.config import CSRConfig, HF_MODEL_CONFIGS
from csr.defense import CSRDefense, FastCSRDefense


def parse_args():
    parser = argparse.ArgumentParser(description="CSR Defense Demo")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="CLIP-L-14")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fast", action="store_true", help="Use FastCSR")
    parser.add_argument("--labels", type=str, nargs="+", default=["cat", "dog", "bird", "car", "airplane"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image
    image = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Load CSR defense
    model_path = HF_MODEL_CONFIGS[args.model]
    DefClass = FastCSRDefense if args.fast else CSRDefense
    csr = DefClass(model_path, device=args.device)

    # Detect
    is_adv = csr.detect(img_tensor)
    print(f"Adversarial detected: {is_adv.item()}")

    # Zero-shot predict
    probs, logits, attack_mask = csr.predict_zero_shot(img_tensor, args.labels)

    print(f"\nZero-Shot Classification Results:")
    print(f"{'Label':<15} {'Probability':>10}")
    print("-" * 28)
    for label, prob in zip(args.labels, probs[0]):
        print(f"{label:<15} {prob.item()*100:>8.2f}%")

    print(f"\nAttack detected: {attack_mask.item()}")


if __name__ == "__main__":
    main()
