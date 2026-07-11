import argparse
import os
import random
from contextlib import redirect_stdout
from datetime import datetime
from typing import List

import clip
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

from datasets import build_dataset
from datasets.utils import build_data_loader
from utils.client import Client
from utils.server import Server
from utils.utils import *


def load_cfg(dataset_name: str, config_dir: str = "configs", encoding: str = "utf-8"):
    cfg_path = os.path.join(config_dir, f"{dataset_name}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    SEED = 1
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbones = {"RN": "RN50", "VIT": "ViT-B/16"}
    clip_model, preprocess = clip.load(backbones[args.backbone])
    clip_model.to(torch.float32)
    clip_model.eval()

    datasets = args.datasets.split("/")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{args.backbone}.txt"
    if args.output_subdir:
        output_dir = os.path.join(args.root_path, "output", args.output_subdir, args.method)
    else:
        output_dir = os.path.join(args.root_path, "output", args.method)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
        print("=== ARGUMENTS ===")
        for name, value in vars(args).items():
            print(f"{name}: {value}")

    for dataset_name in datasets:
        with open(output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
            print("\n" + "=" * 40 + "\n")
            print(f"Dataset: {dataset_name}")

        print("Preparing dataset.")
        dataset = build_dataset(dataset_name, args.root_path, args.num_shots)
        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=64,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
        )
        train_tranform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224,
                    scale=(0.5, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=256,
            tfm=train_tranform,
            is_train=True,
            shuffle=False,
        )
        cupl_path = dataset.cupl_path
        clip_weights = clip_classifier(
            dataset.classnames, dataset.template, clip_model, cupl_path, args.gpt3_prompts
        )
        text_features = clip_weights.t()

        text_features = text_features.to(device)

        print("\nLoading visual features and labels from test set.")
        image_features_load, gts = pre_load_features(
            args, dataset_name, "test", clip_model, test_loader
        )
        image_features_load = image_features_load.to(device)
        gts = gts.to(device)

        print("\nLoading visual features and labels from train set.")
        few_shot_image_features, few_shot_labels_1d = pre_load_features(
            args, dataset_name, "train", clip_model, train_loader
        )
        few_shot_image_features = few_shot_image_features.to(device)
        few_shot_labels_1d = few_shot_labels_1d.to(device)

        acc = zero_shot_accuracy(
            image_embeds=image_features_load,
            prototype_embeds=text_features,
            gts=gts,
        )
        print(f"[Zero shot of [{dataset_name}] Accuracy: {acc:.2f}%")
        with open(output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
            print(f"[Zero shot of [{dataset_name}] Accuracy: {acc:.2f}%")

        DISTRIBUTION_SEED = SEED
        if args.partition == "distribution":
            client_indices_original = distribution_label_skew_split_consistency(
                gts,
                args.num_clients,
                args.dirichlet_alpha,
                consistency_seed=DISTRIBUTION_SEED,
            )
        elif args.partition == "uniform":
            all_indices = torch.arange(image_features_load.size(0))
            client_indices_original = torch.chunk(all_indices, args.num_clients)
        else:
            raise ValueError("Unknown partition method")

        if args.partition == "distribution":
            client_indices_fewshot = distribution_label_skew_split_consistency(
                few_shot_labels_1d,
                args.num_clients,
                args.dirichlet_alpha,
                consistency_seed=DISTRIBUTION_SEED,
            )
        elif args.partition == "uniform":
            all_indices = torch.arange(few_shot_image_features.size(0))
            client_indices_fewshot = torch.chunk(all_indices, args.num_clients)

        cfg = load_cfg(dataset_name)
        args.alpha = cfg["positive"]["alpha"]
        args.beta = cfg["positive"]["beta"]
        args.local_lr = cfg["learning_rate"]["image"]
        args.global_lr = cfg["learning_rate"]["text"]
        if args.lam_reg == 10.0:
            args.lam_reg = cfg["learning_rate"]["align"]
        params_to_record = ["alpha", "beta", "local_lr", "global_lr", "lam_reg"]
        with open(output_file, "a", encoding="utf-8") as f, redirect_stdout(f):
            for name in params_to_record:
                print(f"{name}: {getattr(args, name, None)}")
            print("\n" + "=" * 40 + "\n")

        clients: List[Client] = []
        for cid in range(args.num_clients):
            test_idx = client_indices_original[cid]
            few_shot_idx = client_indices_fewshot[cid]
            clients.append(
                Client(
                    client_id=cid,
                    text_features=text_features,
                    image_features=image_features_load[test_idx],
                    gts=gts[test_idx],
                    args=args,
                    few_shot_image_features=few_shot_image_features[few_shot_idx],
                    few_shot_labels=few_shot_labels_1d[few_shot_idx],
                    device=device,
                    output_file=output_file,
                )
            )

        server = Server(args, text_features, clients, output_file)
        server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSPA: Federated Adaptation via Semantic-Visual Prototype Alignment")

    parser.add_argument(
        "--root_path",
        default="/workspace/FedSPA/DATA",
        type=str,
        help="Dataset root directory containing the 10 image classification benchmarks used by FedSPA",
    )
    parser.add_argument(
        "--cache_dir",
        default="./features",
        type=str,
        help="Root directory for pre-extracted CLIP feature caches (one subdir per dataset)",
    )
    parser.add_argument("--num_shots", type=int, default=8, help="Number of few-shot examples per class")
    parser.add_argument(
        "--load_pre_feat", action="store_true", help="Load cached features instead of re-extracting (default: False)"
    )

    parser.add_argument(
        "--datasets", type=str, default="dtd", help='Datasets to process (separated by "/")'
    )
    parser.add_argument(
        "--backbone", type=str, choices=["RN", "VIT"], default="RN", help="CLIP backbone: RN for ResNet-50, VIT for ViT-B/16"
    )
    
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Optional output subdirectory; logs are saved to {root_path}/output/{output_subdir}/{method}",
    )

    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["distribution", "uniform"],
        default="uniform",
        help="Client data partition strategy",
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=0.1, help="Dirichlet alpha for distribution-based partition"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=20, help="Client-side epochs/iterations for updating personalized visual prototypes")
    parser.add_argument(
        "--local_epochs_server", type=int, default=100, 
        help="Server-side epochs for optimizing global semantic prototypes with regularized contrastive alignment")
    parser.add_argument("--global_epochs", type=int, default=20, help="Number of global federated communication rounds")
    parser.add_argument(
        "--local_epochs_last",
        type=int,
        default=0,
        help="Client-side prototype update epochs used in the final round; 0 means use --local_epochs",
    )
    parser.add_argument("--local_batch_size", type=int, default=8, help="Mini-batch size for client-side personalized visual prototype adaptation")
    parser.add_argument("--global_batch_size", type=int, default=8, help="Mini-batch size for server-side semantic prototype alignment")

    parser.add_argument("--alpha", type=float, default=1, help="Tip-Adapter alpha hyperparameter")
    parser.add_argument("--beta", type=float, default=2, help="Tip-Adapter beta hyperparameter")
    parser.add_argument(
        "--lam_reg", type=float, default=10.0, 
        help="Weight of the semantic stability regularizer for preventing global semantic prototype drift")
    parser.add_argument("--local_lr", type=float, default=0.0002, help="Learning rate for client-side personalized visual prototype updates")
    parser.add_argument("--global_lr", type=float, default=0.0002, help="Learning rate for server-side global semantic prototype updates")

    parser.add_argument(
        "--method", type=str, default="FedSPA", help="Algorithm name"
    )

    parser.add_argument("--gpt3_prompts", action="store_true", help="Load GPT-3 CuPL prompts")
    parser.add_argument(
        "--cosine_compute", action="store_true", 
        help="Compute cross-client cosine-distance statistics of visual prototypes for prototype divergence analysis")

    parser.add_argument(
        "--use_dp", action="store_true", 
        help="Enable feature-level local differential privacy by adding Gaussian noise before local prototype construction")
    parser.add_argument(
        "--dp_clip_C", type=float, default=1.0, 
        help="L2 clipping bound for features or prototype rows before DP noise injection")
    parser.add_argument(
        "--dp_sigma", type=float, default=0.1, 
        help="Gaussian noise scale sigma for feature-level local differential privacy")

    args = parser.parse_args()
    args.method = "FedSPA"
    main(args)
