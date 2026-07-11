import argparse
import os
import random
import warnings

import numpy as np
import torch

from dataprocessing import MultiviewData
from layers import MAGICNetwork
from loss import SemanticAlignmentLoss
from models import pre_train, contrastive_train, valid

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAGIC for incomplete multi-view clustering"
    )

    parser.add_argument(
        "--db",
        type=str,
        default="BDGP",
        choices=["MNIST-USPS", "Fashion", "BDGP", "FMNIST"],
        help="Dataset name."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device index."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed."
    )
    parser.add_argument(
        "--mse_epochs",
        type=int,
        default=None,
        help="Number of reconstruction pretraining epochs."
    )
    parser.add_argument(
        "--con_epochs",
        type=int,
        default=None,
        help="Number of contrastive fine-tuning epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size."
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Weight for semantic alignment loss."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Weight for multi-path consensus loss."
    )
    parser.add_argument(
        "--js_weight",
        type=float,
        default=0.1,
        help="Weight of the JS term in semantic alignment loss."
    )
    parser.add_argument(
        "--load_model",
        default=False,
        action="store_true",
        help="Load a saved model."
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action="store_true",
        help="Save the trained model."
    )
    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=None,
        help="Missing ratio for BDGP dataset (0.1, 0.3, 0.5, 0.7)."
    )
    # Paper hyperparameters
    parser.add_argument(
        "--impute_th_start",
        type=float,
        default=None,
        help="Initial confidence threshold for semantic imputation."
    )
    parser.add_argument(
        "--impute_th_end",
        type=float,
        default=None,
        help="Final confidence threshold for semantic imputation."
    )
    parser.add_argument(
        "--inf_temp",
        type=float,
        default=None,
        help="Aggregation temperature for consensus loss."
    )
    parser.add_argument(
        "--inf_lambda_fuse",
        type=float,
        default=None,
        help="Weight for fused path in consensus loss."
    )
    parser.add_argument(
        "--inf_lambda_uni",
        type=float,
        default=None,
        help="Weight for per-view path in consensus loss."
    )
    parser.add_argument(
        "--inf_lambda_mask",
        type=float,
        default=None,
        help="Weight for masked-fusion path in consensus loss."
    )
    parser.add_argument(
        "--epoch_open_ot",
        type=int,
        default=None,
        help="Epoch to start OT-based imputation."
    )
    parser.add_argument(
        "--epoch_open_knn",
        type=int,
        default=None,
        help="Epoch to start kNN-based feature imputation."
    )
    parser.add_argument(
        "--inf_lambda_cons",
        type=float,
        default=None,
        help="Weight for KL consistency term in consensus loss."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of independent runs with different seeds."
    )

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_config(db):
    if db == "MNIST-USPS":
        return {
            "learning_rate": 1e-4,
            "batch_size": 50,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 25,
            "temperature_l": 0.7,
            "normalized": False,
            "dim_high_feature": 1500,
            "dim_low_feature": 1024,
            "dims": [256, 512, 1024],
            "beta": 0.01,
            "gamma": 1.0,
        }

    if db == "Fashion":
        return {
            "learning_rate": 1e-4,
            "batch_size": 100,
            "seed": 20,
            "mse_epochs": 100,
            "con_epochs": 20,
            "temperature_l": 0.5,
            "normalized": True,
            "dim_high_feature": 2000,
            "dim_low_feature": 500,
            "dims": [256, 512],
            "beta": 0.01,
            "gamma": 1.0,
        }

    if db == "BDGP":
        return {
            "learning_rate": 1e-4,
            "batch_size": 200,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 100,
            "temperature_l": 0.7,
            "normalized": True,
            "dim_high_feature": 2000,
            "dim_low_feature": 1024,
            "dims": [256, 512],
            "beta": 0.01,
            "gamma": 1.0,
            # Paper hyperparameters (Appendix C Table 4)
            "impute_th_start": 0.65,
            "impute_th_end": 0.55,
            "epoch_open_ot": 30,
            "epoch_open_knn": 50,
            "inf_temp": 0.8,
            "inf_lambda_fuse": 1.0 / 3.0,
            "inf_lambda_uni": 1.0 / 6.0,
            "inf_lambda_mask": 1.0 / 2.0,
            "inf_lambda_cons": 0.05,
        }

    if db == "FMNIST":
        return {
            "learning_rate": 1e-4,
            "batch_size": 200,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 100,
            "temperature_l": 0.7,
            "normalized": True,
            "dim_high_feature": 1024,
            "dim_low_feature": 1024,
            "dims": [256, 512, 1024],
            "beta": 0.01,
            "gamma": 1.0,
        }

    raise ValueError(f"Unsupported dataset: {db}")


def apply_overrides(config, args):
    for key in ["seed", "mse_epochs", "con_epochs", "batch_size", "learning_rate", "beta", "gamma"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ["impute_th_start", "impute_th_end", "inf_temp",
                "inf_lambda_fuse", "inf_lambda_uni", "inf_lambda_mask",
                "epoch_open_ot", "epoch_open_knn", "inf_lambda_cons"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def run_single(config, args, device):
    set_seed(config["seed"])

    print("==========")
    print("Method: MAGIC")
    print(f"Dataset: {args.db}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Pretraining epochs: {config['mse_epochs']}")
    print(f"Fine-tuning epochs: {config['con_epochs']}")
    if args.missing_ratio is not None:
        print(f"Missing ratio: {args.missing_ratio}")
    print(f"Impute threshold: {config.get('impute_th_start', 'default')} -> {config.get('impute_th_end', 'default')}")
    print(f"Aggregation temperature: {config.get('inf_temp', 'default')}")
    print(f"Path weights: fuse={config.get('inf_lambda_fuse', 'default')}, "
          f"uni={config.get('inf_lambda_uni', 'default')}, "
          f"mask={config.get('inf_lambda_mask', 'default')}")
    print("==========")

    mv_data = MultiviewData(args.db, device, missing_ratio=args.missing_ratio)

    num_views = len(mv_data.data_views)
    num_clusters = np.unique(mv_data.labels).size
    input_sizes = np.array([view.shape[1] for view in mv_data.data_views], dtype=int)

    model = MAGICNetwork(
        num_views=num_views,
        input_sizes=input_sizes,
        dims=config["dims"],
        dim_high_feature=config["dim_high_feature"],
        dim_low_feature=config["dim_low_feature"],
        num_clusters=num_clusters,
    ).to(device)

    alignment_loss = SemanticAlignmentLoss(
        config["batch_size"],
        num_clusters,
        js_weight=args.js_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=args.weight_decay,
    )

    model_path = f"./models/MAGIC_pytorch_model_{args.db}.pth"

    if args.load_model:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        pre_train(
            model,
            mv_data,
            config["batch_size"],
            config["mse_epochs"],
            optimizer,
        )

        best_acc = 0.0
        best_nmi = 0.0
        best_ari = 0.0
        best_epoch = 0

        for epoch in range(config["con_epochs"]):
            model.set_aug_strength(epoch)

            _ = contrastive_train(
                model,
                mv_data,
                alignment_loss,
                config["batch_size"],
                config["beta"],
                config["gamma"],
                config["temperature_l"],
                config["normalized"],
                epoch,
                optimizer,
                epoch_open_ot=config.get("epoch_open_ot", 30),
                epoch_open_knn=config.get("epoch_open_knn", 50),
                impute_th_start=config.get("impute_th_start", 0.55),
                impute_th_end=config.get("impute_th_end", 0.55),
                inf_temp=config.get("inf_temp", 0.2),
                inf_lambda_fuse=config.get("inf_lambda_fuse", 1.0),
                inf_lambda_uni=config.get("inf_lambda_uni", 1.0),
                inf_lambda_mask=config.get("inf_lambda_mask", 1.0),
                inf_lambda_cons=config.get("inf_lambda_cons", 0.05),
            )

            # Track best metrics
            if (epoch + 1) % 10 == 0:
                acc, nmi, pur, ari = valid(model, mv_data, config["batch_size"])
                if acc > best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_epoch = epoch + 1

        # Final evaluation
        print("\n========== FINAL RESULTS ==========")
        final_acc, final_nmi, final_pur, final_ari = valid(
            model, mv_data, config["batch_size"]
        )
        print(f"Best (epoch {best_epoch}): ACC={best_acc:.4f} NMI={best_nmi:.4f} ARI={best_ari:.4f}")
        print(f"Final (epoch {config['con_epochs']}): ACC={final_acc:.4f} NMI={final_nmi:.4f} ARI={final_ari:.4f}")
        print("==================================\n")

        if args.save_model:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    print("Done.")
    return best_acc, best_nmi, best_ari


def main():
    args = parse_args()
    config = apply_overrides(dataset_config(args.db), args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_acc = []
    all_nmi = []
    all_ari = []

    for run_idx in range(args.n_runs):
        run_seed = config["seed"] + run_idx * 10
        config_run = config.copy()
        config_run["seed"] = run_seed
        print(f"\n{'='*50}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'='*50}")
        acc, nmi, ari = run_single(config_run, args, device)
        all_acc.append(acc)
        all_nmi.append(nmi)
        all_ari.append(ari)

    if args.n_runs > 1:
        print(f"\n{'='*50}")
        print(f"FINAL SUMMARY ({args.n_runs} runs)")
        print(f"ACC: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
        print(f"NMI: {np.mean(all_nmi):.4f} ± {np.std(all_nmi):.4f}")
        print(f"ARI: {np.mean(all_ari):.4f} ± {np.std(all_ari):.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
