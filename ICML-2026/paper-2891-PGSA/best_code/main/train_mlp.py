"""Step 1: pretrain the auxiliary MLP classifier on the source domain."""
from __future__ import annotations

import json
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models
from args import apply_mlp_defaults, build_parser
from data_loader import load_domain_pair
from psahs.paths import checkpoint_suffix, output_dir
from psahs import training_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(mlp_model, src_data, args, opt, scheduler):
    src_data = src_data.to(device)
    mlp_model.train()
    opt.zero_grad()
    _, logits = mlp_model(src_data)
    mask = src_data.source_training_mask
    loss = utils.ce_loss(logits[mask], src_data.y[mask])
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()


def evaluate(data, model, phase):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            mask = data.target_testing_mask
        _, pred = model(data)
        pred = pred[mask]
        label = data.y[mask]
        acc, auc = utils.classification_scores(pred, label)
    return acc, auc


def train(source_dataset, target_dataset, args, seed):
    path = output_dir(args)
    ck = checkpoint_suffix(args)
    input_dim = source_dataset.num_node_features
    output_dim = source_dataset.num_classes

    mlp_model = models.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    scheduler, opt = utils.build_optimizer(args, mlp_model.parameters())

    best_valid = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(mlp_model, source_dataset, args, opt, scheduler)
        acc_tgt_valid, _ = evaluate(target_dataset, mlp_model, "tgt_valid")
        acc_src_valid, _ = evaluate(source_dataset, mlp_model, "src_valid")
        valid_score = acc_src_valid if args.valid_data == "src" else acc_tgt_valid
        if valid_score > best_valid:
            best_valid = valid_score
            torch.save(
                mlp_model.state_dict(),
                os.path.join(path, f"best_mlp_model_seed{seed}{ck}.pt"),
            )
        if (epoch + 1) % 50 == 0:
            print(f"[seed={seed} epoch={epoch + 1}] valid acc={valid_score:.4f}")

    mlp_model.load_state_dict(
        torch.load(os.path.join(path, f"best_mlp_model_seed{seed}{ck}.pt"))
    )
    report = {}
    for phase in ("src_train", "src_valid", "src_test", "tgt_valid", "tgt_test"):
        acc, auc = evaluate(source_dataset if phase.startswith("src") else target_dataset, mlp_model, phase)
        report[f"acc_{phase}"] = acc
        report[f"auc_{phase}"] = auc if auc is not None else 0.0
    return report


def main():
    parser = build_parser("Pretrain auxiliary MLP for PSAHS")
    args = apply_mlp_defaults(parser.parse_args())
    if args.gpu_ratio is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    run_dir = output_dir(args)
    with utils.redirect_stdout_logger(run_dir):
        print(args)
        reports = []
        for seed in args.seeds:
            utils.set_seed(seed)
            source, target = load_domain_pair(args, adjust_source=False)
            report = train(source, target, args, seed)
            reports.append(report)
            print(f"[Seed {seed}] {json.dumps(report, indent=2)}")

        print(f"[Summary] {json.dumps(utils.aggregate_reports(reports), indent=2)}")


if __name__ == "__main__":
    main()
