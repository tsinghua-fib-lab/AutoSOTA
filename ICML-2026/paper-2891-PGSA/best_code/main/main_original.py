"""Step 2: train PSAHS (progressive structure adjustment + domain-adversarial GNN)."""
from __future__ import annotations

import json
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models
from args import apply_psahs_defaults, build_parser
from data_loader import load_domain_pair
from psahs.edge_stats import average_rw_stats, format_edge_statistics
from psahs.paths import checkpoint_suffix, output_dir
from psahs import training_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, mlp_model, src_data, tgt_data, args, opt, scheduler, epoch):
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    mlp_model.eval()
    with torch.no_grad():
        _, tgt_mlp_pred = mlp_model(tgt_data)

    _, mlp_pred_tgt = tgt_mlp_pred.max(dim=1)
    if args.reweight and epoch >= (args.start_epoch - 1):
        _, [pred_tgt, _] = model.forward(tgt_data, 1)
        _, pred_tgt = pred_tgt.max(dim=1)
        mask = pred_tgt == mlp_pred_tgt
        tgt_data.y_hat[mask] = mlp_pred_tgt[mask]
        if (epoch - 1) % args.rw_freq == 0:
            from psahs.data import datasets

            datasets.adjust_graph_structure_fast_target_Plabel(
                tgt_data, h_thresh=args.h_threshold
            )
            tgt_data = tgt_data.to(device)

    da_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)
    [_, _], [pred_src, pred_domain_src] = model.forward(src_data, da_alpha)
    [_, _], [_, pred_domain_tgt] = model.forward(tgt_data, da_alpha)

    mask_src = src_data.source_training_mask
    label_src = src_data.y[mask_src]
    pred_src = pred_src[mask_src]
    pred_domain_src = pred_domain_src[src_data.source_mask]
    pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]

    cls_loss = utils.ce_loss(pred_src, label_src)
    domain_loss = utils.bce_loss(pred_domain_src, torch.zeros_like(pred_domain_src))
    domain_loss += utils.bce_loss(pred_domain_tgt, torch.ones_like(pred_domain_tgt))
    loss = cls_loss + domain_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()


def evaluate(source_dataset, target_dataset, model):
    phases = [
        ("src_train", source_dataset),
        ("src_valid", source_dataset),
        ("src_test", source_dataset),
        ("tgt_valid", target_dataset),
        ("tgt_test", target_dataset),
    ]
    report = {}
    for phase, data in phases:
        data = data.to(device)
        model.eval()
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
            [_, _], [pred, pred_domain] = model.forward(data, 1)
            pred = pred[mask]
            label = data.y[mask]
            if "src" in phase:
                domain_pred = pred_domain[data.source_mask]
                domain_label = torch.zeros_like(domain_pred)
            else:
                domain_pred = pred_domain[data.target_mask]
                domain_label = torch.ones_like(domain_pred)
            acc, auc = utils.classification_scores(pred, label)
            report[f"acc_{phase}"] = acc
            report[f"auc_{phase}"] = auc if auc is not None else 0.0
            report[f"loss_{phase}"] = utils.ce_loss(pred, label).item()
            report[f"dc_loss_{phase.split('_')[0]}"] = utils.bce_loss(domain_pred, domain_label).item()
    return report


def train(source_dataset, target_dataset, args, seed):
    path = output_dir(args)
    ck = checkpoint_suffix(args)
    input_dim = source_dataset.num_node_features
    output_dim = source_dataset.num_classes

    model = models.directed_GNN_adv(input_dim, output_dim, args).to(device)
    mlp_model = models.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    mlp_ckpt = os.path.join(path, f"best_mlp_model_seed{seed}{ck}.pt")
    if not os.path.isfile(mlp_ckpt):
        raise FileNotFoundError(
            f"Missing MLP checkpoint: {mlp_ckpt}\n"
            f"Run first: python main/train_mlp.py -d {args.dataset} "
            f"--src_name {args.src_name} --tgt_name {args.tgt_name}"
        )
    mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    best_valid = 0.0
    if args.reweight:
        target_dataset.rw_stats_history = []
    elif hasattr(target_dataset, "rw_stats_history"):
        delattr(target_dataset, "rw_stats_history")

    for epoch in range(args.epochs):
        model.train()
        train_one_epoch(model, mlp_model, source_dataset, target_dataset, args, opt, scheduler, epoch)
        report = evaluate(source_dataset, target_dataset, model)
        valid_score = report["acc_src_valid"] if args.valid_data == "src" else report["acc_tgt_valid"]
        if valid_score > best_valid:
            best_valid = valid_score
            torch.save(
                model.state_dict(),
                os.path.join(path, f"best_valid_model_{seed}{ck}.pt"),
            )
        if (epoch + 1) % 50 == 0:
            print(
                f"[seed={seed} epoch={epoch + 1}] "
                f"tgt_test={report['acc_tgt_test']:.4f} tgt_val={report['acc_tgt_valid']:.4f}"
            )

    if getattr(args, "best_model", True):
        model = models.directed_GNN_adv(input_dim, output_dim, args).to(device)
        model.load_state_dict(torch.load(os.path.join(path, f"best_valid_model_{seed}{ck}.pt")))
    final_report = evaluate(source_dataset, target_dataset, model)

    rw_hist = getattr(target_dataset, "rw_stats_history", None)
    if rw_hist:
        avg_stats = average_rw_stats(rw_hist)
        print(format_edge_statistics(avg_stats, averaged_over_steps=avg_stats["n_rw_steps_averaged"]))

    return final_report


def main():
    parser = build_parser("Train PSAHS for graph domain adaptation")
    args = apply_psahs_defaults(parser.parse_args())
    if args.gpu_ratio is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    run_dir = output_dir(args)
    with utils.redirect_stdout_logger(run_dir):
        print(args)
        reports = []
        for seed in args.seeds:
            utils.set_seed(seed)
            source, target = load_domain_pair(args, adjust_source=True)
            report = train(source, target, args, seed)
            reports.append(report)
            print(f"[Seed {seed}] {json.dumps(report, indent=2)}")

        print(f"[Summary] {json.dumps(utils.aggregate_reports(reports), indent=2)}")


if __name__ == "__main__":
    main()
