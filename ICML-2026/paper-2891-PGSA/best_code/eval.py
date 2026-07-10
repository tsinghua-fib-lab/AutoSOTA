#!/usr/bin/env python3
"""Evaluation script for PSAHS on DBLP-ACM (ACMv9->DBLPv8) with paper settings."""
import os, sys, json
_REPO = "/repo"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "main"))

import torch
import numpy as np
from args import build_parser
from psahs.paths import output_dir, checkpoint_suffix
from psahs import training_utils as utils
from psahs.data import datasets
import models as models_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = build_parser("PSAHS eval")
    args = parser.parse_args()
    
    args.epochs = 300
    args.h_threshold = 1.0
    args.start_epoch = 200
    args.rw_freq = 15
    args.K = 2
    args.hidden_dim = 128
    args.mlp_conv_dim = 128
    args.lr = 0.005
    args.reweight = "rw" in (args.method or "DANN_rw")
    args.opt_decay_rate = 0.9   # Slower decay
    args.opt_decay_step = 50
    args.conv_dim = 128
    args.cls_dim = 64
    args.alphamin = 1.0
    args.alphatimes = 1.5
    args.dropout = 0.3
    args.weight_decay = 5e-4
    
    data_root = os.path.join(_REPO, "dataset")
    run_dir = output_dir(args)
    ck = checkpoint_suffix(args)
    
    utils.set_seed(42)
    source_mlp = datasets.prepare_dblp_acm(data_root, args.src_name)
    target_mlp = datasets.prepare_dblp_acm(data_root, args.tgt_name)
    datasets.adjust_graph_structure_fast_source(source_mlp, h_thresh=args.h_threshold)
    
    input_dim = source_mlp.num_node_features
    output_dim = source_mlp.num_classes
    
    mlp_shared = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    args.lr, mlp_saved_lr = 0.007, args.lr
    args.epochs, mlp_saved_epochs = 300, args.epochs
    args.weight_decay, mlp_saved_wd = 5e-4, args.weight_decay
    mlp_sched, mlp_opt = utils.build_optimizer(args, mlp_shared.parameters())
    
    best_mlp_shared = 0.0
    for epoch in range(300):
        mlp_shared.train()
        s_data = source_mlp.to(device)
        t_data = target_mlp.to(device)
        mlp_opt.zero_grad()
        _, logits = mlp_shared(s_data)
        loss = utils.ce_loss(logits[s_data.source_training_mask], s_data.y[s_data.source_training_mask])
        loss.backward()
        mlp_opt.step()
        mlp_sched.step()
        mlp_shared.eval()
        with torch.no_grad():
            _, tgt_logits = mlp_shared(t_data)
            tgt_val_acc, _ = utils.classification_scores(
                tgt_logits[target_mlp.target_validation_mask], target_mlp.y[target_mlp.target_validation_mask])
        if tgt_val_acc > best_mlp_shared:
            best_mlp_shared = tgt_val_acc
            torch.save(mlp_shared.state_dict(),
                       os.path.join(run_dir, "best_mlp_model_shared%s.pt" % ck))
    print("[MLP shared] best_tgt_valid=%.4f (seed=42)" % best_mlp_shared)
    
    args.lr = mlp_saved_lr
    args.epochs = mlp_saved_epochs
    args.weight_decay = mlp_saved_wd
    shared_mlp_path = os.path.join(run_dir, "best_mlp_model_shared%s.pt" % ck)
    
    reports = []
    for seed in args.seeds:
        utils.set_seed(seed)
        
        source = datasets.prepare_dblp_acm(data_root, args.src_name)
        target = datasets.prepare_dblp_acm(data_root, args.tgt_name)
        datasets.adjust_graph_structure_fast_source(source, h_thresh=args.h_threshold)
        
        source2 = source.clone()
        target2 = target.clone()
        datasets.adjust_graph_structure_fast_source(source2, h_thresh=args.h_threshold)
        target2.rw_stats_history = []
        
        model = models_module.GNN_adv(input_dim, output_dim, args).to(device)
        mlp_model2 = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
        mlp_model2.load_state_dict(torch.load(shared_mlp_path, map_location=device))
        scheduler, opt = utils.build_optimizer(args, model.parameters())
        
        best_valid, best_test = 0.0, 0.0
        for epoch in range(args.epochs):
            model.train()
            src_data = source2.to(device)
            tgt_data = target2.to(device)
            mlp_model2.eval()
            
            with torch.no_grad():
                _, tgt_mlp_pred = mlp_model2(tgt_data)
            _, mlp_pred_tgt = tgt_mlp_pred.max(dim=1)
            
            if args.reweight and epoch >= (args.start_epoch - 1):
                _, [pred_tgt, _] = model.forward(tgt_data, 1)
                _, pred_tgt = pred_tgt.max(dim=1)
                tgt_data.y_hat[pred_tgt == mlp_pred_tgt] = mlp_pred_tgt[pred_tgt == mlp_pred_tgt]
                if (epoch - 1) % args.rw_freq == 0:
                    datasets.adjust_graph_structure_fast_target_Plabel(tgt_data, h_thresh=args.h_threshold)
                    tgt_data = tgt_data.to(device)
            
            da_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)
            [_, _], [pred_src, pred_domain_src] = model.forward(src_data, da_alpha)
            [_, _], [_, pred_domain_tgt] = model.forward(tgt_data, da_alpha)
            
            mask_src = src_data.source_training_mask
            cls_loss = utils.ce_loss(pred_src[mask_src], src_data.y[mask_src])
            domain_loss = utils.bce_loss(pred_domain_src[src_data.source_mask],
                torch.zeros_like(pred_domain_src[src_data.source_mask]))
            domain_loss += utils.bce_loss(pred_domain_tgt[tgt_data.target_mask],
                torch.ones_like(pred_domain_tgt[tgt_data.target_mask]))
            loss = cls_loss + domain_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                [_, _], [pred, _] = model.forward(tgt_data, 1)
                tgt_test_acc, _ = utils.classification_scores(
                    pred[tgt_data.target_testing_mask], tgt_data.y[tgt_data.target_testing_mask])
                tgt_val_acc, _ = utils.classification_scores(
                    pred[tgt_data.target_validation_mask], tgt_data.y[tgt_data.target_validation_mask])
            if tgt_val_acc > best_valid:
                best_valid = tgt_val_acc
                best_test = tgt_test_acc
        
        reports.append({"acc_tgt_test": best_test, "acc_tgt_valid": best_valid})
        print("[Seed %d] acc_tgt_test=%.4f, acc_tgt_valid=%.4f" % (seed, best_test, best_valid))
    
    accs = [r["acc_tgt_test"] for r in reports]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    summary = {"acc_tgt_test": "%.5f +/- %.5f" % (mean_acc, std_acc)}
    print("[Summary] %s" % json.dumps(summary))
    
    metrics = {
        "accuracy": round(float(mean_acc), 4),
        "accuracy_std": round(float(std_acc), 4),
        "per_seed": {i+1: round(float(accs[i]), 4) for i in range(len(accs))}
    }
    os.makedirs(os.path.join(_REPO, "outputs"), exist_ok=True)
    with open(os.path.join(_REPO, "outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return summary

if __name__ == "__main__":
    main()
