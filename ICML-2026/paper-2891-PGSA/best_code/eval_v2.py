#!/usr/bin/env python3
"""Evaluation script for PSAHS on DBLP-ACM (ACMv9->DBLPv8) with paper settings.
Uses a fixed data split (seed=42) for consistency, varies only model seeds."""
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
    
    # Paper settings
    args.epochs = 300
    args.h_threshold = 1.0
    args.start_epoch = 200
    args.rw_freq = 15
    args.K = 2
    args.hidden_dim = 128
    args.mlp_conv_dim = 128
    args.lr = 0.003
    args.reweight = "rw" in (args.method or "DANN_rw")
    args.opt_decay_rate = 0.8
    args.opt_decay_step = 50
    args.conv_dim = 128
    args.cls_dim = 64
    args.alphamin = 1.0
    args.alphatimes = 1.5
    
    data_root = os.path.join(_REPO, "dataset")
    run_dir = output_dir(args)
    
    # Generate data once with fixed split
    utils.set_seed(42)
    source_template = datasets.prepare_dblp_acm(data_root, args.src_name)
    target_template = datasets.prepare_dblp_acm(data_root, args.tgt_name)
    datasets.adjust_graph_structure_fast_source(source_template, h_thresh=args.h_threshold)
    
    input_dim = source_template.num_node_features
    output_dim = source_template.num_classes
    ck = checkpoint_suffix(args)
    
    reports = []
    for seed in args.seeds:
        utils.set_seed(seed)
        source = source_template.clone()
        target = target_template.clone()
        datasets.adjust_graph_structure_fast_source(source, h_thresh=args.h_threshold)
        
        # Step 1: Train MLP
        mlp_model = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
        mlp_sched, mlp_opt = utils.build_optimizer(args, mlp_model.parameters())
        
        best_mlp = 0.0
        for epoch in range(300):
            mlp_model.train()
            src_data = source.to(device)
            tgt_data = target.to(device)
            mlp_opt.zero_grad()
            _, logits = mlp_model(src_data)
            loss = utils.ce_loss(logits[src_data.source_training_mask], src_data.y[src_data.source_training_mask])
            loss.backward()
            mlp_opt.step()
            mlp_sched.step()
            mlp_model.eval()
            with torch.no_grad():
                _, tgt_logits = mlp_model(tgt_data)
                tgt_val_acc, _ = utils.classification_scores(
                    tgt_logits[target.target_validation_mask], target.y[target.target_validation_mask])
            if tgt_val_acc > best_mlp:
                best_mlp = tgt_val_acc
                torch.save(mlp_model.state_dict(),
                    os.path.join(run_dir, "eval_mlp_seed%d%s.pt" % (seed, ck)))
        
        # Step 2: Train PSAHS
        model = models_module.GNN_adv(input_dim, output_dim, args).to(device)
        mlp_model2 = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
        mlp_model2.load_state_dict(torch.load(
            os.path.join(run_dir, "eval_mlp_seed%d%s.pt" % (seed, ck)), map_location=device))
        scheduler, opt = utils.build_optimizer(args, model.parameters())
        target.rw_stats_history = []
        
        best_valid, best_test = 0.0, 0.0
        for epoch in range(args.epochs):
            model.train()
            src_data = source.to(device)
            tgt_data = target.to(device)
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
    print("[Summary] %.5f +/- %.5f" % (mean_acc, std_acc))
    print("[Per-seed] %s" % json.dumps({i+1: round(float(accs[i]), 4) for i in range(len(accs))}))

if __name__ == "__main__":
    main()
