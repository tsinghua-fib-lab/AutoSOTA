#!/usr/bin/env python3
"""Continue training from a checkpoint with modified hyperparameters."""
import argparse, os, time, sys, random
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from losses import kmeans_obj_batched, SoftKMObj
from ctasks import ClusteringTasks
from kmt import KModel
from utils import lloyds_iters_batched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--extra_steps", type=int, default=5000)
    parser.add_argument("--nsteps_per_eval", type=int, default=200)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--use_cosine", action="store_true")
    parser.add_argument("--warmup_pct", type=float, default=0.1)
    parser.add_argument("--use_clip_grad", action="store_true")
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--odir", type=str, default="runs")
    parser.add_argument("--run_tag", type=str, default="continue")
    args = parser.parse_args()

    GPUE = torch.cuda.is_available()
    DEVICE = torch.cuda.current_device() if GPUE else "cpu"
    print(f"Device: {DEVICE}")

    chkpt = torch.load(args.checkpoint, map_location="cpu")
    cli_args = chkpt["cli_args"]
    val_tasks = chkpt["val_tasks"]
    val_kmeans_obj = chkpt.get("val_kmeans_obj")
    start_step = chkpt.get("step", 0)
    print(f"Loaded: step={start_step}, best_val_rel_obj={chkpt.get('best_val_rel_obj', 'N/A')}")

    ssize = {"onehot": cli_args["nclusters"], "none": 0}
    demb = cli_args["ndims"] + ssize[cli_args["scratch"]]
    use_qk_norm = cli_args.get("use_qk_norm", False)
    model = KModel(demb, cli_args["dqkv"] * demb,
        inv_temp=cli_args["attn_itemp"], dropout_p=cli_args["dropout"],
        act=cli_args["attn_act"], use_qk_norm=use_qk_norm)
    model.load_state_dict(chkpt["model_state_dict"])
    if GPUE:
        model = model.to(DEVICE)

    task_gen = ClusteringTasks(
        cli_args["nclusters"], cli_args["nclusters"], cli_args["ndims"],
        cli_args["nlb"], cli_args["nub"],
        [d for d in cli_args["train_dists"].split("+")],
        scale=cli_args["scale"], equal_mix=cli_args["em"], equal_scales=cli_args["es"])

    # Move val_tasks to GPU if needed
    if GPUE:
        val_tasks = [(XX.to(DEVICE), CC.to(DEVICE)) for (XX, CC) in val_tasks]

    if val_kmeans_obj is None:
        val_kmeans_obj = [lloyds_iters_batched(XX, CC, niters=1)
                          for (XX, CC) in val_tasks]

    criterion = SoftKMObj(gamma=cli_args["loss_itemp"],
                          act=cli_args["loss_act"],
                          logloss=cli_args.get("logloss", False))

    optimizer = Adam(model.parameters(), lr=args.init_lr)

    if args.use_cosine:
        warmup_steps = int(args.warmup_pct * args.extra_steps)
        def warmup_fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        cosine_scheduler = CosineAnnealingLR(optimizer,
            T_max=args.extra_steps - warmup_steps)
        print(f"Cosine: {warmup_steps} warmup + "
              f"{args.extra_steps - warmup_steps} cosine")
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=cli_args["lr_decay"],
                                       patience=cli_args["patience"])

    def fpass(samples, centers):
        b, n, d = samples.shape
        dev = samples.device
        scratch_type = cli_args["scratch"]
        if scratch_type == "none":
            XX = samples.clone().detach()
            CC = centers.clone().detach()
        else:
            YY = torch.zeros(b, n, ssize[scratch_type], device=dev)
            MM = torch.eye(ssize[scratch_type], device=dev).repeat(b, 1, 1)
            XX = torch.cat((samples.clone().detach(), YY), dim=2)
            CC = torch.cat((centers.clone().detach(), MM), dim=2)
        XXX, CCC = model(XX, CC)
        return CCC[:, :, :cli_args["ndims"]]

    nstr = f"continue_{args.run_tag}_s{start_step}+{args.extra_steps}"
    fname_last = os.path.join(args.odir, f"{nstr}_last.pt")
    fname_best = os.path.join(args.odir, f"{nstr}_best.pt")
    fname_csv = os.path.join(args.odir, f"{nstr}.csv")

    stats = []
    curr_best_val = chkpt.get("best_val_rel_obj", float("inf"))
    model.train()

    for step in range(args.extra_steps + 1):
        if step % args.nsteps_per_eval == 0 or step == args.extra_steps:
            model.eval()
            save_best = False
            with torch.no_grad():
                model_kmeans_objs = [
                    kmeans_obj_batched(XX, fpass(XX, CC)).to("cpu")
                    for XX, CC in val_tasks
                ]
                val_obj = torch.mean(torch.stack(model_kmeans_objs)).item()
                rel_val_obj = torch.mean(torch.stack([
                    torch.div(mm, vv[-1])
                    for mm, vv in zip(model_kmeans_objs, val_kmeans_obj)
                ])).item()
                if rel_val_obj < curr_best_val:
                    curr_best_val = rel_val_obj
                    save_best = True
                print(f"Step {step}/{args.extra_steps}: "
                      f"val_obj={val_obj:.2f}, rel={rel_val_obj:.4f}, "
                      f"best_rel={curr_best_val:.4f}")
                stats.append((start_step + step, "val-obj", val_obj))
                stats.append((start_step + step, "val-rel-obj", rel_val_obj))

            save_dict = {
                "model_state_dict": model.state_dict(),
                "best_val_rel_obj": curr_best_val,
                "val_obj": val_obj,
                "step": start_step + step,
                "val_tasks": val_tasks,
                "val_kmeans_obj": val_kmeans_obj,
                "cli_args": cli_args,
            }
            torch.save(save_dict, fname_last)

            if save_best:
                torch.save(save_dict, fname_best)
                print(f"  -> new best!")

            if step == args.extra_steps:
                print(f"Done! {args.extra_steps} extra steps completed.")
                break

            model.train()
            if step > 0:
                if args.use_cosine:
                    if step < warmup_steps:
                        warmup_scheduler.step()
                    else:
                        cosine_scheduler.step()
                else:
                    scheduler.step(rel_val_obj)
                print(f"  lr: {optimizer.param_groups[0]['lr']:.6f}")

        XX, CC = task_gen.sample_batch(cli_args["bsz"],
                                        same_dist_batch=cli_args["sdb"])
        CCC = fpass(XX, CC)
        loss = torch.mean(criterion(XX, CCC))
        optimizer.zero_grad()
        loss.backward()
        if args.use_clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
        optimizer.step()
        stats.append((start_step + step, "train-loss", loss.item()))

    pd.DataFrame(stats, columns=["step", "set", "loss"]).to_csv(
        fname_csv, header=True, index=False)
    print(f"Final checkpoint: {fname_last}")

if __name__ == "__main__":
    main()
