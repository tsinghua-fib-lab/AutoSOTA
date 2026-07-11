#!/usr/bin/env python3
import argparse
import os
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm
import sys
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
from src.evaluation.utils.specific_helpers import load_model
from src.evaluation.utils.helpers import extract_context

from gamba.collators import gLMCollator, gLMMLMCollator


# ----------------------------
# masking (reuse your logic)
# ----------------------------
def apply_effective_region_mask(
    labels: torch.Tensor,                      # (B, 2, T)
    feature_spans: list[tuple[int, int]],      # per-sample (fs, fe) token indices (already shifted if needed)
    is_mlm: bool,
    last_k: int = 1000,
) -> torch.Tensor:
    labels = labels.clone()
    B, two, T = labels.shape
    assert two == 2

    for b, (fs, fe) in enumerate(feature_spans):
        fs = max(0, min(fs, T))
        fe = max(0, min(fe, T))
        roi_len = max(0, fe - fs)

        if roi_len == 0:
            labels[b, 0, :] = -100
            labels[b, 1, :] = -100
            continue

        k = min(last_k, roi_len)
        if not is_mlm:
            eff_fs = fe - k
            eff_fe = fe
        else:
            eff_fs = fs
            eff_fe = fe

        # sequence channel
        if is_mlm:
            keep = torch.zeros(T, dtype=torch.bool, device=labels.device)
            keep[eff_fs:eff_fe] = True
            masked = labels[b, 0, :] != -100
            kill = masked & (~keep)
            labels[b, 0, kill] = -100
        else:
            labels[b, 0, :eff_fs] = -100
            labels[b, 0, eff_fe:] = -100

        # conservation channel
        labels[b, 1, :eff_fs] = -100
        labels[b, 1, eff_fe:] = -100

        # ignore special tokens if present
        labels[b, 0, 0]  = -100
        labels[b, 1, 0]  = -100
        labels[b, 0, -1] = -100
        labels[b, 1, -1] = -100

    return labels


def _masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1)
    return num / den


def _pearsonr_per_row(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred,tgt,mask: (B,T)
    returns: (B,) pearson r, nan if <2 valid points or zero variance
    """
    B, T = pred.shape
    out = torch.full((B,), float("nan"), device=pred.device)

    for b in range(B):
        m = mask[b].bool()
        if m.sum() < 2:
            continue
        x = pred[b][m]
        y = tgt[b][m]
        x = x - x.mean()
        y = y - y.mean()
        vx = (x * x).mean()
        vy = (y * y).mean()
        if vx <= 0 or vy <= 0:
            continue
        out[b] = (x * y).mean() / torch.sqrt(vx * vy)
    return out


# ----------------------------
# batched scoring
# ----------------------------
@torch.no_grad()
def score_regions_batched(
    model,
    tokenizer,
    regions: list[dict],
    *,
    model_type: str,
    training_task: str,
    batch_size: int,
    device: torch.device,
    last_k: int = 1000,
    caduceus_repeats: int = 7,
):
    """
    returns per-region metrics:
      ce_loss (float), phyloP_corr (float)
    """
    from torch.nn import functional as F

    if model_type == "gamba":
        collator = gLMCollator(tokenizer=tokenizer, test=True)
    else:
        collator = gLMMLMCollator(tokenizer=tokenizer, test=True)

    ce_all = []
    corr_all = []

    for i in tqdm(range(0, len(regions), batch_size), desc=f"scoring {model_type}"):
        batch_regions = regions[i:i + batch_size]
        batch_inputs = []
        raw_spans = []

        for r in batch_regions:
            seq_tokens = tokenizer.tokenizeMSA(r["sequence"])
            scores = r["scores"]
            fs = int(r["feature_start_in_window"])
            fe = int(r["feature_end_in_window"])
            batch_inputs.append((seq_tokens, scores))
            raw_spans.append((fs, fe))

        if not batch_inputs:
            continue

        # ---------------- gamba (AR) ----------------
        if model_type == "gamba":
            inputs, labels = collator(batch_inputs)   # labels: (B,2,T)
            inputs = inputs.to(device)
            labels = labels.to(device)

            feature_spans = [(fs + 1, fe + 1) for (fs, fe) in raw_spans]
            labels = apply_effective_region_mask(labels, feature_spans, is_mlm=False, last_k=last_k)

            out = model(inputs, labels)
            seq_logits = out["seq_logits"].float()                 # (B,T,V)
            cons_pred  = out.get("scaling_logits", None)           # (B,T,2) maybe

            # CE (AR shift)
            ce_labels = labels[:, 0, :].long()                     # (B,T)
            logit_shift = seq_logits[:, :-1, :]
            label_shift = ce_labels[:, 1:]
            mask_shift  = label_shift.ne(-100).float()

            ce_tok = F.cross_entropy(
                logit_shift.reshape(-1, logit_shift.size(-1)),
                label_shift.reshape(-1),
                reduction="none"
            ).view(label_shift.size())                             # (B,T-1)

            ce_per_ex = _masked_mean_per_row(ce_tok, mask_shift, dim=1)  # (B,)

            # corr (cons head) over ROI mask
            if cons_pred is not None:
                pred = cons_pred[..., 0].float()                   # (B,T)
                tgt  = labels[:, 1, :].float()                     # (B,T)
                m    = tgt.ne(-100)
                corr_per_ex = _pearsonr_per_row(pred, tgt, m)
            else:
                corr_per_ex = torch.full_like(ce_per_ex, float("nan"))

            ce_all.extend(ce_per_ex.detach().cpu().tolist())
            corr_all.extend(corr_per_ex.detach().cpu().tolist())

        # -------------- caduceus (MLM) --------------
        else:
            B = len(batch_inputs)
            ce_accum   = torch.zeros(B, dtype=torch.float32, device=device)
            corr_accum = torch.zeros(B, dtype=torch.float32, device=device)
            ce_used    = torch.zeros(B, dtype=torch.float32, device=device)
            corr_used  = torch.zeros(B, dtype=torch.float32, device=device)

            feature_spans_shifted = [(fs + 1, fe + 1) for (fs, fe) in raw_spans]

            for _ in range(caduceus_repeats):
                batch = collator(batch_inputs, region=raw_spans)
                input_ids  = batch[0][:, 0, :].long().to(device)
                labels_pack = batch[1].to(device)                  # (B,2,T)

                labels_pack = apply_effective_region_mask(
                    labels_pack, feature_spans_shifted, is_mlm=True, last_k=last_k
                )

                outputs = model(input_ids=input_ids, return_dict=True)

                # CE on masked tokens ∩ ROI
                if "logits" in outputs:
                    logits = outputs["logits"].float()
                    ce_labels = labels_pack[:, 0, :].long()
                    ce_tok = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        ce_labels.reshape(-1),
                        reduction="none"
                    ).view(ce_labels.size())
                    ce_mask = ce_labels.ne(-100).float()
                    ce_per_ex = _masked_mean_per_row(ce_tok, ce_mask, dim=1)
                    # track which rows had any masked tokens inside ROI this repeat
                    has = (ce_mask.sum(dim=1) > 0).float()
                    ce_accum += torch.nan_to_num(ce_per_ex, nan=0.0) * has
                    ce_used  += has
                else:
                    # no logits => can’t compute CE
                    pass

                # corr on conservation head over ROI (not masked-only)
                if "scaling_logits" in outputs:
                    pred = outputs["scaling_logits"][..., 0].float()
                    tgt  = labels_pack[:, 1, :].float()
                    m    = tgt.ne(-100)
                    corr_per_ex = _pearsonr_per_row(pred, tgt, m)
                    has = (m.sum(dim=1) >= 2).float()
                    corr_accum += torch.nan_to_num(corr_per_ex, nan=0.0) * has
                    corr_used  += has
                else:
                    pass

            ce_mean = torch.full((B,), float("nan"), device=device)
            corr_mean = torch.full((B,), float("nan"), device=device)

            ok_ce = ce_used > 0
            ce_mean[ok_ce] = ce_accum[ok_ce] / ce_used[ok_ce]

            ok_corr = corr_used > 0
            corr_mean[ok_corr] = corr_accum[ok_corr] / corr_used[ok_corr]

            ce_all.extend(ce_mean.detach().cpu().tolist())
            corr_all.extend(corr_mean.detach().cpu().tolist())

    return np.array(ce_all, dtype=float), np.array(corr_all, dtype=float)


# ----------------------------
# rmsk parsing + sampling
# ----------------------------
def load_rmsk_table(rmsk_path: str) -> pd.DataFrame:
    """
    expects UCSC rmsk table dump format (no header), like your example.
    columns we use by 0-based index:
      chrom: 5
      start: 6
      end:   7
      strand: 9
      repName: 10
      repClass: 11   (this is your “repeat type”: LINE, SINE, LTR, Simple_repeat, Satellite, DNA...)
      repFamily: 12
    """
    df = pd.read_csv(rmsk_path, sep=r"\s+", header=None, engine="python")
    df = df.rename(columns={
        5: "chrom", 6: "start", 7: "end", 9: "strand",
        10: "repName", 11: "repClass", 12: "repFamily"
    })
    # ensure ints
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def sample_by_repeat_class(df: pd.DataFrame, n_per: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    for rep_class, g in df.groupby("repClass"):
        if len(g) == 0:
            continue
        if len(g) <= n_per:
            out[rep_class] = g.copy()
        else:
            idx = rng.choice(g.index.to_numpy(), size=n_per, replace=False)
            out[rep_class] = g.loc[idx].copy()
    return out


def build_regions_from_rows(
    rows: pd.DataFrame,
    *,
    genome: Fasta,
    bigwig_file: str,
    model_type: str,
) -> list[dict]:
    """
    convert repeat rows -> your region dicts -> extract_context(...) outputs
    """
    regions = []
    #print(f"Rows:", rows)
    for r in rows.itertuples(index=False):
        region = {
            "chrom": getattr(r, "chrom"),
            "start": int(getattr(r, "start")),
            "end": int(getattr(r, "end")),
            "strand": getattr(r, "strand"),
            "feature_id": f"{getattr(r,'repClass')}|{getattr(r,'repName')}|{getattr(r,'chrom')}:{getattr(r,'start')}-{getattr(r,'end')}",
        }
        #print(f"Region:", region["chrom"], region["start"], region["end"], region["feature_id"])
        # only get chromosomes in chrom in {1..22} X
        chromosomes = ["chr" + str(chr) for chr in range(1,23)] + ["chrX"]
        #print(f"Chromosomes:", chromosomes)
        if region["chrom"] in chromosomes:
            ctx = extract_context(bigwig_file, region, genome, model_type)
            #print(f"Example of ctx:", ctx)
            if not ctx or "sequence" not in ctx:
                continue
            # must contain: sequence, scores, feature_start_in_window, feature_end_in_window
            regions.append(ctx)
        else:
            continue
    return regions


# ----------------------------
# main analysis
# ----------------------------
def summarize(metrics: np.ndarray) -> dict:
    m = metrics[np.isfinite(metrics)]
    if len(m) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan}
    return {"n": int(len(m)), "mean": float(m.mean()), "std": float(m.std(ddof=1) if len(m) > 1 else 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmsk_path", type=str, default="/home/mica/gamba/data_processing/data/rmsk.hg38.txt")
    ap.add_argument("--bigwig_file", type=str, default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig")
    ap.add_argument("--genome_fasta", type=str, default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa")

    ap.add_argument("--checkpoint_dir", type=str, default="/home/mica/gamba")
    ap.add_argument("--config_fpath", type=str, default="/home/mica/gamba/configs/jamba-small-240mammalian.json")
    ap.add_argument("--last_step", type=int, default=44000)

    ap.add_argument("--n_per_type", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--training_task", type=str, choices=["dual", "cons_only", "seq_only", "random_init"], required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    # optional: restrict repeats to specific chroms
    ap.add_argument("--chroms", type=str, nargs="*", default=None)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # load rmsk
    df = load_rmsk_table(args.rmsk_path)
    if args.chroms:
        df = df[df["chrom"].isin(set(args.chroms))].copy()

    # sample 1000 per repClass
    samples = sample_by_repeat_class(df, n_per=args.n_per_type, seed=args.seed)
    logging.info(f"repeat classes sampled: {len(samples)}")

    # open genome once
    genome = Fasta(args.genome_fasta)

    # Derive checkpoint_dir 
    gamba_checkpoint_dir = args.checkpoint_dir + "/clean_dcps/CCP/"
    cad_checkpoint_dir = args.checkpoint_dir + "/clean_caduceus_dcps/allPOSMLM"

    # load both models
    logging.info("loading gamba...")
    gamba_model, gamba_tok = load_model(
        gamba_checkpoint_dir, args.config_fpath,
        last_step=args.last_step, device=device,
        training_task=args.training_task, model_type="gamba"
    )
    logging.info("loading caduceus...")
    cad_model, cad_tok = load_model(
        cad_checkpoint_dir, args.config_fpath,
        last_step=args.last_step, device=device,
        training_task=args.training_task, model_type="caduceus"
    )

    rows_out = []

    for rep_class, rows in samples.items():
        logging.info(f"[{rep_class}] building contexts...")
        # build contexts per model type (because extract_context may differ)
        regions_g = build_regions_from_rows(rows, genome=genome, bigwig_file=args.bigwig_file, model_type="gamba")
        regions_c = build_regions_from_rows(rows, genome=genome, bigwig_file=args.bigwig_file, model_type="caduceus")

        logging.info(f"[{rep_class}] n_valid gamba={len(regions_g)} caduceus={len(regions_c)}")

        if len(regions_g) > 0:
            ce_g, corr_g = score_regions_batched(
                gamba_model, gamba_tok, regions_g,
                model_type="gamba", training_task=args.training_task,
                batch_size=args.batch_size, device=device
            )
            s_ce_g = summarize(ce_g)
            s_corr_g = summarize(corr_g)
        else:
            s_ce_g = {"n": 0, "mean": np.nan, "std": np.nan}
            s_corr_g = {"n": 0, "mean": np.nan, "std": np.nan}

        if len(regions_c) > 0:
            ce_c, corr_c = score_regions_batched(
                cad_model, cad_tok, regions_c,
                model_type="caduceus", training_task=args.training_task,
                batch_size=args.batch_size, device=device
            )
            s_ce_c = summarize(ce_c)
            s_corr_c = summarize(corr_c)
        else:
            s_ce_c = {"n": 0, "mean": np.nan, "std": np.nan}
            s_corr_c = {"n": 0, "mean": np.nan, "std": np.nan}

        rows_out.append({
            "repClass": rep_class,
            "n_sampled": int(len(rows)),
            "n_valid_gamba": s_ce_g["n"],
            "gamba_ce_mean": s_ce_g["mean"],
            "gamba_ce_std": s_ce_g["std"],
            "gamba_corr_mean": s_corr_g["mean"],
            "gamba_corr_std": s_corr_g["std"],
            "n_valid_caduceus": s_ce_c["n"],
            "caduceus_ce_mean": s_ce_c["mean"],
            "caduceus_ce_std": s_ce_c["std"],
            "caduceus_corr_mean": s_corr_c["mean"],
            "caduceus_corr_std": s_corr_c["std"],
        })

    out_df = pd.DataFrame(rows_out).sort_values("repClass")
    out_csv = out_dir / "repeatclass_ce_corr_summary.csv"
    out_df.to_csv(out_csv, index=False)
    logging.info(f"wrote: {out_csv}")

    # also save the list of classes + counts
    counts_csv = out_dir / "repeatclass_counts.csv"
    (df.groupby("repClass").size().reset_index(name="n_total")
       .sort_values("n_total", ascending=False)
       .to_csv(counts_csv, index=False))
    logging.info(f"wrote: {counts_csv}")


if __name__ == "__main__":
    main()


# python /home/mica/gamba/src/evaluation/phyloP_corr_on_repeats.py \
#   --rmsk_path /home/mica/gamba/data_processing/data/rmsk.hg38.txt \
#   --bigwig_file /home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig \
#   --genome_fasta /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
#   --checkpoint_dir /home/mica/gamba \
#   --config_fpath /home/mica/gamba/configs/jamba-small-240mammalian.json \
#   --last_step 44000 \
#   --training_task dual \
#   --n_per_type 1000 \
#   --batch_size 32 \
#   --output_dir /home/mica/gamba/data_processing/data/240-mammalian/repeat_loss_eval


