# Code Analysis - GESC (Paper 78)

## Overview
Single-file implementation (gesc.py, 615 lines) of Gauge-Equivariant Graph Networks.

## Evaluation Path
- Entry: train_main()
- Metric: Test accuracy at best validation (Test Acc@BestVal=X.XXXX)
- Output: Best Val Acc=X.XXXX, Test Acc@BestVal=X.XXXX
- Post-processing: C&S + LP by default

## Default Config
hidden=64, layers=2, heads=4, lr=1e-3, wd=5e-4
gamma=0.1, attn_dropout=0.2, feat_dropout=0.5
sic_first=1.5 (first layer only), alpha_skip=0.1
legacy_compat=True (paper_mode=False)
use_cs=True, use_lp=True

## Safe Mod Targets
All CLI args, dropedge(), label smoothing, C&S/LP params, architecture params

## Red Lines
No data/split/label/metric changes. No hard-coded outputs.
