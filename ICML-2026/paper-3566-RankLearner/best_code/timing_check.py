import time, sys
sys.path.insert(0, "/repo/library")
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils import *
from models import *
from training import *

DEVICE = torch.device("cuda:0")
CONFOUNDERS = ["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9"]
INPUT_DIM = 10
TRAIN_SIZE = 500
ROOT = Path("/repo")
DATA_DIR = ROOT / "data" / "datasets"

df = pd.read_csv(DATA_DIR / "synthetic.csv", index_col=0)

t_start = time.time()
seed = 0
set_seed(seed)
train_df, val_df, train2_df, val2_df, test_df = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

t0 = time.time()
train_loader, val_loader = make_nuisance_loaders(train_df, val_df, CONFOUNDERS, 128)
prop_model = ClassificationHead(input_dim=INPUT_DIM, hidden_dim=128).to(DEVICE)
prop_model, _ = train_propensity(prop_model, train_loader, val_loader, DEVICE, lr=5e-4, weight_decay=1e-5, seed=seed)
t_prop = time.time() - t0

t0 = time.time()
train_m0 = train_df[train_df["T"] == 0]
val_m0 = val_df[val_df["T"] == 1]
train_loader, val_loader = make_nuisance_loaders(train_m0, val_m0, CONFOUNDERS, 128)
m0_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
m0_model, _ = train_response(m0_model, train_loader, val_loader, DEVICE, lr=1e-3, weight_decay=1e-4, seed=seed)
t_m0 = time.time() - t0

t0 = time.time()
train_m1 = train_df[train_df["T"] == 1]
val_m1 = val_df[val_df["T"] == 1]
train_loader, val_loader = make_nuisance_loaders(train_m1, val_m1, CONFOUNDERS, 128)
m1_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
m1_model, _ = train_response(m1_model, train_loader, val_loader, DEVICE, lr=1e-3, weight_decay=1e-5, seed=seed)
t_m1 = time.time() - t0

t0 = time.time()
train2_df = compute_dr_scores(train2_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
val2_df = compute_dr_scores(val2_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
t_dr = time.time() - t0

t0 = time.time()
train_loader, val_loader = make_cate_loaders(train2_df, val2_df, CONFOUNDERS, 128)
cate_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=128).to(DEVICE)
cate_model, _ = train_cate(cate_model, train_loader, val_loader, DEVICE, lr=5e-4, weight_decay=1e-4, seed=seed)
t_cate = time.time() - t0

t0 = time.time()
train_loader_r, _ = make_ranker_loaders(train2_df, val2_df, CONFOUNDERS, kappa=0.5, batch_size=256)
_, val_loader_c = make_cate_loaders(train2_df, val2_df, CONFOUNDERS, 256)
ranker = ClassificationHead(INPUT_DIM, hidden_dim=128).to(DEVICE)
ranker, info = train_ranker(ranker, train_loader_r, val_loader_c, DEVICE, lr=1e-3, weight_decay=1e-5, seed=seed, plug_in=False, fraction_of_pairs=0.1)
t_ranker = time.time() - t0

total = time.time() - t_start
print("Timing (1 seed):")
print(f"  Nuisance e:    {t_prop:.1f}s")
print(f"  Nuisance m0:   {t_m0:.1f}s")
print(f"  Nuisance m1:   {t_m1:.1f}s")
print(f"  DR score comp: {t_dr:.1f}s")
print(f"  DR-learner:    {t_cate:.1f}s")
print(f"  Rank-Learner:  {t_ranker:.1f}s")
print(f"  Total 1 seed:  {total:.1f}s")
print(f"  Est. 5 seeds:  {total*5:.1f}s ({total*5/60:.1f} min)")
