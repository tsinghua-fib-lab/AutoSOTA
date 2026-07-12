#!/usr/bin/env python3
"""Evaluate a ConOrd checkpoint on BID dataset."""
import sys, os, numpy as np, torch, random
sys.path.insert(0, "/repo")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--data_root", type=str, default="/datasets/BID/BID")
parser.add_argument("--k_nn", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--split_seed", type=int, default=42)
args = parser.parse_args()

from config.basic import ConfigBasic
from data.get_datasets_BIQA import get_datasets_BIQA
from networks.util import prepare_model
from utils.util import to_np, cal_srocc_plcc
from utils.comparison_utils import find_kNN

cfg = ConfigBasic()
cfg.dataset = "bid"
cfg.data_root = args.data_root
cfg.set_biqa_dataset()
cfg.device = torch.device(f"cuda:{args.gpu}")
cfg.batch_size = 32
cfg.test_batch_size = 1000
cfg.num_workers = 0
cfg.model = "ConOrd"
cfg.backbone = "vitB16"
cfg.ref_mode = "flex"
cfg.ref_point_num = 60
cfg.fiducial_point_num = 60
cfg.start_norm = True
cfg.drct_wieght = 0
cfg.k = args.k_nn
cfg.metric = "L2"
cfg.split_seed = args.split_seed

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

loader_dict = get_datasets_BIQA(cfg)
model = prepare_model(cfg)
ckpt = torch.load(args.checkpoint, map_location=cfg.device)
model.load_state_dict(ckpt["model"])
model = model.to(cfg.device)
model.eval()

embs_train, embs_test = [], []
with torch.no_grad():
    for x, _, _, _ in loader_dict["train_for_val"]:
        embs_train.append(model.encoder(x.to(cfg.device)).cpu())
    for x, _, _, _ in loader_dict["val"]:
        embs_test.append(model.encoder(x.to(cfg.device)).cpu())

embs_train = torch.cat(embs_train).to(cfg.device)
embs_test = torch.cat(embs_test).to(cfg.device)
train_labels = np.array(loader_dict["train_for_val"].dataset.mos)
test_labels = np.array(loader_dict["val"].dataset.mos)

vals, inds = find_kNN(embs_test, embs_train, k=cfg.k, metric=cfg.metric)
inds = np.squeeze(to_np(inds), 0)
if inds.ndim == 1:
    inds = inds[np.newaxis, :]
nn_labels = train_labels[inds[:, :cfg.k]]
preds = np.mean(nn_labels, axis=-1)
srcc, plcc = cal_srocc_plcc(preds, test_labels)
print(f"SRCC: {srcc:.4f}")
print(f"PLCC: {plcc:.4f}")
