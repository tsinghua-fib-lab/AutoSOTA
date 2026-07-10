#!/usr/bin/env python3
"""Preprocess raw SMD data from OmniAnomaly for IMPACT open-set TSAD."""
import numpy as np, os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", default="/autosota_cache/tmp/smd_download/omni_extracted/OmniAnomaly-master/ServerMachineDataset")
parser.add_argument("--out_dir", default="/repo/datasets/SMD")
parser.add_argument("--window_len", type=int, default=100)
parser.add_argument("--n_labeled", type=int, default=10)
parser.add_argument("--contam_rate", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
np.random.seed(args.seed)

train_files = sorted(glob.glob(f"{args.raw_dir}/train/*.txt"))
test_files = sorted(glob.glob(f"{args.raw_dir}/test/*.txt"))
test_label_files = sorted(glob.glob(f"{args.raw_dir}/test_label/*.txt"))

windows_train, windows_test, labels_test = [], [], []
W = args.window_len

for tf, tef, tlf in zip(train_files, test_files, test_label_files):
    td = np.loadtxt(tf, delimiter=",")
    ted = np.loadtxt(tef, delimiter=",")
    tl = np.loadtxt(tlf, delimiter=",")
    feat_mean, feat_std = td.mean(axis=0), td.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    td, ted = (td - feat_mean) / feat_std, (ted - feat_mean) / feat_std
    for i in range(len(td) // W):
        windows_train.append(td[i*W:(i+1)*W])
    for i in range(len(ted) // W):
        windows_test.append(ted[i*W:(i+1)*W])
        labels_test.append(int(np.any(tl[i*W:(i+1)*W] > 0)))

train_data = np.array(windows_train, dtype=np.float32)
test_data = np.array(windows_test, dtype=np.float32)
test_label = np.array(labels_test, dtype=np.int32)

n_contam = int(args.contam_rate * (len(train_data) + args.n_labeled) / (1 - args.contam_rate))
anom_idx = np.where(test_label == 1)[0]
selected = np.random.choice(len(anom_idx), size=n_contam+args.n_labeled, replace=False)
contam_idx, labeled_idx = anom_idx[selected[:n_contam]], anom_idx[selected[n_contam:]]

train_final = np.concatenate([train_data, test_data[contam_idx], test_data[labeled_idx]])
train_label_final = np.concatenate([np.zeros(len(train_data), dtype=np.int32), np.zeros(n_contam, dtype=np.int32), np.ones(args.n_labeled, dtype=np.int32)])
used = np.concatenate([contam_idx, labeled_idx])
test_final = np.delete(test_data, used, axis=0)
test_label_final = np.delete(test_label, used)

np.savez(os.path.join(args.out_dir, "SMD_train_general.npz"), data=train_final, label=train_label_final)
np.savez(os.path.join(args.out_dir, "SMD_test.npz"), data=test_final, label=test_label_final)
print(f"Saved: train={train_final.shape[0]}, test={test_final.shape[0]}, test_anoms={test_label_final.sum()}, contam={n_contam}")
