"""Compute prototypes for swept layers."""
import numpy as np, os, sys
from sklearn.model_selection import KFold

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

os.makedirs('/repo/prototypes_sweep', exist_ok=True)

for layer in [12, 14, 16]:
    path = '/repo/features_sweep/Qwen2.5-7B-Instruct_layer%d.npz' % layer
    if not os.path.exists(path):
        print('SKIP layer %d: no features' % layer)
        continue
    data = np.load(path)
    X, y, qi = data['activations'], data['labels'], data['q_indices']
    uq = np.unique(qi)
    kf = KFold(n_splits=2, shuffle=False)
    for fi, (tr, te) in enumerate(kf.split(uq)):
        tm = np.isin(qi, uq[tr])
        Xt, yt = X[tm], y[tm]
        d = np.mean(Xt[yt==1], axis=0) - np.mean(Xt[yt==0], axis=0)
        mu = normalize(d); mh = -mu
        em = np.isin(qi, uq[te])
        acc = np.mean((np.dot(X[em], mu) > 0) == y[em])
        save_path = '/repo/prototypes_sweep/Qwen2.5-7B-Instruct_layer%d_fold%d.npz' % (layer, fi)
        np.savez(save_path, mu_T=mu, mu_H=mh, test_q_indices=uq[te], fold_idx=fi)
        print('Layer %d Fold %d: proto_acc=%.4f saved to %s' % (layer, fi, acc, save_path))
print('Done computing prototypes')
