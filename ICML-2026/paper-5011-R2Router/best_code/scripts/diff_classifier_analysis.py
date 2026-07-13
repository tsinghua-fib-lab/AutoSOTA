#!/usr/bin/env python3
"""Diff classifier: predict when GPT-5.2 beats best-of-4 models."""
import json, numpy as np, pickle, math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, 'scripts')

with open('/orange/qi855292.ucf/ah872032.ucf/category_router/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

emb = data['embeddings']
cats = data['categories']
gi_list = data['global_indices']

m4 = ['235b', '80b', 'gemini-flash', 'haiku']
best4 = np.zeros(8400)
for m in m4:
    best4 = np.maximum(best4, data['models'][m]['concise']['accuracy'])
gpt = data['models']['gpt-5.2']['concise']['accuracy']

label = (gpt > best4).astype(int)
print(f'Positive (GPT-5.2 > best-of-4): {label.sum()}/{len(label)} ({label.mean()*100:.1f}%)', flush=True)

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
tri, tei = next(sss.split(emb, cats))
print(f'Train: {len(tri)}, Test: {len(tei)}', flush=True)

clf = LogisticRegression(C=1.0, max_iter=500)
clf.fit(emb[tri], label[tri])
yp = clf.predict_proba(emb[tei])[:, 1]
auc = roc_auc_score(label[tei], yp)
print(f'AUC: {auc:.3f}', flush=True)

sw = '/orange/qi855292.ucf/ah872032.ucf/budget_sweep'
gpt_cost = {}
with open(f'{sw}/gpt-5.2/concise.json') as f:
    for e in json.load(f):
        gpt_cost[e['global index']] = e.get('cost', 0)

b4_cost = {}
for m in m4:
    with open(f'{sw}/{m}/concise.json') as f:
        for e in json.load(f):
            g = e['global index']
            a = e.get('accuracy', 0)
            c = e.get('cost', 0)
            p = b4_cost.get(g)
            if p is None or a > p[0] or (a == p[0] and c < p[1]):
                b4_cost[g] = (a, c)

def arena(acc, cost_1kq, beta=0.1):
    C = (math.log2(200) - math.log2(max(cost_1kq, 0.001))) / (math.log2(200) - math.log2(0.0044))
    return ((1 + beta**2) * acc * C) / (beta**2 * acc + C) * 100

print(flush=True)
print('Thresh  GPT-used  Acc%     Cost       Arena', flush=True)
for t in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.9]:
    pred = (yp >= t).astype(int)
    ta = tc = 0
    for i, idx in enumerate(tei):
        g = gi_list[idx]
        if pred[i] == 1:
            ta += gpt[idx]
            tc += gpt_cost.get(g, 0)
        else:
            ta += best4[idx]
            tc += b4_cost.get(g, (0, 0))[1]
    n = len(tei)
    ra = ta / n
    rc = tc / n * 1000
    print(f'  {t:.2f}   {pred.sum():>5}    {ra*100:.2f}    ${rc:.3f}     {arena(ra, rc):.2f}', flush=True)

b4a = best4[tei].mean()
b4cc = np.mean([b4_cost.get(gi_list[i], (0, 0))[1] for i in tei]) * 1000
ga = gpt[tei].mean()
gcc = np.mean([gpt_cost.get(gi_list[i], 0) for i in tei]) * 1000
print(flush=True)
print(f'Best-of-4:   Acc={b4a*100:.2f}%  Cost=${b4cc:.3f}  Arena={arena(b4a, b4cc):.2f}', flush=True)
print(f'Pure GPT52:  Acc={ga*100:.2f}%  Cost=${gcc:.3f}  Arena={arena(ga, gcc):.2f}', flush=True)
