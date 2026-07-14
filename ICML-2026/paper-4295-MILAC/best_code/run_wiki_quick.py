#!/usr/bin/env python3
"""MILCCI Wikipedia Experiment - Quick Pipeline Validation.
90 days, English only, 2 categories (agent, platform).
"""
import os, sys, time, json
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, '/repo')
import milcci
from milcci import per_trial_r2, global_r2

CACHE_DIR = '/datasets/milcci_wikipedia'
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_NPZ = os.path.join(CACHE_DIR, "wiki_quick.npz")

START, END = "20201009", "20210107"
TDAYS = (datetime.strptime(END, "%Y%m%d") - datetime.strptime(START, "%Y%m%d")).days + 1
print(f"Quick validation: {START}-{END} ({TDAYS} days)", flush=True)

PAGES = [
    "Classical_conditioning","Bobo_doll_experiment","Operant_conditioning",
    "Self-concept","Little_Albert_experiment","Unsupervised_learning",
    "Embedding","The_Social_Network","Social_media","Ivan_Pavlov",
    "Mark_Zuckerberg","Data_mining","Computer_science","Supervised_learning",
    "Computer_scientist","Cambridge_Analytica","Facebook","Twitter",
    "Machine_learning","Deep_learning","Artificial_intelligence",
    "Neural_network_(machine_learning)","Natural_language_processing",
    "Reinforcement_learning","Big_data","Algorithm","Statistics",
    "Linear_regression","Decision_tree_learning","Support_vector_machine",
    "Database","Cognitive_psychology",
]
N = len(PAGES)

HEADERS = {"User-Agent": "MILCCI-Reproduction/1.0 (academic@example.com)"}

def main():
    if os.path.exists(CACHE_NPZ):
        print("Loading cached data...", flush=True)
        data = dict(np.load(CACHE_NPZ, allow_pickle=True))
        Y = data['Y']
        labels = data['labels'].tolist()
        nt = data['numbers2tuples'].item()
        numbers2tuples = {int(k): tuple(v) for k, v in nt.items()}
        print(f"Loaded: Y {Y.shape}, {len(labels)} trials", flush=True)
    else:
        # 2 categories: agent(user,spider), platform(desktop,mobile-web,mobile-app)
        # Spider only has desktop + mobile-web (no app)
        trials = [
            ("user","desktop"), ("user","mobile-web"), ("user","mobile-app"),
            ("spider","desktop"), ("spider","mobile-web"),
        ]
        M = len(trials)
        print(f"Downloading {N} pages x {M} trials ({N*M} series)", flush=True)

        Y = np.zeros((N, TDAYS, M), dtype=np.float64)
        sd = datetime.strptime(START, "%Y%m%d")

        for pi, page in enumerate(PAGES):
            t0 = time.time()
            n_ok = 0
            for m, (agent, plat) in enumerate(trials):
                ok = False
                for attempt in range(3):
                    try:
                        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                               f"per-article/en.wikipedia.org/{plat}/{agent}/{page}/"
                               f"daily/{START}/{END}")
                        r = requests.get(url, headers=HEADERS, timeout=60)
                        if r.status_code == 200:
                            for item in r.json().get("items",[]):
                                try:
                                    d = datetime.strptime(item["timestamp"][:8],"%Y%m%d")
                                    idx = (d-sd).days
                                    if 0 <= idx < TDAYS:
                                        Y[pi,idx,m] = item["views"]
                                except:
                                    pass
                            if Y[pi,:,m].sum() > 0:
                                n_ok += 1
                            ok = True
                            break
                        elif r.status_code == 404:
                            break
                        elif r.status_code == 429:
                            time.sleep(int(r.headers.get("Retry-After",5)))
                        else:
                            time.sleep(1)
                    except:
                        time.sleep(1)
                if not ok:
                    pass
                time.sleep(0.05)
            dt = time.time() - t0
            print(f"[{pi+1:2d}/{N}] {page}: {n_ok}/{M} ok, {dt:.1f}s", flush=True)

        # Preprocess: 99th percentile normalization
        print("Preprocessing...", flush=True)
        for m in range(M):
            trial = Y[:,:,m]
            pos = trial[trial>0]
            if len(pos) > 0:
                p99 = np.percentile(pos, 99)
                if p99 > 0:
                    Y[:,:,m] = np.clip(trial/p99, 0, 1)
            Y[:,:,m] -= Y[:,:,m].min()
        for n in range(N):
            td = Y[n,:,:]
            pos = td[td>0]
            if len(pos) > 0:
                p99 = np.percentile(pos, 99)
                if p99 > 0:
                    Y[n,:,:] = np.clip(td/p99, 0, 1)

        # 2 categories: agent (0/1), platform (0/1/2)
        labels = list(range(M))
        numbers2tuples = {
            0: (0,0),  # user, desktop
            1: (0,1),  # user, mobile-web
            2: (0,2),  # user, mobile-app
            3: (1,0),  # spider, desktop
            4: (1,1),  # spider, mobile-web
        }

        nt_save = {str(k): list(v) for k,v in numbers2tuples.items()}
        np.savez_compressed(CACHE_NPZ, Y=Y, labels=np.array(labels),
                           numbers2tuples=nt_save)
        print("Data cached.", flush=True)

    # Run MILCCI - 2 categories, 4 components each = 8 total
    print(f"\n--- MILCCI (Y: {Y.shape}, P=8) ---", flush=True)
    t0 = time.time()
    result = milcci.fit(
        data=Y, labels=labels, numbers2tuples=numbers2tuples,
        n_ensembles=8, n_ensembles_each=[4,4],
        nu=[0.01]*8, lambda_similarity=100, factor_A=5,
        decor_A=2, num_repeats=15,
        cont_axis_list=[], split_A=True,
        params_init_A={'ensemble_positive': True},
        verbose=True, seed=42,
    )
    runtime = time.time() - t0

    # Evaluate
    Phi = result['Phi']
    A_full = result['A_full']
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS (quick validation: {TDAYS}d, en only, 2 categories)", flush=True)
    print(f"{'='*60}", flush=True)
    for i, (a,p) in enumerate([("user","desktop"),("user","mobile-web"),
                                ("user","mobile-app"),("spider","desktop"),
                                ("spider","mobile-web")]):
        print(f"  R2 trial {i} ({a}/{p}): {r2_vec[i]:.4f}", flush=True)
    print(f"Per-trial R2: mean={r2_mean:.4f}  std={np.std(r2_vec):.4f}", flush=True)
    print(f"Global R2:    {r2_global:.4f}", flush=True)
    print(f"Runtime:      {runtime:.2f}s", flush=True)

    # Save
    with open(os.path.join(CACHE_DIR, "quick_results.json"), 'w') as f:
        json.dump({"r2_mean": r2_mean, "r2_global": r2_global,
                   "runtime": runtime, "tdays": TDAYS}, f, indent=2)

    return r2_mean, runtime

if __name__ == '__main__':
    main()
