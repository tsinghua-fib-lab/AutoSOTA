#!/usr/bin/env python3
"""
MILCCI Wikipedia Experiment - English Full Range.
1482 days, 32 pages, English, 2 categories (agent, platform), 8 components.
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
CACHE_EN_FULL = os.path.join(CACHE_DIR, "wiki_en_full_1482d.npz")

START, END = "20201009", "20241029"
TDAYS = 1482

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

HEADERS = {"User-Agent":"MILCCI-Reproduction/1.0 (academic)"}

def download_one(project, access, agent, article):
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
           f"per-article/{project}/{access}/{agent}/{article}/"
           f"daily/{START}/{END}")
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                views = np.zeros(TDAYS, dtype=np.float64)
                sd = datetime.strptime(START, "%Y%m%d")
                for item in r.json().get("items",[]):
                    try:
                        d = datetime.strptime(item["timestamp"][:8],"%Y%m%d")
                        idx = (d-sd).days
                        if 0 <= idx < TDAYS:
                            views[idx] = item["views"]
                    except:
                        pass
                return views
            elif r.status_code == 404:
                return np.zeros(TDAYS)
            elif r.status_code == 429:
                w = int(r.headers.get("Retry-After", 15))
                print(f"  rate-limited, waiting {w}s", flush=True)
                time.sleep(w + 1)
            else:
                time.sleep(2**attempt)
        except Exception as e:
            time.sleep(2**attempt)
    return np.zeros(TDAYS)

def main():
    print("="*60, flush=True)
    print(f"MILCCI Wikipedia EN Full ({TDAYS}d, {len(PAGES)} pages)", flush=True)
    print("="*60, flush=True)

    if os.path.exists(CACHE_EN_FULL):
        print("Loading cached data...", flush=True)
        data = dict(np.load(CACHE_EN_FULL, allow_pickle=True))
        Y = data['Y']
        labels = data['labels'].tolist()
        nt = data['numbers2tuples'].item()
        numbers2tuples = {int(k): tuple(v) for k,v in nt.items()}
        print(f"Y: {Y.shape}, trials: {len(labels)}", flush=True)
    else:
        # 5 trials: user/desktop, user/mobile-web, user/mobile-app, spider/desktop, spider/mobile-web
        trials = [("user","desktop"),("user","mobile-web"),("user","mobile-app"),
                  ("spider","desktop"),("spider","mobile-web")]
        M = len(trials)
        N = len(PAGES)
        print(f"Downloading {N}x{M}={N*M} series...", flush=True)

        Y = np.zeros((N, TDAYS, M), dtype=np.float64)
        for pi, page in enumerate(PAGES):
            t0 = time.time()
            n_ok = 0
            for m, (agent, plat) in enumerate(trials):
                views = download_one("en.wikipedia.org", plat, agent, page)
                if views.sum() > 0:
                    Y[pi,:,m] = views
                    n_ok += 1
                time.sleep(0.3)  # conservative rate limiting
            dt = time.time() - t0
            print(f"[{pi+1:2d}/{N}] {page}: {n_ok}/{M} ok, {dt:.1f}s", flush=True)

        # Preprocess
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

        # Labels: 2 categories (agent, platform)
        labels = list(range(M))
        numbers2tuples = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1)}

        nt_save = {str(k):list(v) for k,v in numbers2tuples.items()}
        np.savez_compressed(CACHE_EN_FULL, Y=Y, labels=np.array(labels),
                           numbers2tuples=nt_save)
        print("Cached.", flush=True)

    # Run MILCCI
    print(f"\n--- MILCCI (Y={Y.shape}, P=8) ---", flush=True)
    ns = np.count_nonzero(Y)
    print(f"Non-zero entries: {ns}/{Y.size} ({100*ns/Y.size:.1f}%)", flush=True)

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

    Phi = result['Phi']
    A_full = result['A_full']
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS (English, {TDAYS}d, {N} pages)", flush=True)
    print(f"{'='*60}", flush=True)
    for i, (a,p) in enumerate([("user","desktop"),("user","mobile-web"),
                                ("user","mobile-app"),("spider","desktop"),
                                ("spider","mobile-web")]):
        print(f"  R2 trial {i} ({a:5s}/{p:10s}): {r2_vec[i]:.4f}", flush=True)
    print(f"Per-trial R2: mean={r2_mean:.4f}  std={np.std(r2_vec):.4f}", flush=True)
    print(f"               min={np.min(r2_vec):.4f}  max={np.max(r2_vec):.4f}", flush=True)
    print(f"Global R2:    {r2_global:.4f}", flush=True)
    print(f"Runtime:      {runtime:.2f}s", flush=True)
    print(f"\nPaper (full):    R2=0.69,  Runtime=31.12s", flush=True)
    print(f"Ours (EN only):  R2={r2_mean:.4f},  Runtime={runtime:.2f}s", flush=True)

    results = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": float(np.std(r2_vec)),
        "r2_global": r2_global,
        "runtime_seconds": runtime,
        "Y_shape": list(Y.shape),
        "n_pages": N, "n_timepoints": TDAYS, "n_trials": Y.shape[2],
        "language": "en only",
    }
    with open(os.path.join(CACHE_DIR, "en_full_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved.", flush=True)

    return r2_mean, runtime

if __name__ == '__main__':
    main()
