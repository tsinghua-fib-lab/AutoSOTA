#!/usr/bin/env python3
"""MILCCI Wikipedia Experiment - single self-contained script."""
import os, sys, time, json, hashlib
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, '/repo')
import milcci
from milcci import per_trial_r2, global_r2

CACHE_DIR = '/datasets/milcci_wikipedia'
os.makedirs(CACHE_DIR, exist_ok=True)
STATUS_FILE = os.path.join(CACHE_DIR, 'status.txt')

def log(msg):
    with open(STATUS_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

# 32 pages
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

LANGS = {"en":"en.wikipedia.org","ar":"ar.wikipedia.org","es":"es.wikipedia.org",
         "fr":"fr.wikipedia.org","he":"he.wikipedia.org","hi":"hi.wikipedia.org",
         "zh":"zh.wikipedia.org"}

AGENTS = ["user","spider"]
PLAT_USER = ["desktop","mobile-web","mobile-app"]
PLAT_SPIDER = ["desktop","mobile-web"]

START, END = "20201009", "20241029"
TDAYS = 1482
HEADERS = {"User-Agent": "MILCCI-Reproduction/1.0 (academic@example.com)"}

CACHE_NPZ = os.path.join(CACHE_DIR, "wiki_full.npz")

def main():
    log("="*60)
    log("MILCCI Wikipedia Experiment")
    log("="*60)

    # Check cache
    if os.path.exists(CACHE_NPZ):
        log("Loading cached data...")
        data = dict(np.load(CACHE_NPZ, allow_pickle=True))
        Y = data['Y']
        labels = data['labels'].tolist()
        nt = data['numbers2tuples'].item()
        numbers2tuples = {int(k): tuple(v) for k, v in nt.items()}
        log(f"Loaded: Y shape {Y.shape}, {len(labels)} trials")
    else:
        # Build trials
        trials = []
        for agent in AGENTS:
            plats = PLAT_USER if agent == "user" else PLAT_SPIDER
            for plat in plats:
                for lc in LANGS:
                    trials.append((agent, plat, lc))
        M = len(trials)
        N = len(PAGES)
        log(f"Downloading {N} pages x {M} trials")

        # Get langlinks for all pages first (fast)
        lang_titles = {}
        for pi, page in enumerate(PAGES):
            titles = {"en": page}
            try:
                params = {"action":"query","titles":page,
                         "prop":"langlinks","lllimit":50,"format":"json"}
                r = requests.get("https://en.wikipedia.org/w/api.php",
                               params=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    for info in r.json().get("query",{}).get("pages",{}).values():
                        for ll in info.get("langlinks",[]):
                            if ll["lang"] in LANGS:
                                titles[ll["lang"]] = ll["*"]
                time.sleep(0.1)
            except Exception as e:
                log(f"  langlinks {page}: {e}")
            lang_titles[page] = titles
        log(f"Langlinks done: {len(lang_titles)} pages")

        # Download all data
        Y = np.zeros((N, TDAYS, M), dtype=np.float64)
        sd = datetime.strptime(START, "%Y%m%d")
        total_ok, total_zero = 0, 0

        for pi, page in enumerate(PAGES):
            t0 = time.time()
            titles = lang_titles[page]
            n_ok, n_zero = 0, 0

            for m, (agent, plat, lc) in enumerate(trials):
                if lc not in titles:
                    n_zero += 1
                    continue
                proj = LANGS[lc]
                art = titles[lc]
                ok = False
                for attempt in range(5):
                    try:
                        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                               f"per-article/{proj}/{plat}/{agent}/{art}/"
                               f"daily/{START}/{END}")
                        r = requests.get(url, headers=HEADERS, timeout=120)
                        if r.status_code == 200:
                            for item in r.json().get("items",[]):
                                try:
                                    d = datetime.strptime(item["timestamp"][:8],"%Y%m%d")
                                    idx = (d-sd).days
                                    if 0 <= idx < TDAYS:
                                        Y[pi,idx,m] = item["views"]
                                except:
                                    pass
                            ok = True
                            break
                        elif r.status_code == 404:
                            break
                        elif r.status_code == 429:
                            time.sleep(int(r.headers.get("Retry-After",10)))
                        else:
                            time.sleep(2**attempt)
                    except:
                        time.sleep(2**attempt)
                if ok:
                    n_ok += 1
                else:
                    n_zero += 1
                time.sleep(0.12)

            total_ok += n_ok
            total_zero += n_zero
            dt = time.time() - t0
            log(f"[{pi+1:2d}/{N}] {page}: {n_ok} ok, {n_zero} zero, {dt:.1f}s")

        log(f"Download done: {total_ok} ok, {total_zero} zero")

        # Build labels
        ag_map = {"user":0,"spider":1}
        pl_map = {"desktop":0,"mobile-web":1,"mobile-app":2}
        lg_map = {lc:i for i,lc in enumerate(LANGS.keys())}
        labels = list(range(M))
        numbers2tuples = {m: (ag_map[a],pl_map[p],lg_map[l])
                         for m,(a,p,l) in enumerate(trials)}

        # Preprocess: 99th percentile normalization
        log("Preprocessing...")
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

        # Cache
        nt_save = {str(k): list(v) for k,v in numbers2tuples.items()}
        np.savez_compressed(CACHE_NPZ, Y=Y, labels=np.array(labels),
                           numbers2tuples=nt_save)
        log("Data cached.")

    # Run MILCCI
    log("\n--- Running MILCCI ---")
    log(f"Y shape: {Y.shape}, non-zero: {np.count_nonzero(Y)}/{Y.size}")

    t0 = time.time()
    result = milcci.fit(
        data=Y, labels=labels, numbers2tuples=numbers2tuples,
        n_ensembles=12, n_ensembles_each=[4,4,4],
        nu=[0.01]*12, lambda_similarity=100, factor_A=5,
        decor_A=2, num_repeats=15, cont_axis_list=[],
        split_A=True, params_init_A={'ensemble_positive': True},
        verbose=True, seed=42,
    )
    runtime = time.time() - t0

    # Evaluate
    Phi = result['Phi']
    A_full = result['A_full']
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_std = float(np.std(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    log(f"\n{'='*60}")
    log(f"RESULTS")
    log(f"{'='*60}")
    log(f"Per-trial R2:  mean={r2_mean:.4f}  std={r2_std:.4f}")
    log(f"               min={np.min(r2_vec):.4f}  max={np.max(r2_vec):.4f}")
    log(f"Global R2:     {r2_global:.4f}")
    log(f"Runtime:       {runtime:.2f}s")
    log(f"\nPaper: R2=0.69, Runtime=31.12s")
    log(f"Ours:  R2={r2_mean:.4f}, Runtime={runtime:.2f}s")

    # Save
    results = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": r2_std,
        "r2_global": r2_global,
        "runtime_seconds": runtime,
        "Y_shape": list(Y.shape),
    }
    with open(os.path.join(CACHE_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return r2_mean, runtime

if __name__ == '__main__':
    main()
