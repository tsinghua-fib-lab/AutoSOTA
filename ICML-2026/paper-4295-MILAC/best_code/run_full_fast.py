#!/usr/bin/env python3
"""MILCCI Wikipedia Full Experiment - Fast Download."""
import os, sys, time, json
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, "/repo")
import milcci
from milcci import per_trial_r2, global_r2

CACHE_DIR = "/datasets/milcci_wikipedia"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FULL = os.path.join(CACHE_DIR, "wiki_full_35trial.npz")

START, END = "20201009", "20241029"
TDAYS = 1482
SEP = "=" * 60

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
][:32]

LANGS = {"en":"en.wikipedia.org","ar":"ar.wikipedia.org","es":"es.wikipedia.org",
         "fr":"fr.wikipedia.org","he":"he.wikipedia.org","hi":"hi.wikipedia.org",
         "zh":"zh.wikipedia.org"}

AGENTS = ["user","spider"]
PLAT_USER = ["desktop","mobile-web","mobile-app"]
PLAT_SPIDER = ["desktop","mobile-web"]
HEADERS = {"User-Agent": "MILCCI-Reproduction/1.0 (academic@example.com)"}

SD = datetime.strptime(START, "%Y%m%d")

def fetch_pageviews(project, access, agent, article):
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
           f"per-article/{project}/{access}/{agent}/{article}/"
           f"daily/{START}/{END}")
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                views = np.zeros(TDAYS, dtype=np.float64)
                for item in r.json().get("items", []):
                    try:
                        d = datetime.strptime(item["timestamp"][:8], "%Y%m%d")
                        idx = (d - SD).days
                        if 0 <= idx < TDAYS:
                            views[idx] = item["views"]
                    except:
                        pass
                return views
            elif r.status_code == 404:
                return None
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 5)) + 1)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def get_langlinks_batch(pages):
    titles_map = {}
    for i in range(0, len(pages), 20):
        batch = pages[i:i+20]
        pipe = "|".join(batch)
        params = {"action": "query", "titles": pipe,
                 "prop": "langlinks", "lllimit": 50, "format": "json"}
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php",
                           params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                for pid, info in r.json().get("query", {}).get("pages", {}).items():
                    pt = info.get("title", "").replace(" ", "_")
                    titles_map[pt] = {"en": pt}
                    for ll in info.get("langlinks", []):
                        if ll["lang"] in LANGS:
                            titles_map[pt][ll["lang"]] = ll["*"]
            time.sleep(0.15)
        except Exception as e:
            print("  langlinks error: " + str(e), flush=True)
    return titles_map

def main():
    print(SEP, flush=True)
    print("MILCCI Wikipedia Full Experiment", flush=True)
    print("  %d days, %d pages, 7 languages" % (TDAYS, len(PAGES)), flush=True)
    print(SEP, flush=True)

    if os.path.exists(CACHE_FULL):
        print("Loading cached full data...", flush=True)
        data = dict(np.load(CACHE_FULL, allow_pickle=True))
        Y = data["Y"]
        labels = data["labels"].tolist()
        nt = data["numbers2tuples"].item()
        numbers2tuples = {int(k): tuple(v) for k, v in nt.items()}
        print("  Y: %s, trials: %d" % (str(Y.shape), len(labels)), flush=True)
    else:
        trials_list = []
        for agent in AGENTS:
            plats = PLAT_USER if agent == "user" else PLAT_SPIDER
            for plat in plats:
                for lc in LANGS:
                    trials_list.append((agent, plat, lc))
        M = len(trials_list)
        N = len(PAGES)

        print("Fetching language links...", flush=True)
        lang_titles = get_langlinks_batch(PAGES)
        total_translations = sum(len(v) for v in lang_titles.values())
        print("  Done: %d translations for %d pages" % (total_translations, len(lang_titles)), flush=True)

        Y = np.zeros((N, TDAYS, M), dtype=np.float64)
        total_ok = 0
        t_start = time.time()

        for pi, page in enumerate(PAGES):
            t_p0 = time.time()
            titles = lang_titles.get(page, {"en": page})
            n_ok = 0
            for m, (agent, plat, lc) in enumerate(trials_list):
                if lc not in titles:
                    continue
                proj = LANGS[lc]
                art = titles[lc]
                views = fetch_pageviews(proj, plat, agent, art)
                if views is not None and views.sum() > 0:
                    Y[pi, :, m] = views
                    n_ok += 1
                time.sleep(0.02)

            total_ok += n_ok
            dt = time.time() - t_p0
            elapsed = time.time() - t_start
            eta = (elapsed / (pi + 1)) * (N - pi - 1) if pi < N - 1 else 0
            print("[%2d/%d] %s: %d/%d ok, %.1fs, total_ok=%d, ETA=%.0fs" % (
                pi+1, N, page, n_ok, M, dt, total_ok, eta), flush=True)

        print("\nTotal ok: %d/%d (%.1f%%)" % (total_ok, N*M, 100*total_ok/(N*M)), flush=True)

        # Preprocess
        print("Preprocessing...", flush=True)
        for m in range(M):
            trial = Y[:, :, m]
            pos = trial[trial > 0]
            if len(pos) > 0:
                p99 = np.percentile(pos, 99)
                if p99 > 0:
                    Y[:, :, m] = np.clip(trial / p99, 0, 1)
            Y[:, :, m] -= Y[:, :, m].min()
        for n in range(N):
            td = Y[n, :, :]
            pos = td[td > 0]
            if len(pos) > 0:
                p99 = np.percentile(pos, 99)
                if p99 > 0:
                    Y[n, :, :] = np.clip(td / p99, 0, 1)

        ag_map = {"user": 0, "spider": 1}
        pl_map = {"desktop": 0, "mobile-web": 1, "mobile-app": 2}
        lg_map = {lc: i for i, lc in enumerate(LANGS.keys())}
        labels = list(range(M))
        numbers2tuples = {m: (ag_map[a], pl_map[p], lg_map[l])
                         for m, (a, p, l) in enumerate(trials_list)}

        nt_save = {str(k): list(v) for k, v in numbers2tuples.items()}
        np.savez_compressed(CACHE_FULL, Y=Y, labels=np.array(labels),
                           numbers2tuples=nt_save)
        print("Data cached to " + CACHE_FULL, flush=True)

    # Run MILCCI
    M = Y.shape[2]
    print("\n--- MILCCI (Y: %s, P=12) ---" % str(Y.shape), flush=True)
    print("Non-zero: %d/%d (%.1f%%)" % (
        np.count_nonzero(Y), Y.size, 100*np.count_nonzero(Y)/Y.size), flush=True)

    t0 = time.time()
    result = milcci.fit(
        data=Y, labels=labels, numbers2tuples=numbers2tuples,
        n_ensembles=12, n_ensembles_each=[4, 4, 4],
        nu=[0.01] * 12, lambda_similarity=100, factor_A=5,
        decor_A=2, num_repeats=15,
        cont_axis_list=[], split_A=True,
        params_init_A={"ensemble_positive": True},
        verbose=True, seed=42,
    )
    runtime = time.time() - t0

    Phi = result["Phi"]
    A_full = result["A_full"]
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_std = float(np.std(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    print("", flush=True)
    print(SEP, flush=True)
    print("FULL EXPERIMENT RESULTS", flush=True)
    print(SEP, flush=True)
    print("Per-trial R2:  mean=%.4f  std=%.4f" % (r2_mean, r2_std), flush=True)
    print("               min=%.4f  max=%.4f" % (np.min(r2_vec), np.max(r2_vec)), flush=True)
    print("Global R2:     %.4f" % r2_global, flush=True)
    print("Runtime:       %.2fs" % runtime, flush=True)
    print("", flush=True)
    print("Paper: R2=0.69, Runtime=31.12s", flush=True)
    print("Ours:  R2=%.4f, Runtime=%.2fs" % (r2_mean, runtime), flush=True)

    results = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": r2_std,
        "r2_global": r2_global,
        "runtime_seconds": runtime,
        "Y_shape": list(Y.shape),
        "n_pages": 32, "n_timepoints": TDAYS, "n_trials": int(Y.shape[2]),
        "paper_r2": 0.69, "paper_runtime": 31.12,
    }
    with open(os.path.join(CACHE_DIR, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved.", flush=True)

    return r2_mean, runtime

if __name__ == "__main__":
    main()
