#!/usr/bin/env python3
"""
MILCCI Wikipedia Pageview Experiment Reproduction.
Reproduces the Wikipedia experiment from ICML 2026 paper.
"""

import os, sys, time, json, hashlib, warnings
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, '/repo')
import milcci
from milcci import per_trial_r2, global_r2

# ==========================================================================
# Configuration
# ==========================================================================
CACHE_DIR = '/datasets/milcci_wikipedia'
os.makedirs(CACHE_DIR, exist_ok=True)

START_DATE = "20201009"
END_DATE   = "20241029"
T_DAYS = 1482

# 32 Wikipedia pages from paper
WIKI_PAGES = [
    "Classical_conditioning", "Bobo_doll_experiment", "Operant_conditioning",
    "Self-concept", "Little_Albert_experiment", "Unsupervised_learning",
    "Embedding", "The_Social_Network", "Social_media", "Ivan_Pavlov",
    "Mark_Zuckerberg", "Data_mining", "Computer_science", "Supervised_learning",
    "Computer_scientist", "Cambridge_Analytica", "Facebook", "Twitter",
    "Machine_learning", "Deep_learning", "Artificial_intelligence",
    "Neural_network_(machine_learning)", "Natural_language_processing",
    "Reinforcement_learning", "Big_data", "Algorithm", "Statistics",
    "Linear_regression", "Decision_tree_learning", "Support_vector_machine",
    "Database", "Cognitive_psychology",
]

LANGUAGES = {
    "en": "en.wikipedia.org",
    "ar": "ar.wikipedia.org",
    "es": "es.wikipedia.org",
    "fr": "fr.wikipedia.org",
    "he": "he.wikipedia.org",
    "hi": "hi.wikipedia.org",
    "zh": "zh.wikipedia.org",
}

AGENTS = ["user", "spider"]
PLATFORMS_USER = ["desktop", "mobile-web", "mobile-app"]
PLATFORMS_SPIDER = ["desktop", "mobile-web"]

HEADERS = {"User-Agent": "MILCCI-Reproduction/1.0 (academic@example.com)"}

# ==========================================================================
# Cached data loading/saving
# ==========================================================================
CACHE_NPZ = os.path.join(CACHE_DIR, "wiki_data.npz")
CACHE_JSON = os.path.join(CACHE_DIR, "wiki_meta.json")


def load_cached_data():
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON):
        print("Loading cached data from disk...")
        data = dict(np.load(CACHE_NPZ, allow_pickle=True))
        Y = data['Y']
        labels = data['labels'].tolist()
        with open(CACHE_JSON) as f:
            meta = json.load(f)
        numbers2tuples = {int(k): tuple(v) for k, v in meta['numbers2tuples'].items()}
        return Y, labels, numbers2tuples
    return None, None, None


def save_cached_data(Y, labels, numbers2tuples):
    np.savez_compressed(CACHE_NPZ, Y=Y, labels=np.array(labels))
    meta = {'numbers2tuples': {str(k): list(v) for k, v in numbers2tuples.items()}}
    with open(CACHE_JSON, 'w') as f:
        json.dump(meta, f)
    print(f"Data cached to {CACHE_NPZ}")


# ==========================================================================
# Data download
# ==========================================================================
def fetch_pageviews(project, access, agent, article):
    """Download pageview time series from Wikimedia REST API."""
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
           f"per-article/{project}/{access}/{agent}/{article}/"
           f"daily/{START_DATE}/{END_DATE}")
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                views = np.zeros(T_DAYS, dtype=np.float64)
                sd = datetime.strptime(START_DATE, "%Y%m%d")
                for item in r.json().get("items", []):
                    try:
                        d = datetime.strptime(item["timestamp"][:8], "%Y%m%d")
                        idx = (d - sd).days
                        if 0 <= idx < T_DAYS:
                            views[idx] = item["views"]
                    except (ValueError, KeyError):
                        pass
                return views
            elif r.status_code == 404:
                return None
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"    rate limited, wait {wait}s")
                time.sleep(wait)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    err: {e}")
            time.sleep(2 ** attempt)
    return None


def download_all_data():
    """Download all pageview data and build the tensor."""
    # Build trial list
    trials = []
    for agent in AGENTS:
        plats = PLATFORMS_USER if agent == "user" else PLATFORMS_SPIDER
        for platform in plats:
            for lang in LANGUAGES:
                trials.append((agent, platform, lang))
    M = len(trials)
    N = len(WIKI_PAGES)

    print(f"Downloading: {N} pages x {M} trials = {N*M} data series")
    print(f"Date range: {START_DATE} to {END_DATE} ({T_DAYS} days)")

    Y = np.zeros((N, T_DAYS, M), dtype=np.float64)  # will hold NaNs for missing

    # Cache for language titles per page
    lang_titles = {}

    for pi, page in enumerate(WIKI_PAGES):
        t_page_start = time.time()
        print(f"\n[{pi+1:2d}/{N}] {page}", end="", flush=True)

        # Get interlanguage titles (cached)
        if page not in lang_titles:
            titles = {"en": page}
            try:
                params = {"action": "query", "titles": page,
                         "prop": "langlinks", "lllimit": 50, "format": "json"}
                r = requests.get("https://en.wikipedia.org/w/api.php",
                               params=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    for info in r.json().get("query", {}).get("pages", {}).values():
                        for ll in info.get("langlinks", []):
                            if ll["lang"] in LANGUAGES:
                                titles[ll["lang"]] = ll["*"]
                time.sleep(0.1)
            except Exception as e:
                print(f" (langlinks: {e})", end="", flush=True)
            lang_titles[page] = titles

        titles = lang_titles[page]
        n_fetched = 0
        n_zeros = 0

        for m, (agent, platform, lang) in enumerate(trials):
            if lang not in titles:
                n_zeros += 1
                continue

            project = LANGUAGES[lang]
            article = titles[lang]
            views = fetch_pageviews(project, platform, agent, article)

            if views is not None and views.sum() > 0:
                Y[pi, :, m] = views
                n_fetched += 1
            else:
                n_zeros += 1

            time.sleep(0.12)  # rate limiting

        dt = time.time() - t_page_start
        print(f" -> {n_fetched} ok, {n_zeros} zero, {dt:.1f}s", flush=True)

    # Build labels
    agent_map = {"user": 0, "spider": 1}
    platform_map = {"desktop": 0, "mobile-web": 1, "mobile-app": 2}
    lang_map = {lc: i for i, lc in enumerate(LANGUAGES.keys())}
    labels = list(range(M))
    numbers2tuples = {m: (agent_map[a], platform_map[p], lang_map[l])
                      for m, (a, p, l) in enumerate(trials)}

    return Y, labels, numbers2tuples


# ==========================================================================
# Preprocessing
# ==========================================================================
def preprocess(Y):
    """Normalize data as described in Appendix F.1."""
    N, T, M = Y.shape
    Yn = Y.copy()

    # Per-trial 99th percentile normalization
    for m in range(M):
        trial = Yn[:, :, m]
        pos = trial[trial > 0]
        if len(pos) > 0:
            p99 = np.percentile(pos, 99)
            if p99 > 0:
                Yn[:, :, m] = np.clip(trial / p99, 0, 1)

    # Per-term 99th percentile normalization
    for n in range(N):
        td = Yn[n, :, :]
        pos = td[td > 0]
        if len(pos) > 0:
            p99 = np.percentile(pos, 99)
            if p99 > 0:
                Yn[n, :, :] = np.clip(td / p99, 0, 1)

    return Yn


# ==========================================================================
# Run MILCCI
# ==========================================================================
def run_milcci(Y, labels, numbers2tuples):
    n_ensembles = 12
    n_ensembles_each = [4, 4, 4]

    print(f"\n--- Running MILCCI ---")
    print(f"  Data: {Y.shape}, non-zero: {np.count_nonzero(Y)}/{Y.size} "
          f"({100*np.count_nonzero(Y)/Y.size:.1f}%)")
    print(f"  Components: {n_ensembles} = {n_ensembles_each}")
    print(f"  Conditions: {len(set(labels))}")

    t0 = time.time()
    result = milcci.fit(
        data=Y,
        labels=labels,
        numbers2tuples=numbers2tuples,
        n_ensembles=n_ensembles,
        n_ensembles_each=n_ensembles_each,
        nu=[0.01] * n_ensembles,
        lambda_similarity=100,
        factor_A=5,
        decor_A=2,
        num_repeats=15,
        cont_axis_list=[],
        split_A=True,
        params_init_A={'ensemble_positive': True},
        verbose=True,
        seed=42,
    )
    runtime = time.time() - t0
    return result, runtime


# ==========================================================================
def main():
    print("=" * 70)
    print("MILCCI Wikipedia Pageview Experiment")
    print("=" * 70)

    # 1. Load or download data
    Y, labels, numbers2tuples = load_cached_data()

    if Y is None:
        print("\n--- Downloading data ---")
        Y, labels, numbers2tuples = download_all_data()
        print("\n--- Preprocessing ---")
        Y = preprocess(Y)
        save_cached_data(Y, labels, numbers2tuples)
    else:
        print(f"Loaded: Y shape {Y.shape}, {len(labels)} trials")

    # 2. Run MILCCI
    result, runtime = run_milcci(Y, labels, numbers2tuples)

    # 3. Evaluate
    Phi = result['Phi']
    A_full = result['A_full']

    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Per-trial R2:  mean={r2_mean:.4f}  std={np.std(r2_vec):.4f}")
    print(f"               min={np.min(r2_vec):.4f}  max={np.max(r2_vec):.4f}")
    print(f"Global R2:     {r2_global:.4f}")
    print(f"Runtime:       {runtime:.2f}s")
    print(f"\nPaper comparison:")
    print(f"  R2 (paper):       0.69")
    print(f"  R2 (ours):        {r2_mean:.4f}")
    print(f"  Runtime (paper):  31.12s")
    print(f"  Runtime (ours):   {runtime:.2f}s")

    # Save results
    results = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": float(np.std(r2_vec)),
        "r2_global": r2_global,
        "runtime_seconds": runtime,
        "Y_shape": list(Y.shape),
    }
    with open(os.path.join(CACHE_DIR, "wiki_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return r2_mean, runtime


if __name__ == '__main__':
    main()
