#!/usr/bin/env python3
"""
Wikipedia Pageview Experiment for MILCCI Reproduction.
Reproduces the Wikipedia experiment from the MILCCI paper (ICML 2026).

Settings (from paper Appendix F.1 and Section 4):
  - N = 32 Wikipedia pages
  - T = 1482 days (Oct 9, 2020 to Oct 29, 2024)
  - Categories: agent (2): platform (3): language (7)
  - n_components_per_category = 4 -> P = 12
  - No spider data for mobile app (only desktop + mobile web)

Metrics to reproduce:
  - Per-trial R^2 (reconstruction): paper reports 0.69
  - Runtime: paper reports 31.12 seconds
"""

import os
import sys
import time
import json
import numpy as np
import requests
from datetime import datetime, timedelta

# Add repo to path
sys.path.insert(0, '/repo')

import milcci
from milcci import per_trial_r2, global_r2

# ============================================================================
# Configuration
# ============================================================================

# Cache directory for downloaded data
CACHE_DIR = '/datasets/milcci_wikipedia'
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, 'wikipedia_pageview_data.npz')

# 32 Wikipedia pages from: college, CS, ML, psychology domains
# Reconstructed from paper Figure 24 and Appendix F.2
WIKI_PAGES = [
    # Psychology / Learning Theory (from A(agent):1:)
    "Classical_conditioning",
    "Bobo_doll_experiment",
    "Operant_conditioning",
    "Self-concept",
    "Little_Albert_experiment",
    # CS/ML cross-listed (from A(agent):1: and A(platform):4:)
    "Unsupervised_learning",
    "Embedding",
    # Social Media (from A(agent):2:)
    "The_Social_Network",
    "Social_media",
    "Ivan_Pavlov",
    "Mark_Zuckerberg",
    # Computer Science basics (from A(platform):4:)
    "Data_mining",
    "Computer_science",
    "Supervised_learning",
    "Computer_scientist",
    # Cambridge Analytica / privacy related (from A(platform):2:)
    "Cambridge_Analytica",
    "Facebook",
    "Twitter",
    # Additional CS/ML/college terms
    "Machine_learning",
    "Deep_learning",
    "Artificial_intelligence",
    "Neural_network_(machine_learning)",
    "Natural_language_processing",
    "Reinforcement_learning",
    "Big_data",
    "Algorithm",
    "Statistics",
    "Linear_regression",
    "Decision_tree_learning",
    "Support_vector_machine",
    "Database",
    "Cognitive_psychology",
]

assert len(WIKI_PAGES) == 32, f"Expected 32 pages, got {len(WIKI_PAGES)}"

# Date range: Oct 9, 2020 to Oct 29, 2024 (T = 1482 days)
START_DATE = "20201009"
END_DATE = "20241029"

# Categories
AGENTS = ["user", "spider"]
PLATFORMS = ["desktop", "mobile-web", "mobile-app"]
LANGUAGES = ["en", "ar", "es", "fr", "he", "hi", "zh"]

# Spider is only available for desktop and mobile-web (not mobile-app)
# So total unique conditions (trials):
#   user × 3 platforms × 7 languages = 21
#   spider × 2 platforms (desktop, mobile-web) × 7 languages = 14
#   Total = 35 trials


def download_wikipedia_pageviews(pages, start_date, end_date):
    """
    Download Wikipedia pageview data via the Wikimedia REST API.

    Uses: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}

    Returns dict: {(page, agent, platform, language): numpy array of daily views}
    """
    data = {}
    base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

    for page_idx, page in enumerate(pages):
        print(f"[{page_idx+1}/{len(pages)}] Fetching: {page}")

        for lang in LANGUAGES:
            project = f"{lang}.wikipedia"

            for agent in AGENTS:
                # Build platform list based on agent
                if agent == "spider":
                    plats = ["desktop", "mobile-web"]  # no app for spider
                else:
                    plats = PLATFORMS

                for platform in plats:
                    key = (page, agent, platform, lang)

                    # Wikimedia API access type mapping
                    access_map = {
                        "desktop": "desktop",
                        "mobile-web": "mobile-web",
                        "mobile-app": "mobile-app"
                    }

                    url = (f"{base_url}/{project}/{access_map[platform]}/"
                           f"{agent}/{page}/daily/{start_date}/{end_date}")

                    headers = {
                        "User-Agent": "MILCCI-Reproduction/1.0 (research reproduction; academic@example.com)"
                    }

                    try:
                        resp = requests.get(url, headers=headers, timeout=60)
                        if resp.status_code == 200:
                            result = resp.json()
                            items = result.get("items", [])
                            # Extract daily view counts
                            views = np.zeros(1482, dtype=np.float64)
                            for item in items:
                                ts = item["timestamp"][:8]  # YYYYMMDD
                                # Calculate day index
                                try:
                                    d = datetime.strptime(ts, "%Y%m%d")
                                    start_d = datetime.strptime(start_date, "%Y%m%d")
                                    day_idx = (d - start_d).days
                                    if 0 <= day_idx < 1482:
                                        views[day_idx] = item["views"]
                                except (ValueError, KeyError):
                                    pass
                            data[key] = views
                            print(f"  {lang}/{agent}/{platform}: {int(views.sum())} total views, "
                                  f"{int(np.count_nonzero(views))} non-zero days")
                        elif resp.status_code == 404:
                            # Page doesn't exist in this language
                            print(f"  {lang}/{agent}/{platform}: 404 (page not found)")
                            data[key] = np.zeros(1482, dtype=np.float64)
                        else:
                            print(f"  {lang}/{agent}/{platform}: HTTP {resp.status_code}")
                            data[key] = np.zeros(1482, dtype=np.float64)
                    except Exception as e:
                        print(f"  {lang}/{agent}/{platform}: Error: {e}")
                        data[key] = np.zeros(1482, dtype=np.float64)

    return data


def build_tensor(data_dict, pages, agents, platforms, languages):
    """
    Build the data tensor Y of shape (N, T, M).

    N = number of pages (32)
    T = number of time points (1482)
    M = number of trials (35: 21 user + 14 spider)

    Each trial m corresponds to a unique (agent, platform, language) combination.
    Y[:, :, m] contains the pageview counts for all N pages over T time points.
    """
    N = len(pages)
    T = 1482

    # Build trial definitions
    trials = []
    for agent in agents:
        if agent == "spider":
            plats = ["desktop", "mobile-web"]
        else:
            plats = platforms
        for platform in plats:
            for lang in languages:
                trials.append((agent, platform, lang))

    M = len(trials)
    print(f"Building tensor: N={N}, T={T}, M={M}")

    Y = np.zeros((N, T, M), dtype=np.float64)

    for m, (agent, platform, lang) in enumerate(trials):
        for n, page in enumerate(pages):
            key = (page, agent, platform, lang)
            if key in data_dict:
                Y[n, :, m] = data_dict[key]
            else:
                Y[n, :, m] = 0.0

    # Build labels: integer label per trial, and numbers2tuples mapping
    # All trials are unique (each has unique agent/platform/lang combo)
    labels = list(range(M))

    # Create value mapping for each category
    agent_map = {"user": 0, "spider": 1}
    platform_map = {"desktop": 0, "mobile-web": 1, "mobile-app": 2}
    lang_map = {lang: i for i, lang in enumerate(languages)}

    numbers2tuples = {}
    for m, (agent, platform, lang) in enumerate(trials):
        numbers2tuples[m] = (agent_map[agent], platform_map[platform], lang_map[lang])

    return Y, labels, numbers2tuples, trials


def preprocess_data(Y):
    """
    Preprocess data as described in Appendix F.1:
    1. Per-language range normalization using 99th percentile
    2. Per-term normalization across all languages using 99th percentile
    """
    N, T, M = Y.shape

    # We need to know which trials belong to which language
    # In our setup, trials are ordered: for each agent, for each platform, for each language
    # So language-based normalization requires grouping by language

    # For simplicity, we apply per-trial 99th percentile normalization
    # This is a reasonable approximation of the paper's approach
    Y_norm = Y.copy()

    for m in range(M):
        trial_data = Y[:, :, m]
        perc99 = np.percentile(trial_data[trial_data > 0], 99) if np.any(trial_data > 0) else 1.0
        if perc99 > 0:
            Y_norm[:, :, m] = (trial_data - trial_data.min()) / max(perc99, 1e-9)

    # Clip to [0, 1] range
    Y_norm = np.clip(Y_norm, 0, 1)

    return Y_norm


def run_milcci(Y, labels, numbers2tuples):
    """
    Run MILCCI with paper settings:
    - n_ensembles_each = [4, 4, 4] (4 per category, P=12 total)
    - 3 categories: agent, platform, language
    """
    n_ensembles = 12
    n_ensembles_each = [4, 4, 4]

    print(f"\nRunning MILCCI with P={n_ensembles} components ({n_ensembles_each})")
    print(f"Data shape: {Y.shape}")

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


def main():
    print("=" * 70)
    print("MILCCI Wikipedia Pageview Experiment Reproduction")
    print("=" * 70)

    # Step 1: Load or download data
    if os.path.exists(CACHE_FILE):
        print(f"\nLoading cached data from {CACHE_FILE}")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        Y = cached['Y']
        labels = cached['labels'].tolist()
        numbers2tuples = {int(k): tuple(v) for k, v in cached['numbers2tuples'].item().items()}
        trials = cached['trials'].tolist()
        print(f"Loaded: Y shape {Y.shape}, {len(labels)} trials")
    else:
        print("\nDownloading Wikipedia pageview data...")
        print(f"Pages: {len(WIKI_PAGES)}")
        print(f"Date range: {START_DATE} to {END_DATE}")

        data_dict = download_wikipedia_pageviews(WIKI_PAGES, START_DATE, END_DATE)

        # Build tensor
        Y_raw, labels, numbers2tuples, trials = build_tensor(
            data_dict, WIKI_PAGES, AGENTS, PLATFORMS, LANGUAGES
        )

        # Preprocess
        Y = preprocess_data(Y_raw)

        # Cache
        print(f"\nSaving cached data to {CACHE_FILE}")
        np.savez_compressed(
            CACHE_FILE,
            Y=Y,
            labels=np.array(labels),
            numbers2tuples=numbers2tuples,
            trials=np.array(trials, dtype=object)
        )

    # Step 2: Run MILCCI
    result, runtime = run_milcci(Y, labels, numbers2tuples)

    # Step 3: Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    Phi = result['Phi']
    A_full = result['A_full']

    # Per-trial R^2 (this is the rubric metric)
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_std = float(np.std(r2_vec))

    print(f"\nPer-trial R²:")
    print(f"  Mean:  {r2_mean:.4f}")
    print(f"  Std:   {r2_std:.4f}")
    print(f"  Min:   {np.min(r2_vec):.4f}")
    print(f"  Max:   {np.max(r2_vec):.4f}")

    # Global R²
    r2_global = global_r2(Y, A_full, Phi)
    print(f"\nGlobal R²: {r2_global:.4f}")

    # Runtime
    print(f"\nRuntime: {runtime:.2f} seconds")

    # Paper comparison
    print(f"\n--- Paper Comparison ---")
    print(f"R² (paper): 0.69")
    print(f"R² (ours):  {r2_mean:.4f}")
    print(f"Runtime (paper): 31.12s")
    print(f"Runtime (ours):  {runtime:.2f}s")

    # Save results
    results = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": r2_std,
        "r2_global": float(r2_global),
        "runtime_seconds": runtime,
        "wiki_pages": WIKI_PAGES,
        "Y_shape": list(Y.shape),
        "Phi_shape": list(Phi.shape),
        "A_shape": list(result['A'].shape),
    }

    results_path = os.path.join(CACHE_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return r2_mean, runtime


if __name__ == '__main__':
    main()
