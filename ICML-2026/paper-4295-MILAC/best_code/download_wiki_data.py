#!/usr/bin/env python3
"""
Download Wikipedia pageview data for MILCCI reproduction.
Downloads data page by page, caching results to avoid re-downloading.
"""

import os
import sys
import time
import json
import gzip
import numpy as np
import requests
from datetime import datetime, timedelta

sys.path.insert(0, '/repo')

CACHE_DIR = '/datasets/milcci_wikipedia'
os.makedirs(CACHE_DIR, exist_ok=True)

# Configuration
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

START_DATE = "20201009"
END_DATE = "20241029"
T_DAYS = 1482

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
PLATFORMS_ALL = ["desktop", "mobile-web", "mobile-app"]
SPIDER_PLATFORMS = ["desktop", "mobile-web"]  # no app for spider

HEADERS = {"User-Agent": "MILCCI-Reproduction/1.0 (research@example.com)"}


def get_interlanguage_titles(en_title):
    """Get the page title in other languages using the Wikipedia API."""
    cache_file = os.path.join(CACHE_DIR, f"langlinks_{en_title.replace('/', '_')}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": en_title,
        "prop": "langlinks",
        "lllimit": 50,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            titles = {"en": en_title}
            for page_id, page_info in pages.items():
                langlinks = page_info.get("langlinks", [])
                for ll in langlinks:
                    if ll["lang"] in LANGUAGES:
                        titles[ll["lang"]] = ll["*"]
            with open(cache_file, 'w') as f:
                json.dump(titles, f, indent=2)
            time.sleep(0.1)
            return titles
    except Exception as e:
        print(f"  Error getting langlinks: {e}")
    return {"en": en_title}


def fetch_pageviews(project, access, agent, article, start_date, end_date):
    """Fetch pageview data for a specific article and access type."""
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
           f"per-article/{project}/{access}/{agent}/{article}/"
           f"daily/{start_date}/{end_date}")

    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                result = r.json()
                views = np.zeros(T_DAYS, dtype=np.float64)
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                for item in result.get("items", []):
                    ts = item["timestamp"][:8]
                    try:
                        d = datetime.strptime(ts, "%Y%m%d")
                        day_idx = (d - start_dt).days
                        if 0 <= day_idx < T_DAYS:
                            views[day_idx] = item["views"]
                    except ValueError:
                        pass
                return views
            elif r.status_code == 404:
                return None  # no data
            elif r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 5))
                print(f"    Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
            else:
                print(f"    HTTP {r.status_code}, retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"    Error: {e}, retrying...")
            time.sleep(2)
    return None


def main():
    print("=" * 70)
    print("Wikipedia Pageview Data Download")
    print("=" * 70)

    all_data = {}

    for pi, page in enumerate(WIKI_PAGES):
        print(f"\n[{pi+1}/{len(WIKI_PAGES)}] {page}")

        # Get interlanguage titles
        titles = get_interlanguage_titles(page)
        print(f"  Available in: {list(titles.keys())}")

        for lang_code, project in LANGUAGES.items():
            if lang_code not in titles:
                continue

            article = titles[lang_code]

            for agent in AGENTS:
                if agent == "spider":
                    platforms = SPIDER_PLATFORMS
                else:
                    platforms = PLATFORMS_ALL

                for platform in platforms:
                    key = (page, agent, platform, lang_code)

                    # Check cache
                    cache_file = os.path.join(
                        CACHE_DIR,
                        f"pv_{page}_{agent}_{platform}_{lang_code}.npz"
                    )
                    if os.path.exists(cache_file):
                        cached = np.load(cache_file)
                        views = cached['views']
                        all_data[key] = views
                        if views.sum() > 0:
                            print(f"  {lang_code}/{agent}/{platform}: cached, "
                                  f"{int(views.sum())} views")
                        continue

                    views = fetch_pageviews(
                        project, platform, agent, article,
                        START_DATE, END_DATE
                    )

                    if views is not None:
                        all_data[key] = views
                        if views.sum() > 0:
                            print(f"  {lang_code}/{agent}/{platform}: "
                                  f"{int(views.sum())} views")
                        # Save cache
                        np.savez_compressed(cache_file, views=views)
                    else:
                        all_data[key] = np.zeros(T_DAYS, dtype=np.float64)
                        # Cache empty results too
                        np.savez_compressed(cache_file,
                                          views=np.zeros(T_DAYS, dtype=np.float64))
                        print(f"  {lang_code}/{agent}/{platform}: no data")

                    # Rate limiting
                    time.sleep(0.15)

        # Save intermediate results
        print(f"  Saving progress...")
        progress = {
            'pages_processed': pi + 1,
            'total_pages': len(WIKI_PAGES),
            'keys_collected': len(all_data),
        }
        with open(os.path.join(CACHE_DIR, 'download_progress.json'), 'w') as f:
            json.dump(progress, f)

    print(f"\n{'=' * 70}")
    print(f"Download complete: {len(all_data)} data entries collected")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
