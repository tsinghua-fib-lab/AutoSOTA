#!/usr/bin/env python3
"""
Fetch opinion texts for bankruptcy fraud violations from CourtListener API
"""

import json
import requests
import re
import time
from typing import Dict, List, Optional
import os

# API configuration
API_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN", "YOUR_TOKEN_HERE")
BASE_URL = "https://www.courtlistener.com/api/rest/v4"

def fetch_opinion_text(cluster_id: int) -> Optional[str]:
    """
    Fetch the full opinion text using the cluster ID

    Args:
        cluster_id: The cluster ID from search results

    Returns:
        The plain text of the opinion, or None if error
    """
    # First fetch the cluster to get sub_opinions
    url = f"{BASE_URL}/clusters/{cluster_id}/"
    headers = {
        "Authorization": f"Token {API_TOKEN}"
    }

    params = {
        "fields": "sub_opinions,case_name"
    }

    try:
        print(f"  Fetching cluster {cluster_id}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        cluster_data = response.json()

        # Get the opinion URLs
        opinions = cluster_data.get('sub_opinions', [])
        if not opinions:
            print(f"    No opinions found for cluster {cluster_id}")
            return None

        # Extract opinion ID from the first opinion URL
        first_opinion_url = opinions[0]
        opinion_id_match = re.search(r'/opinions/(\d+)/', first_opinion_url)

        if not opinion_id_match:
            print(f"    Could not extract opinion ID from {first_opinion_url}")
            return None

        actual_opinion_id = opinion_id_match.group(1)
        print(f"    Fetching opinion {actual_opinion_id}...")

        # Now fetch the actual opinion text
        opinion_url = f"{BASE_URL}/opinions/{actual_opinion_id}/"
        params = {
            "fields": "plain_text"
        }

        response = requests.get(opinion_url, headers=headers, params=params)
        response.raise_for_status()
        opinion_data = response.json()

        return opinion_data.get('plain_text', '')

    except requests.exceptions.RequestException as e:
        print(f"    Error fetching opinion: {e}")
        return None

def main(num_cases: int = 100):
    """Main function to fetch opinion texts for first num_cases cases"""

    # List of config files to process
    config_files = ['usc_config_2.json', 'usc_config_3.json',
                    'usc_config_4.json', 'usc_config_5.json', 'usc_config_6.json']

    for config_file in config_files:
        print(f"\n{'#'*60}\nProcessing: {config_file}\n{'#'*60}")

        # Load USC codes from config file
        with open(config_file, 'r') as f:
            config = json.load(f)
        titles = config['titles']
        sections = config['sections']

        for title, section in zip(titles, sections):
            # Load the search results
            input_file = f"courtlistener_search_results/usc_{title}_{section}/results_{title}_U.S.C.{section}.json"
            output_file = f"opinion_texts/usc_{title}_{section}/opinions_{title}_U.S.C.{section}.json"

            # Check if output file already exists and has content
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        existing_data = json.load(f)
                        if existing_data.get("results"):
                            print(f"Output file already exists with {len(existing_data['results'])} results. Skipping {title} U.S.C. {section}...")
                            continue
                except (json.JSONDecodeError, IOError):
                    print("Output file exists but is invalid. Reprocessing...")

            print("Loading search results...")
            with open(input_file, 'r') as f:
                data = json.load(f)

            # Keep the original metadata
            output_data = {
                "metadata": data.get("metadata", {}),
                "results": []
            }

            # Process first num_cases cases
            cases_to_process = data.get("results", [])[:num_cases]

            print(f"\nProcessing {len(cases_to_process)} cases...")
            print("=" * 60)

            for idx, case in enumerate(cases_to_process, 1):
                print(f"\n[{idx}/{num_cases}] Processing: {case.get('case_name', 'Unknown')}")

                cluster_id = case.get('cluster_id')

                if not cluster_id:
                    print("  No cluster_id found, skipping...")
                    continue

                # Fetch the opinion text
                opinion_text = fetch_opinion_text(cluster_id)

                if opinion_text:
                    # Create the result entry with required fields
                    result = {
                        "case_name": case.get('case_name', ''),
                        "frontend_url": case.get('frontend_url', ''),
                        "opinion_text": opinion_text
                    }

                    output_data["results"].append(result)
                    print(f"  ✓ Successfully fetched opinion ({len(opinion_text)} characters)")
                else:
                    print(f"  ✗ Failed to fetch opinion text")

                # Rate limiting between requests
                if idx < len(cases_to_process):
                    time.sleep(1)

            # Save the results
            print("\n" + "=" * 60)
            print(f"Saving results to: {output_file}")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Print summary
            print(f"\nSummary:")
            print(f"  Total cases processed: {len(cases_to_process)}")
            print(f"  Successfully fetched: {len(output_data['results'])}")
            print(f"  Failed: {len(cases_to_process) - len(output_data['results'])}")

            # Show brief info about each fetched case
            print(f"\nFetched opinions:")
            for idx, result in enumerate(output_data['results'], 1):
                text_len = len(result['opinion_text'])
                print(f"  {idx}. {result['case_name'][:50]}: {text_len:,} characters")

if __name__ == "__main__":
    main(num_cases=100)