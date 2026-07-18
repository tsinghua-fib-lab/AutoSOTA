#!/usr/bin/env python3
"""
Script to extract all unique superdomains and domains
from one or more JSONL files and save them to text files.
"""

import os
import json
import argparse
from typing import List, Set


def load_jsonl_as_list(path: str) -> List[dict]:
    """Load JSONL file as list of dictionaries."""
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data


def extract_superdomains_and_domains(files: List[str]) -> (Set[str], Set[str]):
    superdomains = set()
    domains = set()

    for path in files:
        print(f"Loading: {path}")
        records = load_jsonl_as_list(path)
        for record in records:
            if "superdomain" in record:
                superdomains.add(record["superdomain"])
            if "domain" in record:
                domains.add(record["domain"])

    return superdomains, domains


def write_list_to_file(path: str, values: Set[str]):
    """Write each value to a new line in a text file."""
    with open(path, "w", encoding="utf-8") as f:
        for v in sorted(values):
            f.write(v + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract superdomains and domains from JSONL files"
    )
    parser.add_argument("jsonl_files", nargs="+", help="One or more JSONL files to process")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    args = parser.parse_args()

    superdomains, domains = extract_superdomains_and_domains(args.jsonl_files)

    os.makedirs(args.output_dir, exist_ok=True)
    superdomains_file = os.path.join(args.output_dir, "superdomains.txt")
    domains_file = os.path.join(args.output_dir, "domains.txt")

    write_list_to_file(superdomains_file, superdomains)
    write_list_to_file(domains_file, domains)

    print(f"Saved {len(superdomains)} unique superdomains to {superdomains_file}")
    print(f"Saved {len(domains)} unique domains to {domains_file}")


if __name__ == "__main__":
    main()
