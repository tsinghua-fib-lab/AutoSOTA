#!/usr/bin/env python3
"""
Pre-build the embedding index for topic retrieval.

Run this BEFORE starting the vLLM server to cache the embeddings on GPU.
Then the actual taboo eval can load from cache without needing GPU memory.

Usage:
    python build_embedding_index.py
    python build_embedding_index.py --index-strategy title_plus_all_labels
    python build_embedding_index.py --index-strategy title_only --device cuda:0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.embedding_retrieval.topic_retrieval_eval import (
    TopicRetrievalConfig,
    TopicRetrievalIndex,
    TopicDataset,
    IndexStrategy,
)


def build_index(
    index_strategy: str = "title_only",
    device: str = "cuda",
    force_rebuild: bool = False,
):
    """
    Build and cache the embedding index.
    
    Args:
        index_strategy: Which indexing strategy to use
        device: Device for embedding computation
        force_rebuild: If True, rebuild even if cache exists
    """
    print("=" * 80)
    print("Building Embedding Index")
    print("=" * 80)
    print(f"Strategy: {index_strategy}")
    print(f"Device: {device}")
    print(f"Force rebuild: {force_rebuild}")
    print("=" * 80 + "\n")
    
    config = TopicRetrievalConfig(
        dataset=TopicDataset.FIFTY_THOUSAND,
        device=device,
        index_strategy=IndexStrategy(index_strategy),
    )
    
    index = TopicRetrievalIndex(config)
    
    cache_path = index.get_default_cache_path()
    print(f"Cache path: {cache_path}\n")
    
    if not force_rebuild and (cache_path / "embeddings.pt").exists():
        print(f"Cache already exists at {cache_path}")
        print("Use --force-rebuild to rebuild anyway.")
        return
    
    # Build and save
    index.build_or_load_index(force_rebuild=force_rebuild)
    
    print("\n" + "=" * 80)
    print("Index built and cached successfully!")
    print(f"Cache location: {cache_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build embedding index for topic retrieval"
    )
    parser.add_argument(
        "--index-strategy", type=str, default="title_only",
        choices=["title_only", "title_plus_first_label", "title_plus_all_labels", "mean_of_all"],
        help="Indexing strategy (default: title_only)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for embedding computation (default: cuda)"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild even if cache exists"
    )
    parser.add_argument(
        "--all-strategies", action="store_true",
        help="Build all indexing strategies"
    )
    
    args = parser.parse_args()
    
    if args.all_strategies:
        strategies = ["title_only", "title_plus_first_label", "title_plus_all_labels", "mean_of_all"]
        for strategy in strategies:
            print(f"\n{'#' * 80}")
            print(f"# Building: {strategy}")
            print(f"{'#' * 80}\n")
            build_index(
                index_strategy=strategy,
                device=args.device,
                force_rebuild=args.force_rebuild,
            )
    else:
        build_index(
            index_strategy=args.index_strategy,
            device=args.device,
            force_rebuild=args.force_rebuild,
        )


if __name__ == "__main__":
    main()
