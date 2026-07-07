"""Download the released M-DESIGN knowledge base from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jilwang804/M-DESIGN-Knowledge-Base")
    parser.add_argument("--local-dir", default="knowledge_retrieval/knowledge_base")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install Hugging Face support with `pip install -e .[hf]`.") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[
            "*.db",
            "ecc_predictor.pt",
            "model_graph.pt",
            "manifest.json",
            "metadata.csv",
            "README.md",
        ],
    )
    print(f"Downloaded knowledge base to {local_dir}")


if __name__ == "__main__":
    main()
