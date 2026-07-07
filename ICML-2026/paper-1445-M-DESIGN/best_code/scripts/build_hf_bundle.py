"""Create Hugging Face metadata for the released M-DESIGN knowledge base."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

RELEASE_PT_FILES = {"ecc_predictor.pt", "model_graph.pt"}


def is_release_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() == ".db":
        return True
    return path.suffix.lower() == ".pt" and path.name in RELEASE_PT_FILES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(file for file in root.rglob("*") if is_release_artifact(file)):
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        rows.append(
            {
                "path": relative,
                "task": parts[0] if len(parts) >= 3 else "unknown",
                "dataset": parts[1] if len(parts) >= 3 else path.stem,
                "artifact_type": "sqlite_database" if path.suffix == ".db" else "torch_artifact",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_outputs(root: Path, manifest: list[dict[str, object]]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "task", "dataset", "artifact_type", "bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="knowledge_retrieval/knowledge_base")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = build_manifest(root)
    if not manifest:
        raise SystemExit(f"No release artifacts found under {root}")
    write_outputs(root, manifest)
    print(f"Wrote manifest for {len(manifest)} artifacts under {root}")


if __name__ == "__main__":
    main()
