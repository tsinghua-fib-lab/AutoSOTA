#!/usr/bin/env bash
#
# Reproduce the MalTree pipeline from released embeddings.
#
#   bash reproduce.sh --example     # run on the small bundled example subset
#   bash reproduce.sh               # run on the full released data in data/
#
# Stages: L2-normalize -> distance matrix -> PHYLIP -> phylogenetic tree
#         -> inter-family graph (GraphML) -> embedding-drift figure.
#
# Prerequisites: a Python environment with the packages in requirements.txt
# (or environment.yml). Override the interpreter with the PYTHON env var, e.g.
#   PYTHON=python3.10 bash reproduce.sh --example
#
# The full released data is fetched with: python scripts/download_data.py
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PT="$ROOT/phylogenetic_tree"

if [[ "${1:-}" == "--example" ]]; then
    DATA="$ROOT/data/example"
    OUT="$ROOT/results/example"
else
    DATA="$ROOT/data"
    OUT="$ROOT/results/full"
fi

FUSED="$DATA/embeddings_fused.json"
FAMILIES="$DATA/family_labels.json"
TIMESTAMPS="$DATA/timestamps.json"

if [[ ! -f "$FUSED" ]]; then
    echo "error: $FUSED not found." >&2
    echo "  run: python scripts/download_data.py            (full data)" >&2
    echo "  or:  python scripts/download_data.py --example  (example subset)" >&2
    exit 1
fi
mkdir -p "$OUT"

echo "=== MalTree reproduction ==="
echo "fused embeddings : $FUSED"
echo "outputs          : $OUT"
echo

echo "[1/6] L2-normalize embeddings"
"$PYTHON" "$PT/normalize_embeddings.py" --input "$FUSED" --output "$OUT/normalized.json"

echo "[2/6] pairwise Euclidean distance matrix"
"$PYTHON" "$PT/generate_distance_matrix.py" --input "$OUT/normalized.json" \
    --output "$OUT/distance_matrix.npy" --output-labels "$OUT/sample_labels.json"

echo "[3/6] PHYLIP conversion"
"$PYTHON" "$PT/convert_to_pyphil_format.py" --matrix "$OUT/distance_matrix.npy" \
    --embeddings "$FUSED" --output "$OUT/distances.phy" \
    --label-mapping "$OUT/label_mapping.txt"

echo "[4/6] build phylogenetic tree (UPGMA)"
"$PYTHON" "$PT/tree_io.py" build "$OUT/distances.phy" --method upgma \
    --output "$OUT/tree.nwk"

echo "[5/6] inter-family evolutionary graph"
"$PYTHON" "$PT/inter_family_analysis.py" --tree "$OUT/tree.nwk" \
    --output "$OUT/inter_family.json" --graphml-prefix "$OUT/graph"

echo "[6/6] embedding-drift analysis (paper Figure 5)"
"$PYTHON" "$PT/embedding_drift.py" --embeddings "$FUSED" \
    --timestamps "$TIMESTAMPS" --families "$FAMILIES" \
    --output "$OUT/drift.json" --figure "$OUT/figure5.png"

echo
echo "=== done ==="
echo "tree   : $OUT/tree.nwk"
echo "graph  : $OUT/graph_simplified.graphml"
echo "figure : $OUT/figure5.png"
echo
echo "For temporal validation (paper Tables 2-3) and outlier detection on the"
echo "full tree, see the commands in README.md / DATA.md."
