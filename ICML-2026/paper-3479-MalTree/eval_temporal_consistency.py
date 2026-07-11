#!/usr/bin/env python3
"""
MalTree Temporal Consistency Evaluation
========================================
Reproduces paper Table 2 result: NJ + Outgroup temporal consistency.

Usage:
    python3 eval_temporal_consistency.py [--mode direct|all] [--max-descendants N]

Output: temporal consistency score (0-1), matching paper Table 2.
"""
import json
import os
import sys
import argparse
from ete3 import Tree


def prepare_leaf_data(tree_path, mapping_path, ts_path, fam_path, output_dir):
    """Prepare leaf-level timestamp and family mappings if needed."""
    leaf_ts_file = os.path.join(output_dir, "leaf_timestamps.json")
    leaf_fam_file = os.path.join(output_dir, "leaf_families.json")

    if os.path.exists(leaf_ts_file) and os.path.exists(leaf_fam_file):
        return leaf_ts_file, leaf_fam_file

    print("Preparing leaf-level data mappings...")

    # Load leaf -> SHA mapping
    leaf_to_sha = {}
    with open(mapping_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                leaf_to_sha[parts[0]] = parts[1]

    # Load timestamps (SHA -> timestamp)
    with open(ts_path) as f:
        ts_data = json.load(f)

    # Load families (SHA -> family)
    with open(fam_path) as f:
        fam_data = json.load(f)

    # Create leaf-keyed mappings
    leaf_ts = {}
    leaf_fam = {}
    for leaf_label, sha in leaf_to_sha.items():
        if sha in ts_data:
            entry = ts_data[sha]
            if isinstance(entry, dict):
                leaf_ts[leaf_label] = entry.get("first_submission", "")
            else:
                leaf_ts[leaf_label] = str(entry)
        if sha in fam_data:
            leaf_fam[leaf_label] = fam_data[sha]

    with open(leaf_ts_file, "w") as f:
        json.dump(leaf_ts, f)
    with open(leaf_fam_file, "w") as f:
        json.dump(leaf_fam, f)

    print(f"  Mapped {len(leaf_ts)} timestamps, {len(leaf_fam)} families")
    return leaf_ts_file, leaf_fam_file


def clean_name(name):
    return name.strip("'\"")


def parse_ts(ts):
    if not ts:
        return (0, 0)
    ts = ts.strip().split()[0]
    try:
        parts = ts.replace("/", "-").split("-")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 1)
    except (ValueError, IndexError):
        return (0, 0)


def validate_direct_siblings(tree, timestamps, families, granularity="year"):
    """Compare leaf pairs that are direct children of the same parent."""
    correct = 0
    total = 0
    intra_c, intra_t = 0, 0
    inter_c, inter_t = 0, 0

    for node in tree.traverse():
        if node.is_leaf():
            continue
        leaf_children = [c for c in node.children if c.is_leaf()]
        if len(leaf_children) < 2:
            continue

        for i in range(len(leaf_children)):
            li = leaf_children[i]
            ni = clean_name(li.name)
            if ni not in timestamps:
                continue
            ti = parse_ts(timestamps[ni])
            if ti[0] == 0:
                continue

            for j in range(i + 1, len(leaf_children)):
                lj = leaf_children[j]
                nj = clean_name(lj.name)
                if nj not in timestamps:
                    continue
                tj = parse_ts(timestamps[nj])
                if tj[0] == 0:
                    continue

                total += 1
                if granularity == "year":
                    tvi, tvj = ti[0], tj[0]
                else:
                    tvi, tvj = ti[0] * 12 + ti[1], tj[0] * 12 + tj[1]

                if tvi == tvj or abs(li.dist - lj.dist) < 1e-10:
                    correct += 1
                    continue

                is_correct = (li.dist < lj.dist) == (tvi < tvj)
                if is_correct:
                    correct += 1

                if families:
                    same = families.get(ni) == families.get(nj)
                    if same:
                        intra_t += 1
                        if is_correct:
                            intra_c += 1
                    else:
                        inter_t += 1
                        if is_correct:
                            inter_c += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "intra_accuracy": intra_c / intra_t if intra_t > 0 else 0,
        "intra_correct": intra_c,
        "intra_total": intra_t,
        "inter_accuracy": inter_c / inter_t if inter_t > 0 else 0,
        "inter_correct": inter_c,
        "inter_total": inter_t,
    }


def validate_all_descendants(tree, timestamps, families, granularity="year",
                              max_descendants=100):
    """Compare all leaf pairs under each parent node."""
    correct = 0
    total = 0
    intra_c, intra_t = 0, 0
    inter_c, inter_t = 0, 0
    skipped = 0
    processed = 0

    parent_set = set()
    for leaf in tree.get_leaves():
        if leaf.up is not None:
            parent_set.add(leaf.up)

    for parent in parent_set:
        all_leaves = parent.get_leaves()
        n = len(all_leaves)
        if n < 2:
            continue
        if n > max_descendants:
            skipped += 1
            continue

        processed += 1
        if processed % 10000 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  [{processed}] pairs={total}, acc={acc:.4f}")

        valid = []
        for leaf_obj in all_leaves:
            name = clean_name(leaf_obj.name)
            if name not in timestamps:
                continue
            t = parse_ts(timestamps[name])
            if t[0] == 0:
                continue
            d = parent.get_distance(leaf_obj)
            tv = t[0] if granularity == "year" else t[0] * 12 + t[1]
            valid.append((name, d, tv))

        m = len(valid)
        if m < 2:
            continue

        for i in range(m):
            ni, di, tvi = valid[i]
            for j in range(i + 1, m):
                nj, dj, tvj = valid[j]
                total += 1

                if tvi == tvj or abs(di - dj) < 1e-10:
                    correct += 1
                    continue

                is_correct = (di < dj) == (tvi < tvj)
                if is_correct:
                    correct += 1

                if families:
                    same = families.get(ni) == families.get(nj)
                    if same:
                        intra_t += 1
                        if is_correct:
                            intra_c += 1
                    else:
                        inter_t += 1
                        if is_correct:
                            inter_c += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "intra_accuracy": intra_c / intra_t if intra_t > 0 else 0,
        "intra_correct": intra_c,
        "intra_total": intra_t,
        "inter_accuracy": inter_c / inter_t if inter_t > 0 else 0,
        "inter_correct": inter_c,
        "inter_total": inter_t,
        "skipped_large_parents": skipped,
        "processed_parents": processed,
        "max_descendants": max_descendants,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MalTree Temporal Consistency Evaluation"
    )
    parser.add_argument("--tree", default="/repo/trees/tree_nj.nwk",
                       help="Path to Newick tree file")
    parser.add_argument("--leaf-mapping", default="/repo/trees/leaf_mapping.tsv",
                       help="Path to leaf mapping TSV")
    parser.add_argument("--timestamps", default="/repo/data/timestamps.json",
                       help="Path to timestamps JSON (SHA-keyed)")
    parser.add_argument("--families", default="/repo/data/family_labels.json",
                       help="Path to family labels JSON (SHA-keyed)")
    parser.add_argument("--granularity", choices=["year", "month"], default="year")
    parser.add_argument("--mode", choices=["direct", "all"], default="direct")
    parser.add_argument("--max-descendants", type=int, default=100,
                       help="Max leaf descendants per parent (all mode only)")
    parser.add_argument("--output", default=None,
                       help="Output JSON path for results")
    parser.add_argument("--data-dir", default="/repo/results",
                       help="Directory for intermediate data files")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Prepare leaf-level mappings
    leaf_ts_file, leaf_fam_file = prepare_leaf_data(
        args.tree, args.leaf_mapping, args.timestamps, args.families, args.data_dir
    )

    # Load data
    print(f"Loading tree: {args.tree}")
    tree = Tree(args.tree, format=1)
    print(f"  {len(tree.get_leaves())} leaves")

    with open(leaf_ts_file) as f:
        timestamps = json.load(f)
    print(f"  {len(timestamps)} timestamps")

    families = None
    if os.path.exists(leaf_fam_file):
        with open(leaf_fam_file) as f:
            families = json.load(f)
        print(f"  {len(families)} families")

    # Run validation
    if args.mode == "direct":
        print(f"Mode: direct siblings ({args.granularity})")
        result = validate_direct_siblings(tree, timestamps, families, args.granularity)
    else:
        print(f"Mode: all descendants, max {args.max_descendants} ({args.granularity})")
        result = validate_all_descendants(tree, timestamps, families,
                                           args.granularity, args.max_descendants)

    # Display results
    print(f"\n{'='*60}")
    print(f"Temporal Consistency: {result['accuracy']:.4f}")
    print(f"  Correct pairs: {result['correct']}")
    print(f"  Total pairs:   {result['total']}")
    if "intra_accuracy" in result:
        print(f"  Intra-family:  {result['intra_accuracy']:.4f} "
              f"({result['intra_correct']}/{result['intra_total']})")
        print(f"  Inter-family:  {result['inter_accuracy']:.4f} "
              f"({result['inter_correct']}/{result['inter_total']})")
    print(f"{'='*60}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return result


if __name__ == "__main__":
    main()
