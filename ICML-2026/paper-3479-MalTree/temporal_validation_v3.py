"""
Temporal Validation - Paper Algorithm 3.
Compares leaf sibling pairs sharing a direct parent.
Processes each unique parent node once.
Strips quote characters from leaf names for timestamp lookup.
"""
import json
import os
import sys
from ete3 import Tree


def clean_name(name):
    """Strip surrounding quotes from leaf names."""
    return name.strip("'\"")


def load_tree(tree_path):
    return Tree(tree_path)


def load_timestamps(ts_file):
    with open(ts_file, 'r') as f:
        data = json.load(f)
    timestamps = {}
    for sha, value in data.items():
        if isinstance(value, dict):
            timestamps[sha] = value.get('first_submission', value.get('timestamp', ''))
        else:
            timestamps[sha] = str(value)
    return timestamps


def parse_timestamp(timestamp):
    if not timestamp:
        return (0, 0)
    timestamp = timestamp.strip().split()[0]
    try:
        parts = timestamp.replace('/', '-').split('-')
        year = int(parts[0]) if len(parts) > 0 else 0
        month = int(parts[1]) if len(parts) > 1 else 1
        return (year, month)
    except (ValueError, IndexError):
        return (0, 0)


def validate_siblings(tree, timestamps, family_labels=None, granularity='year',
                       max_descendants=0):
    """
    Compare all leaf pairs sharing an immediate parent.
    Each unique parent processed once.
    """
    correct_pairs = 0
    total_comparisons = 0
    intra_correct = 0
    intra_total = 0
    inter_correct = 0
    inter_total = 0
    skipped_no_ts = 0
    skipped_invalid_ts = 0
    skipped_ties = 0
    skipped_large = 0
    processed_parents = 0

    # Find unique leaf-parent nodes
    parent_nodes = set()
    for leaf in tree.get_leaves():
        if leaf.up is not None:
            parent_nodes.add(leaf.up)

    print(f"Unique leaf-parent nodes: {len(parent_nodes)}")

    for parent in parent_nodes:
        all_leaves = parent.get_leaves()
        n = len(all_leaves)

        if n < 2:
            continue

        if max_descendants > 0 and n > max_descendants:
            skipped_large += 1
            continue

        processed_parents += 1

        if processed_parents % 5000 == 0:
            acc = correct_pairs / total_comparisons if total_comparisons > 0 else 0
            print(f"  [{processed_parents} parents] {total_comparisons} pairs, accuracy={acc:.4f}")

        # Build leaf data (name, distance, timestamp) for valid leaves
        leaf_data = []
        for leaf_obj in all_leaves:
            name = clean_name(leaf_obj.name)
            if name not in timestamps:
                continue

            t = parse_timestamp(timestamps[name])
            if t[0] == 0:
                continue

            d = parent.get_distance(leaf_obj)
            if granularity == 'year':
                t_val = t[0]
            else:
                t_val = t[0] * 12 + t[1]

            leaf_data.append((name, d, t_val, leaf_obj))

        m = len(leaf_data)
        if m < 2:
            continue

        # Compare all pairs
        for i in range(m):
            name_i, d_i, t_i, obj_i = leaf_data[i]
            for j in range(i + 1, m):
                name_j, d_j, t_j, obj_j = leaf_data[j]

                total_comparisons += 1

                if t_i == t_j:
                    correct_pairs += 1
                    skipped_ties += 1
                    continue

                if abs(d_i - d_j) < 1e-10:
                    correct_pairs += 1
                    skipped_ties += 1
                    continue

                tree_order = d_i < d_j
                time_order = t_i < t_j
                is_correct = (tree_order == time_order)

                if is_correct:
                    correct_pairs += 1

                if family_labels:
                    fam_i = family_labels.get(name_i, '')
                    fam_j = family_labels.get(name_j, '')
                    same_family = (fam_i == fam_j)
                    if same_family:
                        intra_total += 1
                        if is_correct:
                            intra_correct += 1
                    else:
                        inter_total += 1
                        if is_correct:
                            inter_correct += 1

    accuracy = correct_pairs / total_comparisons if total_comparisons > 0 else 0.0

    result = {
        'accuracy': accuracy,
        'correct_pairs': correct_pairs,
        'total_comparisons': total_comparisons,
        'processed_parents': processed_parents,
        'skipped_large': skipped_large,
        'skipped_ties': skipped_ties,
    }

    if family_labels:
        result['intra_family'] = {
            'accuracy': intra_correct / intra_total if intra_total > 0 else 0,
            'correct': intra_correct,
            'total': intra_total
        }
        result['inter_family'] = {
            'accuracy': inter_correct / inter_total if inter_total > 0 else 0,
            'correct': inter_correct,
            'total': inter_total
        }

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', required=True)
    parser.add_argument('--timestamps', required=True)
    parser.add_argument('--families', default=None)
    parser.add_argument('--granularity', choices=['year', 'month'], default='year')
    parser.add_argument('--output', default=None)
    parser.add_argument('--max-descendants', type=int, default=0,
                       help='Skip parents with > N leaf descendants (0=no limit)')
    args = parser.parse_args()

    print("Loading tree...")
    tree = load_tree(args.tree)
    print(f"Loaded tree with {len(tree.get_leaves())} leaves")

    print("Loading timestamps...")
    timestamps = load_timestamps(args.timestamps)
    print(f"Loaded {len(timestamps)} timestamps")

    family_labels = None
    if args.families:
        print("Loading family labels...")
        with open(args.families, 'r') as f:
            family_labels = json.load(f)
        print(f"Loaded {len(family_labels)} family labels")

    print(f"Running validation (granularity={args.granularity})...")
    if args.max_descendants > 0:
        print(f"  Skipping parents with > {args.max_descendants} leaf descendants")

    results = validate_siblings(tree, timestamps, family_labels, args.granularity,
                                 args.max_descendants)

    print(f"\n=== Temporal Validation Results ({args.granularity}-level) ===")
    if 'intra_family' in results:
        print(f"Intra-family: {results['intra_family']['accuracy']:.4f} "
              f"({results['intra_family']['correct']}/{results['intra_family']['total']})")
        print(f"Inter-family: {results['inter_family']['accuracy']:.4f} "
              f"({results['inter_family']['correct']}/{results['inter_family']['total']})")
    print(f"Overall temporal consistency: {results['accuracy']:.4f} "
          f"({results['correct_pairs']}/{results['total_comparisons']})")
    print(f"Parents processed: {results['processed_parents']}, skipped (too large): {results.get('skipped_large', 0)}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
