"""
Optimized Temporal Validation - iterates over internal nodes instead of leaves.
Processes each ancestor's sibling pairs exactly once, not once per leaf.
"""
import json
import sys
from ete3 import Tree


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


def validate_optimized(tree, timestamps, family_labels=None, granularity='year'):
    """Iterate over internal nodes only, process each sibling group once."""
    correct_pairs = 0
    total_comparisons = 0
    intra_correct = 0
    intra_total = 0
    inter_correct = 0
    inter_total = 0
    processed_nodes = 0
    nodes_with_leaves = 0

    for node in tree.traverse():
        if node.is_leaf():
            continue
        processed_nodes += 1

        # Get leaf children of this node
        sibling_leaves = [leaf.name for leaf in node.get_leaves()]

        if len(sibling_leaves) < 2:
            continue

        nodes_with_leaves += 1

        # Skip if this is a deep internal node (not direct parent of leaves)
        # Only compare siblings that share an immediate parent
        # Get direct children that are leaves
        direct_leaves = [c for c in node.children if c.is_leaf()]
        if len(direct_leaves) < 2:
            continue

        # Compare all pairs of direct leaf children
        for i, leaf1 in enumerate(direct_leaves):
            for leaf2 in direct_leaves[i+1:]:
                name1, name2 = leaf1.name, leaf2.name

                if name1 not in timestamps or name2 not in timestamps:
                    continue

                # Distance from parent to each leaf
                d1 = leaf1.dist
                d2 = leaf2.dist

                t1 = parse_timestamp(timestamps[name1])
                t2 = parse_timestamp(timestamps[name2])

                if t1[0] == 0 or t2[0] == 0:
                    continue

                if granularity == 'year':
                    t1_val, t2_val = t1[0], t2[0]
                else:
                    t1_val = t1[0] * 12 + t1[1]
                    t2_val = t2[0] * 12 + t2[1]

                total_comparisons += 1

                if d1 != d2 and t1_val != t2_val:
                    tree_order = d1 < d2
                    time_order = t1_val < t2_val
                    is_correct = tree_order == time_order
                else:
                    is_correct = True

                if is_correct:
                    correct_pairs += 1

                if family_labels:
                    same_family = family_labels.get(name1) == family_labels.get(name2)
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
        'processed_nodes': processed_nodes,
        'nodes_with_direct_leaves': nodes_with_leaves,
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
    results = validate_optimized(tree, timestamps, family_labels, args.granularity)

    print(f"\n=== Temporal Validation Results ({args.granularity}-level) ===")
    if 'intra_family' in results:
        print(f"Intra-family accuracy: {results['intra_family']['accuracy']:.4f} ({results['intra_family']['correct']}/{results['intra_family']['total']})")
        print(f"Inter-family accuracy: {results['inter_family']['accuracy']:.4f} ({results['inter_family']['correct']}/{results['inter_family']['total']})")
    print(f"Overall temporal consistency: {results['accuracy']:.4f} ({results['correct_pairs']}/{results['total_comparisons']})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
