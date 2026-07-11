"""
Temporal Validation Framework for Phylogenetic Trees

Implements Algorithm 3 from the paper: Validate Chronological Order of Phylogenetic Tree Leaves.
Compares tree-inferred divergence order against VirusTotal timestamps.
"""
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from ete3 import Tree


def load_tree(tree_path: str) -> Tree:
    """Load a phylogenetic tree from Newick format file."""
    return Tree(tree_path)


def load_timestamps(timestamp_file: str) -> Dict[str, str]:
    """
    Load VirusTotal timestamps from JSON file.

    Expected format: {"sha": {"first_submission": "YYYY-MM-DD"}, ...}
    or: {"sha": "YYYY-MM-DD", ...}
    """
    with open(timestamp_file, 'r') as f:
        data = json.load(f)

    timestamps = {}
    for sha, value in data.items():
        if isinstance(value, dict):
            timestamps[sha] = value.get('first_submission', value.get('timestamp', ''))
        else:
            timestamps[sha] = str(value)

    return timestamps


def get_direct_ancestor(tree: Tree, leaf: str) -> Tree:
    """Get the direct ancestor (parent node) of a leaf."""
    leaf_node = tree.search_nodes(name=leaf)
    if not leaf_node:
        return None
    return leaf_node[0].up


def get_leaves_from_ancestor(ancestor: Tree) -> List[str]:
    """Get all leaf names descending from an ancestor node."""
    if ancestor is None:
        return []
    return [leaf.name for leaf in ancestor.get_leaves()]


def get_path_distance(tree: Tree, node1_name: str, node2_name: str) -> float:
    """Calculate the path distance between two nodes in the tree."""
    try:
        node1 = tree.search_nodes(name=node1_name)
        node2 = tree.search_nodes(name=node2_name)

        if not node1 or not node2:
            return float('inf')

        return node1[0].get_distance(node2[0])
    except Exception:
        return float('inf')


def parse_timestamp(timestamp: str) -> Tuple[int, int]:
    """
    Parse timestamp to (year, month) tuple for comparison.
    Handles formats: YYYY-MM-DD, YYYY/MM/DD, YYYY-MM, YYYY
    """
    if not timestamp:
        return (0, 0)

    # Clean the timestamp
    timestamp = timestamp.strip().split()[0]  # Remove time if present

    try:
        # Try different formats
        parts = timestamp.replace('/', '-').split('-')
        year = int(parts[0]) if len(parts) > 0 else 0
        month = int(parts[1]) if len(parts) > 1 else 1
        return (year, month)
    except (ValueError, IndexError):
        return (0, 0)


def validate_chronological_order(tree: Tree,
                                  timestamps: Dict[str, str],
                                  granularity: str = 'year') -> Dict:
    """
    Validate chronological order of phylogenetic tree leaves.

    Implements Algorithm 3 from the paper.

    Args:
        tree: Phylogenetic tree (ete3 Tree object)
        timestamps: Dictionary mapping leaf names to timestamps
        granularity: 'year' or 'month' for comparison granularity

    Returns:
        Dictionary with validation results including accuracy score
    """
    correct_pairs = 0
    total_comparisons = 0
    errors = []

    # Get all leaves
    leaves = [leaf.name for leaf in tree.get_leaves()]

    for leaf_name in leaves:
        # Get direct ancestor
        ancestor = get_direct_ancestor(tree, leaf_name)
        if ancestor is None:
            continue

        # Get all sibling leaves (leaves sharing the same parent)
        sibling_leaves = get_leaves_from_ancestor(ancestor)

        # Compare all pairs of siblings
        for i, leaf1 in enumerate(sibling_leaves):
            for leaf2 in sibling_leaves[i+1:]:
                # Skip if timestamps not available
                if leaf1 not in timestamps or leaf2 not in timestamps:
                    continue

                # Calculate path distances from ancestor
                d1 = get_path_distance(tree, ancestor.name if ancestor.name else "root", leaf1)
                d2 = get_path_distance(tree, ancestor.name if ancestor.name else "root", leaf2)

                # Get timestamps
                t1 = parse_timestamp(timestamps[leaf1])
                t2 = parse_timestamp(timestamps[leaf2])

                # Skip invalid timestamps
                if t1[0] == 0 or t2[0] == 0:
                    continue

                # Compare based on granularity
                if granularity == 'year':
                    t1_val, t2_val = t1[0], t2[0]
                else:  # month
                    t1_val = t1[0] * 12 + t1[1]
                    t2_val = t2[0] * 12 + t2[1]

                # Check if tree order matches timestamp order
                total_comparisons += 1

                # If distances are different, check consistency
                if d1 != d2 and t1_val != t2_val:
                    tree_order = d1 < d2  # True if leaf1 is closer (emerged earlier)
                    time_order = t1_val < t2_val  # True if leaf1 is older

                    if tree_order == time_order:
                        correct_pairs += 1
                    else:
                        errors.append({
                            'leaf1': leaf1,
                            'leaf2': leaf2,
                            'd1': d1,
                            'd2': d2,
                            't1': timestamps[leaf1],
                            't2': timestamps[leaf2]
                        })
                elif d1 == d2 or t1_val == t2_val:
                    # Ties count as correct (can't distinguish)
                    correct_pairs += 1

    accuracy = correct_pairs / total_comparisons if total_comparisons > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct_pairs': correct_pairs,
        'total_comparisons': total_comparisons,
        'num_errors': len(errors),
        'errors': errors[:100]  # Limit error list size
    }


def validate_with_families(tree: Tree,
                           timestamps: Dict[str, str],
                           family_labels: Dict[str, str],
                           granularity: str = 'year') -> Dict:
    """
    Validate with family-aware analysis.

    Separates intra-family and inter-family validation as described in Section 4.2.

    Args:
        tree: Phylogenetic tree
        timestamps: Leaf name to timestamp mapping
        family_labels: Leaf name to family name mapping
        granularity: 'year' or 'month'

    Returns:
        Validation results for both intra-family and inter-family comparisons
    """
    intra_family_correct = 0
    intra_family_total = 0
    inter_family_correct = 0
    inter_family_total = 0

    leaves = [leaf.name for leaf in tree.get_leaves()]

    for leaf_name in leaves:
        ancestor = get_direct_ancestor(tree, leaf_name)
        if ancestor is None:
            continue

        sibling_leaves = get_leaves_from_ancestor(ancestor)

        for i, leaf1 in enumerate(sibling_leaves):
            for leaf2 in sibling_leaves[i+1:]:
                if leaf1 not in timestamps or leaf2 not in timestamps:
                    continue
                if leaf1 not in family_labels or leaf2 not in family_labels:
                    continue

                d1 = get_path_distance(tree, ancestor.name or "root", leaf1)
                d2 = get_path_distance(tree, ancestor.name or "root", leaf2)

                t1 = parse_timestamp(timestamps[leaf1])
                t2 = parse_timestamp(timestamps[leaf2])

                if t1[0] == 0 or t2[0] == 0:
                    continue

                if granularity == 'year':
                    t1_val, t2_val = t1[0], t2[0]
                else:
                    t1_val = t1[0] * 12 + t1[1]
                    t2_val = t2[0] * 12 + t2[1]

                # Determine if intra or inter family
                same_family = family_labels[leaf1] == family_labels[leaf2]

                if d1 != d2 and t1_val != t2_val:
                    tree_order = d1 < d2
                    time_order = t1_val < t2_val
                    is_correct = tree_order == time_order
                else:
                    is_correct = True

                if same_family:
                    intra_family_total += 1
                    if is_correct:
                        intra_family_correct += 1
                else:
                    inter_family_total += 1
                    if is_correct:
                        inter_family_correct += 1

    return {
        'intra_family': {
            'accuracy': intra_family_correct / intra_family_total if intra_family_total > 0 else 0,
            'correct': intra_family_correct,
            'total': intra_family_total
        },
        'inter_family': {
            'accuracy': inter_family_correct / inter_family_total if inter_family_total > 0 else 0,
            'correct': inter_family_correct,
            'total': inter_family_total
        },
        'overall': {
            'accuracy': (intra_family_correct + inter_family_correct) /
                       (intra_family_total + inter_family_total)
                       if (intra_family_total + inter_family_total) > 0 else 0,
            'correct': intra_family_correct + inter_family_correct,
            'total': intra_family_total + inter_family_total
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate phylogenetic tree temporal consistency')
    parser.add_argument('--tree', required=True, help='Path to tree file (Newick format)')
    parser.add_argument('--timestamps', required=True, help='Path to timestamps JSON file')
    parser.add_argument('--families', help='Path to family labels JSON file (optional)')
    parser.add_argument('--granularity', choices=['year', 'month'], default='year',
                       help='Temporal granularity for comparison')
    parser.add_argument('--output', help='Path to save results JSON')

    args = parser.parse_args()

    # Load data
    tree = load_tree(args.tree)
    timestamps = load_timestamps(args.timestamps)

    # Run validation
    if args.families:
        with open(args.families, 'r') as f:
            family_labels = json.load(f)
        results = validate_with_families(tree, timestamps, family_labels, args.granularity)
    else:
        results = validate_chronological_order(tree, timestamps, args.granularity)

    # Print results
    print(f"\n=== Temporal Validation Results ({args.granularity}-level) ===")
    if 'overall' in results:
        print(f"Intra-family accuracy: {results['intra_family']['accuracy']:.3f}")
        print(f"Inter-family accuracy: {results['inter_family']['accuracy']:.3f}")
        print(f"Overall accuracy: {results['overall']['accuracy']:.3f}")
    else:
        print(f"Temporal consistency: {results['accuracy']:.3f}")
        print(f"Correct pairs: {results['correct_pairs']}/{results['total_comparisons']}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
