"""
Fast Temporal Validation - multiple modes.
Mode 1: Direct siblings only (leaves sharing immediate parent)
Mode 2: All leaf descendants under each parent (with max_descendants threshold)
"""
import json
import os
from ete3 import Tree


def clean_name(name):
    return name.strip("'\"")


def load_tree(path):
    return Tree(path)


def load_timestamps(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = v.get('first_submission', v.get('timestamp', ''))
        else:
            out[k] = str(v)
    return out


def parse_ts(ts):
    if not ts:
        return (0, 0)
    ts = ts.strip().split()[0]
    try:
        parts = ts.replace('/', '-').split('-')
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 1)
    except:
        return (0, 0)


def validate_direct_siblings(tree, timestamps, families, granularity='year'):
    """Only compare leaves that are direct children of the same parent (strict siblings)."""
    correct = 0
    total = 0
    intra_c = 0
    intra_t = 0
    inter_c = 0
    inter_t = 0
    ties = 0

    for node in tree.traverse():
        if node.is_leaf():
            continue
        # Get direct leaf children
        leaf_children = [c for c in node.children if c.is_leaf()]
        if len(leaf_children) < 2:
            continue

        # Compare all pairs of direct leaf siblings
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
                if granularity == 'year':
                    tvi, tvj = ti[0], tj[0]
                else:
                    tvi, tvj = ti[0] * 12 + ti[1], tj[0] * 12 + tj[1]

                if tvi == tvj or abs(li.dist - lj.dist) < 1e-10:
                    correct += 1
                    ties += 1
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

    acc = correct / total if total > 0 else 0
    result = {'accuracy': acc, 'correct': correct, 'total': total, 'ties': ties,
              'mode': 'direct_siblings'}
    if families:
        result['intra'] = {'acc': intra_c / intra_t if intra_t > 0 else 0,
                          'correct': intra_c, 'total': intra_t}
        result['inter'] = {'acc': inter_c / inter_t if inter_t > 0 else 0,
                          'correct': inter_c, 'total': inter_t}
    return result


def validate_all_descendants(tree, timestamps, families, granularity='year',
                              max_descendants=100):
    """Compare all leaf pairs under each parent, with threshold for large groups."""
    correct = 0
    total = 0
    intra_c = 0
    intra_t = 0
    inter_c = 0
    inter_t = 0
    ties = 0
    skipped = 0
    processed = 0

    # Collect unique leaf-parent nodes
    parent_set = set()
    for leaf in tree.get_leaves():
        if leaf.up is not None:
            parent_set.add(leaf.up)

    parents = list(parent_set)
    print(f"  Total leaf-parent nodes: {len(parents)}")

    for parent in parents:
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

        # Pre-filter valid leaves
        valid = []
        for leaf_obj in all_leaves:
            name = clean_name(leaf_obj.name)
            if name not in timestamps:
                continue
            t = parse_ts(timestamps[name])
            if t[0] == 0:
                continue
            d = parent.get_distance(leaf_obj)
            tv = t[0] if granularity == 'year' else t[0] * 12 + t[1]
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
                    ties += 1
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

    acc = correct / total if total > 0 else 0
    result = {'accuracy': acc, 'correct': correct, 'total': total, 'ties': ties,
              'mode': 'all_descendants', 'max_descendants': max_descendants,
              'processed_parents': processed, 'skipped_large': skipped}
    if families:
        result['intra'] = {'acc': intra_c / intra_t if intra_t > 0 else 0,
                          'correct': intra_c, 'total': intra_t}
        result['inter'] = {'acc': inter_c / inter_t if inter_t > 0 else 0,
                          'correct': inter_c, 'total': inter_t}
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True)
    p.add_argument('--timestamps', required=True)
    p.add_argument('--families')
    p.add_argument('--granularity', default='year')
    p.add_argument('--output')
    p.add_argument('--mode', choices=['direct', 'all'], default='direct')
    p.add_argument('--max-descendants', type=int, default=100)
    args = p.parse_args()

    print("Loading tree...")
    tree = load_tree(args.tree)
    print(f"  {len(tree.get_leaves())} leaves")

    print("Loading timestamps...")
    ts = load_timestamps(args.timestamps)
    print(f"  {len(ts)} entries")

    fams = None
    if args.families:
        print("Loading families...")
        with open(args.families) as f:
            fams = json.load(f)
        print(f"  {len(fams)} entries")

    if args.mode == 'direct':
        print("Mode: direct siblings only")
        result = validate_direct_siblings(tree, ts, fams, args.granularity)
    else:
        print(f"Mode: all descendants (max {args.max_descendants})")
        result = validate_all_descendants(tree, ts, fams, args.granularity,
                                           args.max_descendants)

    print(f"\n=== Temporal Validation ({args.granularity}, {args.mode}) ===")
    if 'intra' in result:
        print(f"Intra-family: {result['intra']['acc']:.4f} ({result['intra']['correct']}/{result['intra']['total']})")
        print(f"Inter-family: {result['inter']['acc']:.4f} ({result['inter']['correct']}/{result['inter']['total']})")
    print(f"Overall: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
