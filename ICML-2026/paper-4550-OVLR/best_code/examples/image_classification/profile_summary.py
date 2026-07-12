"""
OVLR Example: BP vs OVLR Profile Summary

Generates comparative summary of BP vs OVLR profiling results.
"""

import json
import os
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate BP vs OVLR profile summary')
    parser.add_argument('--bp-dir', type=str, default='./results_bp_profile',
                        help='BP results directory')
    parser.add_argument('--ovlr-dir', type=str, default='./results_ovlr_profile',
                        help='OVLR results directory')
    return parser.parse_args()


def load_results(directory, prefix):
    results = {}
    pattern = os.path.join(directory, f'*_{prefix}.json')
    for path in glob.glob(pattern):
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        results[model] = data
    return results


def main():
    args = parse_args()

    bp_results = load_results(args.bp_dir, 'bp')
    ovlr_results = load_results(args.ovlr_dir, 'ovlr')

    if not bp_results or not ovlr_results:
        print("No results found!")
        return

    dataset = list(bp_results.values())[0]['dataset']
    all_models = set(bp_results.keys()) | set(ovlr_results.keys())

    print("=" * 80)
    print(f"BP vs OVLR Performance Comparison: {dataset}")
    print("=" * 80)
    print(f"{'Model':<20} {'Method':<8} {'Time(s)':>10} {'Mem(MB)':>10} {'Acc(%)':>10}")
    print("-" * 80)

    for model in sorted(all_models):
        if model in bp_results:
            r = bp_results[model]
            print(f"{model:<20} {'BP':<8} {r['train_time_seconds_mean']:>10.2f} "
                  f"{r['max_memory_allocated_MB']:>10.1f} {r['final_accuracy']:>10.2f}")
        if model in ovlr_results:
            r = ovlr_results[model]
            print(f"{model:<20} {'OVLR':<8} {r['train_time_seconds_mean']:>10.2f} "
                  f"{r['max_memory_allocated_MB']:>10.1f} {r['final_accuracy']:>10.2f}")

        if model in bp_results and model in ovlr_results:
            time_ratio = ovlr_results[model]['train_time_seconds_mean'] / \
                        bp_results[model]['train_time_seconds_mean']
            mem_ratio = ovlr_results[model]['max_memory_allocated_MB'] / \
                       bp_results[model]['max_memory_allocated_MB']
            print(f"{' ' * 28} slowdown: {time_ratio:.2f}x  overhead: {mem_ratio:.2f}x")

    print("-" * 80)


if __name__ == '__main__':
    main()
