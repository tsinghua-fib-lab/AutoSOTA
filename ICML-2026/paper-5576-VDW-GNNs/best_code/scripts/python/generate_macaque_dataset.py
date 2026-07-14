#!/usr/bin/env python3
"""
Generate macaque dataset artifacts:
 - Split + PCA + filtering + diff/velocity vectors creation (per day)
 - Save per-day path graphs (without Q) as .pt
 - Build/load per-day spatial graphs with O_i frames as .pt
 - Compute Q per trajectory and save to HDF5 nested by day/trial
Downstream, in prep_dataset.py, the Q sparse tensors saved in the HDF5 file are opened and assigned to the correct trajectory's path graph.

CLI lets you control: data_root, workers, data leakage mode (inductive/transductive), days to process (by index), spatial (CkNN) graph k and delta, PCA components, and GO_CUE.

MARBLE settings:
- mode transductive  # they fit PCA and construct CkNN graph on all data from a given day, but mask out test/valid/test nodes while fitting the model (using unsupervised / contrastive loss)
- k_neighbors 30
- delta 1.5
- go_cue 25
- n_components 5
- include_lever_velocity True

Example usage:
python3 scripts/generate_macaque_dataset.py \
--data_root /Users/davidjohnson/Downloads/macaque_reaching \
--mode inductive \
--days 0 \
--n_workers 4 \
--k_neighbors 30 \
--delta 1.5 \
--go_cue 25 \
--n_components 5 \
--include_lever_velocity

Notes:
For k=30, delta=1.5 and 'inductive' mode, we get (for Day 0):
    Neighbor count stats per day (spatial graphs):
    - Day 0: min=21.0, q1=53.0, median=57.0, q3=62.0, max=82.0
"""

import os
import sys
import argparse
from typing import List, Optional
import multiprocessing as mp


def _ensure_project_root_on_path() -> None:
    """
    Ensure project root is on sys.path so intra-repo imports work when script is run directly.
    """
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    pr_str = str(project_root)
    if pr_str not in sys.path:
        sys.path.insert(0, pr_str)

_ensure_project_root_on_path()
from data_processing import macaque_reaching as mr  # noqa: E402


def parse_days(days_arg: Optional[str]) -> Optional[List[int]]:
    if not days_arg:
        return None
    parts = [
        p.strip() for p in days_arg.split(',') \
        if p.strip() != ''
    ]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate macaque dataset artifacts"
    )
    parser.add_argument(
        '--data_root', type=str, required=True, 
        help='Root directory with macaque pickles'
    )
    parser.add_argument(
        '--mode', type=str, 
        choices=['transductive', 'inductive'], default='transductive', 
        help='Experimental design'
    )
    parser.add_argument(
        '--days', type=str, default=None, 
        help='Comma-separated indices of days to process (e.g., 0,2,5). If omitted, process all days.'
    )
    parser.add_argument(
        '--parallel_backend', type=str, default='threads',
        choices=['processes', 'threads'],
        help='Parallel backend for day-level tasks'
    )
    parser.add_argument(
        '--n_workers', type=int, default=max(1, mp.cpu_count() - 1), 
        help='Parallel workers for day-level tasks'
    )
    parser.add_argument(
        '--transform_workers', type=int, default=None, 
        help='Thread workers for per-trial transforms (defaults to n_workers)'
    )
    parser.add_argument(
        '--q_workers', type=int, default=None, 
        help='Thread workers for per-path Q building (defaults to n_workers)'
    )
    parser.add_argument(
        '--k_neighbors', type=int, default=8, 
        help='k for spatial CkNN graphs'
    )
    parser.add_argument(
        '--delta', type=float, default=1.0, 
        help='delta parameter for CkNN'
    )
    parser.add_argument(
        '--n_components', type=int, default=5, 
        help='PCA components (neural velocity dimension)'
    )
    parser.add_argument(
        '--pca_variance', type=float, default=0.95,
        help='If in (0,1), retain this fraction of variance; set to 1.0 to disable variance-based selection'
    )
    parser.add_argument(
        '--no_pca', action='store_true', default=False,
        help='Disable global PCA entirely'
    )
    parser.add_argument('--go_cue', type=int, default=25, 
        help='GO_CUE index (slice from here to end)'
    )
    parser.add_argument(
        '--include_lever_velocity', 
        action='store_true', default=True, 
        help='Include lever velocity in merged dicts'
    )
    parser.add_argument(
        '--no_include_lever_velocity', action='store_false', dest='include_lever_velocity'
    )
    parser.add_argument(
        '--path_graphs_dir', type=str, default=None, 
        help='Directory to save per-day path graphs (default: <data_root>/path_graphs)'
    )
    parser.add_argument(
        '--spatial_graph_dir', type=str, default=None, 
        help='Directory to save per-day spatial graphs (default: <data_root>/spatial_graphs)'
    )
    parser.add_argument(
        '--h5_path', type=str, default=None, 
        help='Output HDF5 path for Q tensors (default: <data_root>/Qs.h5)'
    )
    parser.add_argument(
        '--force_recompute_spatial', action='store_true', default=False, 
        help='Ignore saved spatial graphs and recompute'
    )
    args = parser.parse_args()

    # Resolve defaults dependent on data_root
    data_root = args.data_root
    path_graphs_dir = args.path_graphs_dir \
        or os.path.join(data_root, 'path_graphs')
    spatial_graph_dir = args.spatial_graph_dir \
        or os.path.join(data_root, 'spatial_graphs')
    h5_path = args.h5_path or os.path.join(data_root, 'Qs.h5')
    days_included_idx = parse_days(args.days)

    # Workers
    transform_workers = args.transform_workers \
        if (args.transform_workers is not None) \
        else args.n_workers
    q_workers = args.q_workers \
        if (args.q_workers is not None) \
        else args.n_workers

    # Override module-level constants when requested
    mr.GO_CUE = int(args.go_cue)
    mr.GLOBAL_PCA_REDUCED_DIM = None if args.no_pca else int(args.n_components)
    mr.GLOBAL_PCA_VARIANCE_RETAINED = float(args.pca_variance)

    print(f"Loading macaque data from {data_root}...")
    kinematics, trial_ids, spike_data = mr.load_data_dicts(data_root)

    print("Merging dictionaries (per-trial)...")
    merged = mr.merge_data_dicts(
        kinematics,
        trial_ids,
        spike_data,
        include_lever_velocity=args.include_lever_velocity,
        num_workers=None,
    )

    print(f"Processing by day with splits (mode={args.mode})...")
    outputs_by_day = mr.process_by_day_with_splits(
        merged,
        n_components=args.n_components,
        fit_pca_on_filtered=False,
        apply_filter_before_transform=True,
        mode=args.mode,
        transform_num_workers=transform_workers,
    )

    print(f"Saving daily collections of trajectory-level path graphs (without Q) to {path_graphs_dir}...")
    mr.save_path_graphs_by_day(
        outputs_by_day,
        save_dir=path_graphs_dir,
        days_included_idx=days_included_idx,
        num_workers=args.n_workers,
        backend=args.parallel_backend,
    )

    print(f"Building/loading per-day spatial graphs with O_i (mode={args.mode}) to {spatial_graph_dir}...")
    graphs_by_day = mr.macaque_build_spatial_graphs_with_O(
        outputs_by_day,
        days_included_idx=days_included_idx,
        mode=args.mode,
        spatial_graph_dir=spatial_graph_dir,
        force_recompute=args.force_recompute_spatial,
        n_neighbors=args.k_neighbors,
        cknn_delta=args.delta,
        metric='euclidean',
        include_self=False,
        num_workers=args.n_workers,
        backend=args.parallel_backend,
        reweight_with_median_kernel=True,
    )

    # Print neighbor count stats per day
    print("Neighbor count stats per day (spatial graphs):")
    stats = mr.get_graph_nbr_ct_stats(graphs_by_day)
    for day, s in stats.items():
        print(
            f"- Day {day}: "
            f"min={s['min']:.1f}, q1={s['q1']:.1f}, median={s['median']:.1f}, "
            f"q3={s['q3']:.1f}, max={s['max']:.1f}"
        )

    print(f"Computing Qs and saving to HDF5: {h5_path}...")
    mr.macaque_compute_Qs_to_hdf5(
        outputs_by_day,
        graphs_by_day,
        h5_path=h5_path,
        days_included_idx=days_included_idx,
        mode=args.mode,
        path_graph_dir=path_graphs_dir,
        q_num_workers=q_workers,
    )

    print("Done.")


if __name__ == '__main__':
    main()


