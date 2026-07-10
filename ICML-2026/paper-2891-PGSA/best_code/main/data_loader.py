"""Load source/target graph pairs for PSAHS benchmarks."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from psahs.data import datasets
from psahs.paths import dataset_root, noncircle_default_pickle_paths


def load_domain_pair(args, *, adjust_source: bool = False):
    """Return (source_graph, target_graph) for the configured dataset."""
    data_root = dataset_root()

    if args.dataset == "Blog":
        source = datasets.load_data_from_mat(os.path.join(data_root, "Blog"), args.src_name)
        target = datasets.load_data_from_mat(os.path.join(data_root, "Blog"), args.tgt_name)
    elif args.dataset == "Twitch":
        twitch_root = os.path.join(data_root, "Twitch")
        source = datasets.prepare_Twitch(twitch_root, args.src_name)
        target = datasets.prepare_Twitch(twitch_root, args.tgt_name)
    elif args.dataset == "dblp_acm":
        source = datasets.prepare_dblp_acm(data_root, args.src_name)
        target = datasets.prepare_dblp_acm(data_root, args.tgt_name)
    elif args.dataset == "Airport":
        airport_root = os.path.join(data_root, "Airport")
        source = datasets.prepare_airport(airport_root, args.src_name)
        target = datasets.prepare_airport(airport_root, args.tgt_name)
    elif args.dataset == "Noncircle":
        src_pkl, tgt_pkl = noncircle_default_pickle_paths(args)
        src_pkl = args.noncircle_src or src_pkl
        tgt_pkl = args.noncircle_tgt or tgt_pkl
        source = datasets.prepare_noncircle_pickle(src_pkl)
        target = datasets.prepare_noncircle_pickle(tgt_pkl)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if adjust_source:
        datasets.adjust_graph_structure_fast_source(source, h_thresh=args.h_threshold)

    return source, target
