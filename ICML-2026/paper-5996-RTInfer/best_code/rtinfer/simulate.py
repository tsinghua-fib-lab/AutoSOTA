from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from .atlas import AtlasConfig, build_variant_atlas
from .delta_graph import DeltaGraph
from .pantheon_io import load_memory_budget_mib, load_repository, load_tasks
from .scheduler import OnlineScheduler, PolicyName


POLICIES: tuple[PolicyName, ...] = (
    "rms-p",
    "dms-p",
    "pantheon",
    "rtinfer",
    "rtinfer-wo-alc",
    "rtinfer-wo-ms",
    "rtinfer-wo-dlp",
)

DEVICE_PRESETS = {
    "none": {},
    # The memory scale maps the current Pantheon profile MiB range to the
    # paper's multi-GB early-exit deployment footprint.
    "jetson_xavier_nx": {
        "memory_mib": 6144.0,
        "memory_scale": 180.0,
        "latency_scale": 1.0,
        "bandwidth_gbps": 4.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RTInfer server-side reproduction simulation.")
    parser.add_argument("--pantheon-repo", type=Path, default=Path(os.environ.get("PANTHEON_ROOT", "../Pantheon")))
    parser.add_argument("--profile-root", type=Path, default=Path(os.environ.get("PROFILE_ROOT", "../Pantheon_Datasets_Models/3_Exported_JIT_Models")))
    parser.add_argument("--deploy-json", type=Path, required=True)
    parser.add_argument("--workload-json", type=Path, required=True)
    parser.add_argument("--duration-us", type=int, default=1_000_000)
    parser.add_argument("--memory-mib", type=float, default=None)
    parser.add_argument("--memory-scale", type=float, default=None)
    parser.add_argument("--latency-scale", type=float, default=None)
    parser.add_argument("--policy", choices=POLICIES + ("all",), default="all")
    parser.add_argument("--accuracy-cap", type=float, default=0.20)
    parser.add_argument("--bandwidth-gbps", type=float, default=None)
    parser.add_argument("--device-preset", choices=tuple(DEVICE_PRESETS), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = DEVICE_PRESETS[args.device_preset]
    memory_mib = (
        args.memory_mib
        if args.memory_mib is not None
        else float(preset.get("memory_mib", load_memory_budget_mib(args.deploy_json)))
    )
    memory_scale = args.memory_scale if args.memory_scale is not None else float(preset.get("memory_scale", 1.0))
    latency_scale = args.latency_scale if args.latency_scale is not None else float(preset.get("latency_scale", 1.0))
    bandwidth_gbps = args.bandwidth_gbps if args.bandwidth_gbps is not None else float(preset.get("bandwidth_gbps", 4.0))
    models = load_repository(args.pantheon_repo, args.profile_root)
    tasks = load_tasks(args.workload_json)
    delta_graph = DeltaGraph(bandwidth_floor_gbps=bandwidth_gbps)
    atlas = build_variant_atlas(
        models,
        delta_graph,
        AtlasConfig(
            accuracy_cap=args.accuracy_cap,
            latency_scale=latency_scale,
            memory_scale=memory_scale,
        ),
    )
    policies: List[PolicyName] = list(POLICIES) if args.policy == "all" else [args.policy]

    print(
        f"models={len(models)} tasks={len(tasks)} preset={args.device_preset} "
        f"memory_mib={memory_mib:.1f} memory_scale={memory_scale:.2f} "
        f"latency_scale={latency_scale:.2f} bandwidth_gbps={bandwidth_gbps:.2f} "
        f"duration_us={args.duration_us}"
    )
    print("policy,total,dmr,avg_accuracy,avg_latency_ms,avg_load_ms")
    for policy in policies:
        scheduler = OnlineScheduler(models, atlas, memory_mib, delta_graph, policy)
        result = scheduler.run(tasks, args.duration_us)
        print(
            f"{result.policy},{result.total_jobs},{result.deadline_miss_rate:.4f},"
            f"{result.average_accuracy:.4f},{result.average_latency_us / 1000.0:.3f},"
            f"{result.average_load_us / 1000.0:.3f}"
        )


if __name__ == "__main__":
    main()
