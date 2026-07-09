"""
Separate two runs that landed in the same output folder.
Run 1 (2026-03-25) -> 0.1lam3
Run 2 (2026-03-26) -> 0.001lam3
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

BASE = "outputs"
OLD_PREFIX = "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_lam3_ms100"
NEW_PREFIX_RUN1 = "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.1lam3_ms100"
NEW_PREFIX_RUN2 = "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.001lam3_ms100"

CUTOFF = datetime(2026, 3, 26, 0, 0, 0)

DATA_SEEDS = range(42, 62)
SIM_IDS = range(12)

dry_run = False  # Set to False to actually move files

for ds in DATA_SEEDS:
    old_dir = Path(BASE) / f"{OLD_PREFIX}-ds{ds}"
    if not old_dir.exists():
        print(f"SKIP: {old_dir} not found")
        continue

    dir_run1 = Path(BASE) / f"{NEW_PREFIX_RUN1}-ds{ds}"
    dir_run2 = Path(BASE) / f"{NEW_PREFIX_RUN2}-ds{ds}"

    for sim in SIM_IDS:
        sim_src = old_dir / str(sim)
        if not sim_src.exists():
            continue

        sim_dst1 = dir_run1 / str(sim)
        sim_dst2 = dir_run2 / str(sim)

        # Create target dirs
        for d in [sim_dst1, sim_dst2,
                  sim_dst1 / "checkpoints", sim_dst2 / "checkpoints",
                  sim_dst1 / "wandb", sim_dst2 / "wandb",
                  sim_dst1 / ".hydra", sim_dst2 / ".hydra",
                  sim_dst1 / "metrics", sim_dst2 / "metrics"]:
            if not dry_run:
                d.mkdir(parents=True, exist_ok=True)

        # --- Checkpoints: split by date ---
        ckpt_dir = sim_src / "checkpoints"
        if ckpt_dir.exists():
            for f in ckpt_dir.iterdir():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < CUTOFF:
                    dst = sim_dst1 / "checkpoints" / f.name
                else:
                    dst = sim_dst2 / "checkpoints" / f.name
                if dry_run:
                    print(f"  cp {f.name} -> {'run1' if mtime < CUTOFF else 'run2'} ({mtime.date()})")
                else:
                    shutil.copy2(f, dst)

        # --- WANDB: split by run dir name ---
        wandb_dir = sim_src / "wandb"
        if wandb_dir.exists():
            for item in wandb_dir.iterdir():
                if item.name.startswith("run-20260325"):
                    dst = sim_dst1 / "wandb" / item.name
                    if dry_run:
                        print(f"  wandb {item.name} -> run1")
                    else:
                        shutil.copytree(item, dst, dirs_exist_ok=True, ignore_dangling_symlinks=True, copy_function=shutil.copy2)
                elif item.name.startswith("run-20260326"):
                    dst = sim_dst2 / "wandb" / item.name
                    if dry_run:
                        print(f"  wandb {item.name} -> run2")
                    else:
                        shutil.copytree(item, dst, dirs_exist_ok=True, ignore_dangling_symlinks=True, copy_function=shutil.copy2)
                # skip debug logs, latest-run symlink

        # --- training_stats.json: split two lines ---
        stats_file = sim_src / "training_stats.json"
        if stats_file.exists():
            lines = stats_file.read_text().strip().split("\n")
            if len(lines) == 2:
                if not dry_run:
                    (sim_dst1 / "training_stats.json").write_text(lines[0] + "\n")
                    (sim_dst2 / "training_stats.json").write_text(lines[1] + "\n")
                else:
                    print(f"  training_stats.json -> split 2 lines")

        # --- train.log: split by timestamp ---
        log_file = sim_src / "train.log"
        if log_file.exists():
            content = log_file.read_text()
            log_lines = content.split("\n")
            run1_lines = []
            run2_lines = []
            for line in log_lines:
                if line.startswith("[2026-03-25") or (run1_lines and not line.startswith("[2026-03-26") and not run2_lines):
                    run1_lines.append(line)
                elif line.startswith("[2026-03-26") or run2_lines:
                    run2_lines.append(line)
            if not dry_run:
                (sim_dst1 / "train.log").write_text("\n".join(run1_lines))
                (sim_dst2 / "train.log").write_text("\n".join(run2_lines))
            else:
                print(f"  train.log -> split ({len(run1_lines)}/{len(run2_lines)} lines)")

        # --- Other files: copy to run2 (they were overwritten) ---
        # run_identity.json, runtime_info.json, hardware_info.json, dgp_args.pt, .hydra/
        for fname in ["run_identity.json", "runtime_info.json", "hardware_info.json", "dgp_args.pt"]:
            src = sim_src / fname
            if src.exists():
                if not dry_run:
                    shutil.copy2(src, sim_dst1 / fname)
                    shutil.copy2(src, sim_dst2 / fname)

        # .hydra config (was overwritten by run 2, copy as-is to both)
        hydra_dir = sim_src / ".hydra"
        if hydra_dir.exists():
            for f in hydra_dir.iterdir():
                if not dry_run:
                    shutil.copy2(f, sim_dst1 / ".hydra" / f.name)
                    shutil.copy2(f, sim_dst2 / ".hydra" / f.name)

        # metrics dir: split by date
        metrics_dir = sim_src / "metrics"
        if metrics_dir.exists():
            for f in metrics_dir.iterdir():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < CUTOFF:
                    if not dry_run:
                        shutil.copy2(f, sim_dst1 / "metrics" / f.name)
                else:
                    if not dry_run:
                        shutil.copy2(f, sim_dst2 / "metrics" / f.name)

    if dry_run:
        print(f"\n--- ds{ds}: would create {dir_run1.name} and {dir_run2.name}")
        # Only print details for first seed
        if ds == 42:
            print("(showing details for ds42 only)\n")
        break  # Only show first seed in dry run

print("\nDone!" if not dry_run else "\n=== DRY RUN === Set dry_run=False to execute.")
