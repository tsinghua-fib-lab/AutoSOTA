#
# Software Name : learning-parities-with-product-networks
# SPDX-FileCopyrightText: Copyright (c) 2026 Orange S.A.
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License .,
# see the "LICENSE.md" file for more details or https://opensource.org/licenses/MIT
#
# Author: Guillaume Larue, guillaume.larue@orange.com
# Software description: Source code of the paper "Learning High-Dimensional Parity Functions with Product Networks"
#

"""
Run all studies — Multi GPU scheduler
===============================================
Executes every individual study script in isolated subprocesses.

If multiple GPUs are available, studies are dispatched across them:
one study per GPU at a time.  As soon as a GPU finishes its study the
next queued study is launched on that GPU.  On CPU-only machines the
behaviour falls back to sequential execution.

Device injection works via the CUDA_VISIBLE_DEVICES environment variable:
each subprocess sees only its assigned GPU as "cuda:0".

Usage::

    python studies/run_all_studies.py               # run all
    python studies/run_all_studies.py A C F         # run only A, C and F

Studies are executed in the following order:
    A  -> B -> C -> D -> E -> F -> G -> H
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Study registry (label -> script path relative to this file)
# ---------------------------------------------------------------------------
STUDIES_DIR = Path(__file__).parent

STUDY_SCRIPTS = {
    "A":     STUDIES_DIR / "run_study_A.py",
    "B":     STUDIES_DIR / "run_study_B.py",
    "C":     STUDIES_DIR / "run_study_C.py",
    "D":     STUDIES_DIR / "run_study_D.py",
    "E":     STUDIES_DIR / "run_study_E.py",
    "F":     STUDIES_DIR / "run_study_F.py",
    "G":     STUDIES_DIR / "run_study_G.py",
    "H":     STUDIES_DIR / "run_study_H.py",
}

POLL_INTERVAL = 60  # seconds between checks for free GPU slots
LOGS_DIR = STUDIES_DIR / "logs"  # per-study log files

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_gpu_count() -> int:
    """Return the total number of CUDA GPUs visible to this machine (0 if none)."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_free_gpu_ids(our_pids: set[int]) -> list[int]:
    """Return indices of GPUs that have no compute processes other than our own.

    Queries ``nvidia-smi -x -q`` (XML output) to list per-GPU processes.  A GPU
    is considered *free* when every compute process running on it belongs to one
    of *our_pids* (i.e. our own study subprocesses).

    Falls back to an empty list if ``nvidia-smi`` is unavailable or fails.
    """
    import xml.etree.ElementTree as ET
    try:
        r = subprocess.run(
            ["nvidia-smi", "-x", "-q"],
            capture_output=True, text=True, timeout=15,
        )
        root = ET.fromstring(r.stdout)
        free = []
        for gpu in root.findall("gpu"):
            idx = int(gpu.find("minor_number").text)
            processes = gpu.find("processes")
            external = [
                p for p in (processes.findall("process_info") if processes is not None else [])
                if int(p.find("pid").text) not in our_pids
            ]
            if not external:
                free.append(idx)
        return sorted(free)
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def launch_study(label: str, script: Path, gpu_id: int | None) -> tuple[subprocess.Popen, Path]:
    """Launch a study script as a non-blocking subprocess on the given GPU.

    stdout and stderr are redirected to a dedicated log file in ``LOGS_DIR``
    so that concurrent studies do not interleave their output on the terminal.
    The log can be monitored live with ``tail -f logs/study_<label>.log``.

    Args:
        label:   Study label (for display).
        script:  Path to the study script.
        gpu_id:  CUDA device index to assign, or None for CPU-only.

    Returns:
        Tuple of (Popen object, log file Path).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # force line-by-line flushing to log file
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""   # hide all GPUs → forces CPU
        device_str = "cpu"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"study_{label}.log"
    log_file = open(log_path, "w", buffering=1)  # line-buffered

    print(f"  → Launching study {label} on {device_str}  (log: {log_path.relative_to(STUDIES_DIR.parent)})")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(STUDIES_DIR.parent),
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    # Keep file handle alive — it will be closed when the process ends
    proc._log_file = log_file  # type: ignore[attr-defined]
    return proc, log_path


def _fmt(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_all_gpu(selected: dict) -> dict:
    """Dispatch studies across available GPUs, re-evaluating free GPUs at every poll.

    At each iteration the scheduler queries ``nvidia-smi`` to determine which
    GPUs are truly free (no external compute processes).  This means GPUs taken
    or released by other users between two polls are automatically accounted for.
    On CPU-only machines the behaviour falls back to sequential execution.

    Returns a dict {label: success (bool)}.
    """
    cpu_only = (get_gpu_count() == 0)
    if cpu_only:
        print("  No CUDA GPUs detected — running sequentially on CPU.\n")

    queue   = list(selected.items())   # [(label, script), …]
    # running: {gpu_id | None: (label, Popen, t_start, log_path)}
    running: dict = {}
    results = {}
    t_total = time.time()

    while queue or running:
        # ---- Poll finished processes ------------------------------------
        for gpu_id in list(running):
            label, proc, t_start, log_path = running[gpu_id]
            ret = proc.poll()
            if ret is not None:
                proc._log_file.close()  # type: ignore[attr-defined]
                elapsed = time.time() - t_start
                gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                if ret == 0:
                    print(f"  ✓ Study {label} finished in {_fmt(elapsed)}  ({gpu_str})")
                else:
                    print(f"  ✗ Study {label} FAILED (exit={ret}) after {_fmt(elapsed)}  ({gpu_str})")
                    print(f"    └─ see {log_path}")
                results[label] = (ret == 0)
                del running[gpu_id]

        # ---- Determine available slots and launch queued studies --------
        if queue:
            if cpu_only:
                # Sequential: one study at a time on CPU
                available = [None] if not running else []
            else:
                our_pids = {proc.pid for _, proc, _, _ in running.values()}
                free_gpus = get_free_gpu_ids(our_pids)
                # Exclude GPUs we already occupy
                available = [g for g in free_gpus if g not in running]
                if not available:
                    print(f"  No free GPU available right now — will retry in {POLL_INTERVAL}s.",
                          flush=True)
                else:
                    print(f"  Free GPU(s) this poll: {available}")

            for gpu_id in available:
                if not queue:
                    break
                label, script = queue.pop(0)
                print(f"\n{'='*80}")
                gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                print(f"  STARTING STUDY {label}  ({script.name})  on {gpu_str}")
                print(f"{'='*80}")
                proc, log_path = launch_study(label, script, gpu_id)
                running[gpu_id] = (label, proc, time.time(), log_path)

        # ---- Status line -----------------------------------------------
        if running:
            print(f"\n  {'─'*76}")
            print(f"  elapsed {_fmt(time.time()-t_total)}  |  {len(queue)} pending")
            for g, (lbl, _, t_start, log_path) in running.items():
                gpu_str = f"GPU {g}" if g is not None else "CPU"
                study_elapsed = _fmt(time.time() - t_start)
                # Show last non-empty line of the log as a progress hint
                last_line = ""
                try:
                    with open(log_path, "rb") as lf:
                        lf.seek(0, 2)
                        size = lf.tell()
                        lf.seek(max(0, size - 4096))
                        last_line = lf.read().decode(errors="replace").rstrip().rsplit("\n", 1)[-1][:120]
                except OSError:
                    pass
                print(f"  [{lbl}→{gpu_str}] {study_elapsed}  └─ {last_line}", flush=True)
            print(f"  {'─'*76}")

        if queue or running:
            time.sleep(POLL_INTERVAL)

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine which studies to run
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        selected = {}
        for label in requested:
            key = label.upper() if label.upper() in STUDY_SCRIPTS else label
            if key not in STUDY_SCRIPTS:
                print(f"Unknown study: {label}")
                print(f"Available studies: {', '.join(STUDY_SCRIPTS.keys())}")
                sys.exit(1)
            selected[key] = STUDY_SCRIPTS[key]
    else:
        selected = STUDY_SCRIPTS

    print(f"\n{'='*80}")
    print(f"  Scheduling {len(selected)} studies: {', '.join(selected.keys())}")
    print(f"{'='*80}\n")

    t_total = time.time()
    results = run_all_gpu(selected)
    elapsed_total = time.time() - t_total

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    for label, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED "
        print(f"  Study {label:6s} : {status}")
    print(f"\n  Total duration: {_fmt(elapsed_total)}")
    print(f"{'='*80}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
