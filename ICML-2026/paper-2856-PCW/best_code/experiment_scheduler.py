#!/usr/bin/env python3
"""
Unified Experiment Scheduler

Runs all watermark experiments (GRID, KLFRONT, DP, OPT, QQ) in parallel.
Spawns run_experiment_combined.py subprocesses with MAX_CONCURRENT_PROCESSES limit.
Skips experiments whose result JSON already exists in OUTPUT_DIR.

Usage:
    python experiment_scheduler.py                          # run all methods
    python experiment_scheduler.py --methods GRID,OPT       # run specific methods
    python experiment_scheduler.py --output-dir results_final
"""

import json
import sys
import time
import subprocess
import re
import threading
import queue
import os

import numpy as np
from pathlib import Path
from collections import deque

from experiment_config import create_config, MODEL_OPTIONS, DATASET_OPTIONS
from theorytools import compute_dp_params, load_klfront_params

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONCURRENT_PROCESSES = 1
OUTPUT_DIR = 'results_final'
VERBOSE = True

MODELS = ['opt-125m', 'gpt2', 'pythia-160m']
DATASETS = ['c4', 'lfqa', 'wikipedia']
GAMMA_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DELTA_LIST = [0.5, 1, 2, 5, 10]
SEED_MAX = 5
TEMPERATURE = 0.7
NUM_SAMPLES = 100

# ============================================================================
# PARAMETER COMPUTATION
# ============================================================================


_compute_dp_params = compute_dp_params
_load_klfront_params = load_klfront_params


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================


def _make_config(model, dataset, seed, gamma, delta, exp_name,
                 watermark_type=None, max_new_tokens=200, output_dir=OUTPUT_DIR,
                 save_mode='compact'):
    """Create experiment config dict."""
    config = create_config(
        model_key=model, dataset_key=dataset,
        num_samples=NUM_SAMPLES, truncate_at=50,
        max_new_tokens=max_new_tokens,
        gamma=gamma, delta=delta,
        experiment_name=exp_name, seed=seed,
        output_dir=output_dir,
    )
    if watermark_type:
        config['watermark_type'] = watermark_type
    config['save_mode'] = save_mode
    return config


def build_experiments(methods=None, output_dir=OUTPUT_DIR, save_mode='compact'):
    """Build experiment list for requested methods. Skip existing results."""
    if methods is None:
        methods = {'GRID', 'KLFRONT', 'DP', 'OPT', 'QQ'}
    else:
        methods = set(m.upper() for m in methods)

    os.makedirs(output_dir, exist_ok=True)
    done = set(os.listdir(output_dir))
    scheduled = set(done)
    experiments = []

    def add(config):
        filename = f"{config['experiment_name']}.json"
        if filename not in scheduled:
            experiments.append(config)
            scheduled.add(filename)

    def mk(model, dataset, seed, gamma, delta, name, **kwargs):
        return _make_config(model, dataset, seed, gamma, delta, name,
                            output_dir=output_dir, save_mode=save_mode, **kwargs)

    t = TEMPERATURE

    # --- GRID: 9 gamma values x 5 delta values + baseline (delta=0) ---
    if 'GRID' in methods:
        for model in MODELS:
            for dataset in DATASETS:
                for seed in range(SEED_MAX):
                    for gamma in GAMMA_LIST:
                        for delta in DELTA_LIST + [0]:
                            name = f"GRID_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}"
                            add(mk(model, dataset, seed, gamma, delta, name))

    # --- KLFRONT: from CSV + baseline (delta=0) ---
    if 'KLFRONT' in methods:
        kl_params = _load_klfront_params()
        for model in MODELS:
            for dataset in DATASETS:
                for seed in range(SEED_MAX):
                    for gamma, delta in kl_params:
                        # Watermarked
                        name = f"KLFRONT_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}"
                        add(mk(model, dataset, seed, gamma, delta, name))
                        # Baseline
                        name_b = f"KLFRONT_m_{model}_d_{dataset}_g{gamma:.2f}_d0.00000_t{t:.2f}_s{seed}"
                        add(mk(model, dataset, seed, gamma, 0, name_b))

    # --- DP: min-KL per deltaP + baseline (delta=0) ---
    if 'DP' in methods:
        dp_params = _compute_dp_params()
        for model in MODELS:
            for dataset in DATASETS:
                for seed in range(SEED_MAX):
                    for gamma, delta in dp_params:
                        # Watermarked
                        name = f"DP_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}"
                        add(mk(model, dataset, seed, gamma, delta, name))
                        # Baseline
                        name_b = f"DP_m_{model}_d_{dataset}_g{gamma:.2f}_d0.00000_t{t:.2f}_s{seed}"
                        add(mk(model, dataset, seed, gamma, 0, name_b))

    # --- OPT: threshold watermark, same grid as GRID, no baseline ---
    if 'OPT' in methods:
        for model in MODELS:
            for dataset in DATASETS:
                for seed in range(SEED_MAX):
                    for gamma in GAMMA_LIST:
                        for delta in DELTA_LIST:
                            name = f"OPT_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}"
                            add(mk(model, dataset, seed, gamma, delta, name,
                                   watermark_type='opt'))

    # --- QQ: normality test, opt-125m x c4, 25 seeds, longer generation ---
    if 'QQ' in methods:
        for seed in range(25):
            for delta in [0, 1]:
                name = f"QQ_m_opt-125m_d_c4_g0.20_d{delta:.5f}_t{t:.2f}_s{seed}"
                add(mk('opt-125m', 'c4', seed, 0.2, delta, name,
                        max_new_tokens=500))

    return experiments


# ============================================================================
# SCHEDULER
# ============================================================================


class SimpleExperimentScheduler:
    """Manage parallel experiment execution with concurrent process limit."""

    def __init__(self, max_concurrent=2, verbose=True):
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.processes = {}
        self.experiment_queue = deque()
        self.completed = []
        self.failed = []
        self.start_time = None
        self.process_progress = {}

    def add_experiments(self, experiments):
        self.experiment_queue.extend(experiments)

    def parse_progress_line(self, line, pid):
        if not line:
            return
        line = line.strip()

        if '[PROGRESS]' in line:
            msg = line.split('[PROGRESS]', 1)[1].strip()
            if pid not in self.process_progress:
                self.process_progress[pid] = {
                    'stage': 'initializing', 'message': '',
                    'samples_done': 0, 'samples_total': 0, 'percent': 0
                }
            p = self.process_progress[pid]
            if 'Loading model:' in msg:
                p['stage'], p['message'] = 'loading_model', msg
            elif 'Model loaded on' in msg:
                p['stage'], p['message'] = 'model_loaded', msg
            elif 'Setting up watermark' in msg:
                p['stage'], p['message'] = 'setup_watermark', msg
            elif 'Loading dataset:' in msg:
                p['stage'], p['message'] = 'loading_dataset', msg
            elif 'Dataset loaded' in msg:
                p['stage'], p['message'] = 'dataset_loaded', msg
            elif 'Processing' in msg and 'sentences' in msg:
                p['stage'] = 'processing'
                match = re.search(r'(\d+)\s+sentences', msg)
                if match:
                    p['samples_total'] = int(match.group(1))
                p['message'] = msg
            elif 'Generated' in msg and 'successfully' in msg:
                p['stage'], p['message'] = 'completed_generation', msg
            elif 'Saving results' in msg:
                p['stage'], p['message'] = 'saving', msg
            elif 'Results saved' in msg:
                p['stage'], p['message'] = 'saved', msg
            else:
                p['message'] = msg

        elif 'Processing:' in line and '%' in line:
            if pid not in self.process_progress:
                self.process_progress[pid] = {
                    'stage': 'processing', 'message': '',
                    'samples_done': 0, 'samples_total': 0, 'percent': 0
                }
            p = self.process_progress[pid]
            m = re.search(r'(\d+)%', line)
            if m:
                p['percent'] = int(m.group(1))
            m = re.search(r'(\d+)/(\d+)', line)
            if m:
                p['samples_done'] = int(m.group(1))
                p['samples_total'] = int(m.group(2))
                p['stage'] = 'processing'

    def _read_process_output(self, process, output_queue):
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_queue.put(line.strip())
        except:
            pass

    def update_process_progress(self):
        for pid, task_info in list(self.processes.items()):
            oq = task_info.get('output_queue')
            if not oq:
                continue
            try:
                while True:
                    try:
                        self.parse_progress_line(oq.get_nowait(), pid)
                    except queue.Empty:
                        break
            except:
                pass

    def start_experiment(self, config):
        config_json = json.dumps(config)
        cmd = [sys.executable, 'run_experiment_combined.py', '--config', config_json, '--quiet']
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            pid = process.pid
            oq = queue.Queue()
            reader = threading.Thread(target=self._read_process_output,
                                      args=(process, oq), daemon=True)
            reader.start()
            self.processes[pid] = {
                'process': process, 'config': config,
                'start_time': time.time(), 'output_queue': oq, 'reader_thread': reader
            }
            self.process_progress[pid] = {
                'stage': 'starting', 'message': 'Process started',
                'samples_done': 0, 'samples_total': config['num_samples'], 'percent': 0
            }
            if self.verbose:
                print(f"[STARTED] {config['experiment_name']} (PID {pid})")
            return pid
        except Exception as e:
            print(f"[ERROR] Failed to start {config['experiment_name']}: {e}")
            return None

    def check_completed(self):
        completed_pids = []
        for pid, task_info in list(self.processes.items()):
            rc = task_info['process'].poll()
            if rc is not None:
                config = task_info['config']
                duration = time.time() - task_info['start_time']
                if rc == 0:
                    self.completed.append(config['experiment_name'])
                    if self.verbose:
                        print(f"[COMPLETE] {config['experiment_name']} ({duration:.1f}s)")
                else:
                    self.failed.append(config['experiment_name'])
                    if self.verbose:
                        print(f"[FAILED] {config['experiment_name']} (code {rc})")
                completed_pids.append(pid)
        for pid in completed_pids:
            del self.processes[pid]
            self.process_progress.pop(pid, None)
        return len(completed_pids) > 0

    def try_start_next(self):
        started = False
        while len(self.processes) < self.max_concurrent and self.experiment_queue:
            config = self.experiment_queue.popleft()
            if self.start_experiment(config):
                started = True
        return started

    def print_status(self):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT SCHEDULER | Concurrent: {self.max_concurrent} | Running: {len(self.processes)}")
        print("=" * 80)

        if self.processes:
            for pid, task_info in self.processes.items():
                config = task_info['config']
                duration = time.time() - task_info['start_time']
                p = self.process_progress.get(pid, {})
                stage = p.get('stage', 'unknown')
                done = p.get('samples_done', 0)
                total = p.get('samples_total', config['num_samples'])
                pct = p.get('percent', 0)

                stage_str = {
                    'starting': 'Starting', 'loading_model': 'Loading Model',
                    'model_loaded': 'Model Ready', 'setup_watermark': 'Setup Watermark',
                    'loading_dataset': 'Loading Dataset', 'dataset_loaded': 'Dataset Ready',
                    'processing': f'Processing ({done}/{total})',
                    'completed_generation': 'Generation Done',
                    'saving': 'Saving', 'saved': 'Saved',
                }.get(stage, stage)

                bar = ""
                if stage == 'processing' and total > 0:
                    w = 20
                    filled = int(w * done / total)
                    bar = f" [{'#' * filled}{'-' * (w - filled)}] {pct}%"

                print(f"  PID {pid} | {duration:.0f}s | {config['model_key']} {config['dataset_key']} "
                      f"gamma={config['gamma']:.2f} delta={config['delta']:.2f} | {stage_str}{bar}")

        queued = len(self.experiment_queue)
        if queued > 0:
            nxt = self.experiment_queue[0]
            print(f"\nQueued ({queued}): next -> {nxt['experiment_name'][:60]}")

        print(f"\nCompleted: {len(self.completed)} | Failed: {len(self.failed)}")
        if self.start_time:
            elapsed = time.time() - self.start_time
            total = len(self.completed) + len(self.failed) + len(self.processes) + queued
            if total > 0:
                print(f"Progress: {(len(self.completed) + len(self.failed)) / total * 100:.1f}% | "
                      f"Elapsed: {elapsed / 60:.1f} min")

    def run(self):
        self.start_time = time.time()
        total = len(self.experiment_queue)
        print(f"\n{'=' * 80}")
        print(f"STARTING SCHEDULER | {total} experiments | max {self.max_concurrent} concurrent")
        print(f"{'=' * 80}\n")

        self.try_start_next()
        last_status = time.time()
        while self.processes or self.experiment_queue:
            self.update_process_progress()
            if self.check_completed():
                self.try_start_next()
            if time.time() - last_status > 10:
                self.print_status()
                last_status = time.time()
            time.sleep(1)

        self.print_status()
        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print(f"DONE | {len(self.completed)}/{total} succeeded | {len(self.failed)} failed | "
              f"{elapsed / 60:.1f} min")
        print(f"{'=' * 80}\n")
        if self.failed:
            print("Failed:")
            for exp in self.failed:
                print(f"  - {exp}")
        return len(self.failed) == 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified experiment scheduler')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods: GRID,KLFRONT,DP,OPT,QQ (default: all)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--max-concurrent', type=int, default=MAX_CONCURRENT_PROCESSES)
    parser.add_argument('--save-mode', type=str, default='compact',
                        choices=['stats', 'compact', 'full'],
                        help='Output mode: stats (summary only), '
                             'compact (tokens + plot/attack metrics + summary, default), '
                             'full (everything including OPT internals)')
    args = parser.parse_args()

    methods = args.methods.split(',') if args.methods else None
    experiments = build_experiments(methods=methods, output_dir=args.output_dir,
                                   save_mode=args.save_mode)

    print(f"Experiments to run: {len(experiments)}")
    print(f"Max concurrent: {args.max_concurrent}")
    if methods:
        print(f"Methods: {', '.join(m.upper() for m in methods)}")
    print()

    if not experiments:
        print("No new experiments to run (all done or none defined).")
        sys.exit(0)

    scheduler = SimpleExperimentScheduler(
        max_concurrent=args.max_concurrent, verbose=VERBOSE
    )
    scheduler.add_experiments(experiments)
    success = scheduler.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
