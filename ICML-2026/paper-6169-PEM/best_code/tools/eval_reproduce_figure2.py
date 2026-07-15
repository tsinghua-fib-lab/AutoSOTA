#!/usr/bin/env python3
"""Reproduce Figure 2 metrics for paper 6169: RB-PEM vs Resample(k=5) on 4 representative bbob-noisy functions (d=40, B=100d)."""
import subprocess, sys, os, glob, shutil
import pandas as pd
import numpy as np

REPO = '/repo'

def run_cmd(cmd_list):
    print(f'  RUN: {" ".join(cmd_list)}')
    result = subprocess.run(cmd_list, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'STDOUT: {result.stdout}')
        print(f'STDERR: {result.stderr}')
        sys.exit(result.returncode)
    for line in result.stdout.strip().split(chr(10)):
        s = line.strip()
        if any(k in s for k in ['COCO INFO','Running','Finished','[   ','Wrote:','Rows:']):
            print(f'    {s}')
    return result.stdout

def main():
    # Clean previous results
    for path in glob.glob(os.path.join(REPO, 'exdata', 'noisy_eval_*')):
        shutil.rmtree(path, ignore_errors=True)
    for path in glob.glob(os.path.join(REPO, 'Results', 'eval_reproduce*')):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    
    print('=== Step 1: Running COCO bbob-noisy experiments ===')
    run_cmd([
        'python3', 'tools/run_coco_bbob_noisy.py',
        '--algorithms', 'BERW-Hetero,CMA-ES-Resample(k=5)',
        '--dims', '40', '--budgets', '100',
        '--functions', '10,13,16,25',
        '--instances', '1-15',
        '--results-dir', 'Results/eval_reproduce', '--tag', 'eval'
    ])
    
    print('=== Step 2: Extracting noise-free deltas ===')
    exdata_dirs = sorted(glob.glob(os.path.join(REPO, 'exdata', 'noisy_eval_*')))
    cmd = ['python3', 'tools/summarize_coco_noisefree_from_exdata.py',
           '--output-dir', 'Results/eval_reproduce_noisefree',
           '--exdata-dirs'] + exdata_dirs
    run_cmd(cmd)
    
    print('=== Step 3: Computing metrics ===')
    df = pd.read_csv(os.path.join(REPO, 'Results/eval_reproduce_noisefree/bbob_summary.csv'))
    rep4 = [110, 113, 116, 125]
    for algo in sorted(df['algorithm'].unique()):
        sub = df[(df['function'].isin(rep4)) & (df['algorithm'] == algo)]
        vals = np.maximum(sub['best_f'].values, 1e-10)
        med_log10 = np.median(np.log10(vals))
        print(f'  {algo}: median log10 regret = {med_log10:.4f}')
    print('=== Reproduction complete ===')

if __name__ == '__main__':
    main()
