#!/bin/bash
# PISD Poisson Inverse Evaluation Script (SOTA-optimized)
# Usage: bash eval_sota.sh [N_RUNS] [CONFIG_PATH]
#   N_RUNS: number of evaluation runs (default: 30)
#   CONFIG_PATH: path to YAML config (default: configs/poisson_inverse_u500.yaml)
#
# For full 100-run validation: N_RUNS=100 bash eval_sota.sh

set -e
cd /repo

N_RUNS=${1:-30}
CONFIG=${2:-"configs/poisson_inverse_u500.yaml"}
RESULTS_FILE="eval_results_sota.json"

echo "Starting PISD Poisson inverse evaluation ($N_RUNS runs)..."
echo "Config: $CONFIG"
echo "Model: trained_models/pretrained-poisson_hyperbolic_44-fourier.pkl"

python3 -u -c "
import subprocess, re, os, numpy as np, yaml, json, glob

results = []
for run in range($N_RUNS):
    with open('$CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    config['data']['offset'] = run
    config['generate']['seed'] = run
    cfg_path = f'configs/_eval_run_{run}.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ['python3', 'generate_pde.py', '--config', cfg_path],
        capture_output=True, text=True, cwd='/repo',
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1'},
        timeout=180
    )
    os.remove(cfg_path)
    # Clean up generated .mat file
    for mf in glob.glob(f'poisson_results_{run}_obs_u*.mat'):
        os.remove(mf)

    output = result.stdout + result.stderr

    m_a = re.search(r'relative_error_a:([\d\.e\+\-]+)', output)
    m_pde = re.search(r'L_pde:([\d\.e\+\-]+)', output)
    m_fd = re.search(r'loss_fd:([\d\.e\+\-]+)', output)

    if m_a and m_pde:
        rel_a = float(m_a.group(1)) * 100
        pde = float(m_pde.group(1))
        fd = float(m_fd.group(1)) if m_fd else 0.0
        results.append({'run': run, 'rel_err_a_pct': rel_a, 'pde_res': pde, 'loss_fd': fd})
        print(f'Run {run:3d}/$N_RUNS: Rel.err(a)={rel_a:.4f}%, PDE res.={pde:.6f}, FD={fd:.6f}')
    else:
        print(f'Run {run:3d}/$N_RUNS: FAILED (returncode={result.returncode})')
        print(f'  stdout tail: {output[-500:]}')

if results:
    a = np.array([r['rel_err_a_pct'] for r in results])
    p = np.array([r['pde_res'] for r in results])
    f = np.array([r['loss_fd'] for r in results])
    stats = {
        'n_runs': len(results),
        'rel_err_a_pct_mean': float(a.mean()),
        'rel_err_a_pct_std': float(a.std()),
        'pde_res_mean': float(p.mean()),
        'pde_res_fd_mean': float(f.mean()),
    }
    print(f\"\\nRESULTS: Rel.err(a)={a.mean():.2f}±{a.std():.2f}%, PDE res.={p.mean():.4f}, PDE res.(FD)={f.mean():.4f}\")
    with open('$RESULTS_FILE', 'w') as fp:
        json.dump(stats, fp, indent=2)
else:
    print('No successful runs!')
    exit(1)
"
echo "Evaluation complete. Results saved to $RESULTS_FILE"
