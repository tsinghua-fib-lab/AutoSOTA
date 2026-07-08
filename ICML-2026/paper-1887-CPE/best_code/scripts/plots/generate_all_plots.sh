# Generate all plots and results for all experiments

# Synthetic
#PYTHONPATH=. python compare_policies_synthetic.py --rootdir=results_final/results_synthetic --policies eig uncertainty random static_eig  --max_round=190

# Sachs
# PYTHONPATH=. python summarize_sachs.py --outdir=results_final/results_sachs --policies eig uncertainty random static_eig

# Causalbench
# PYTHONPATH=. python summarize_cb50.py --outdir=results_final/results_causalbench/cb50 --policies eig uncertainty random static_eig

# Sachs Using DAG GFN as a prior
# PYTHONPATH=. python summarize_sachs.py --outdir=results_final/results_sachs_dag_gfn --policies eig uncertainty random static_eig

# Scalability results
PYTHONPATH=. python compare_policies_synthetic.py --rootdir=results_final/results_scalability/D50 --policies eig uncertainty random static_eig

