#!/usr/bin/env python3
import subprocess

outdir="results/results_sachs_dag_gfn"

common_args = [
    "python", "sachs_hitl_causal_dpo.py",
    "--download",
    "--S", "500",
    "--T", "40",
    "--runs", "10",
    "--policies", "eig,uncertainty,random,static_eig",
    "--beta_edge", "10.0",
    "--beta_dir", "10.0",
    "--lam", "0.0",
    "--screen_k", "200",
    "--rejuvenate_samples",
    "--rejuvenate_steps", "2",
    "--seed0", "123",
    "--outdir", outdir,
    "--use_dag_gfn_prior"
]
subprocess.run(common_args)

# # Plotting results
# plot_args = [
#     "python", "summarize_sachs.py",
#     "--outdir", outdir,
#     "--heatmap_policy", "eig",
#     "--heatmap_which", "directed"
# ]
# subprocess.run(plot_args)
#
# plot_args = [
#     "python", "plot_posterior_sachs.py",
#     "--run_json", outdir + "/sachs_eig_seed123.json",
#     "--meta_json", outdir + "/sachs_meta.json",
#     "--outdir", outdir + "/graphs",
#     "--mode", "topk",
#     "--k", "17",
#     "--which", "final",
# ]
# subprocess.run(plot_args)
