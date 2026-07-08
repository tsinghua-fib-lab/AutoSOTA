# Example of how to run

python causalbench_hitl_causal_dpo.py --dataset_npz data/causalbench/exports/weissmann_k562_50.npz --outdir results_cb50 --policy static_eig --seed 123  --T 200 --S 800 --screen_k 800 --rejuvenate_samples --rejuvenate_steps 2
