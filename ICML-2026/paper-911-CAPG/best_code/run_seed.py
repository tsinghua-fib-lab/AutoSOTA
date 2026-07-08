import sys, os, json, torch, numpy as np
sys.path.insert(0, "/repo")
os.chdir("/repo/experiments/synthetic")
from experiments.synthetic.function_kuairec import setup_data_generation_process, initialize_trainable_policy, train_online_pg_policy

seed = int(sys.argv[1])
dim_emb = int(sys.argv[2])
n_moe = int(sys.argv[3])
lr = float(sys.argv[4])
K = int(sys.argv[5])
n_epoch = int(sys.argv[6])
outfile = sys.argv[7]

device = torch.device("cuda:0")
env = setup_data_generation_process(
    dataset_path="/repo/experiments/synthetic/data/kuairec_small_matrix.csv",
    n_output_action=1, device=device, random_seed=12345)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

policy, _ = initialize_trainable_policy(
    env=env, dim_model_emb=dim_emb, n_moe_model=n_moe,
    device=device, random_seed=seed)

_, logs = train_online_pg_policy(
    env=env, early_stage_policy=policy, early_stage_lr=lr,
    late_stage_optimality="optimal", credit_assignment_type="TOP1",
    is_vanilla_replacement=False, n_epoch=n_epoch, n_epochs_per_log=100,
    n_candidate_action_train=K, n_candidate_action_eval=K,
    device=device, random_seed=seed, use_wandb=False)

val = float(logs["policy_values"][-1].item())
history = [float(x) for x in logs["policy_values"].cpu().tolist()]
result = {"seed": seed, "policy_value": val, "history": history,
          "config": {"dim_emb": dim_emb, "n_moe": n_moe, "lr": lr, "K": K, "n_epoch": n_epoch}}

with open(outfile, "w") as f:
    json.dump(result, f)

print(f"Seed {seed}: policy_value = {val:.4f}", flush=True)
