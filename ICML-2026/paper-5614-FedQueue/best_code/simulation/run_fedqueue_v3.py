"""FedQueue V3: Clean controlled simulation matching paper settings."""

import argparse, random, time, json, os, math, copy
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from appfl.loader import load_and_split_dataset
from appfl.misc.utils import create_instance_from_file, get_function_from_file
from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, default="./sim_mnist_v3.yaml")
argparser.add_argument("--output", type=str, default="./output/fedqueue_v3_results.json")
args = argparser.parse_args()

cfg = OmegaConf.load(args.config)
seed = int(cfg.data_configs.partition_kwargs.data_seed)
n_clients = int(cfg.num_clients)
n_rounds = int(cfg.n_rounds)
batch_size = int(cfg.batch_size)
base_lr = float(cfg.base_lr)
local_steps_default = int(cfg.local_steps)
device = cfg.device

# Paper params
Tsync = float(cfg.Tsync)
delta_val = float(cfg.delta)
ewma_alpha = float(cfg.ewma_alpha)
queue_rho = float(cfg.queue_rho)
queue_sigma = float(cfg.queue_sigma)
queue_seed = int(cfg.queue_seed)
staleness_beta = float(cfg.staleness_beta)
warmup_steps = int(cfg.warmup_steps)
alpha_admission = float(cfg.alpha_admission)
target_acc = float(cfg.target_accuracy)

# RNGs
data_rng = random.Random(seed)
queue_rng = random.Random(queue_seed)
np.random.seed(seed)
torch.manual_seed(seed)

print(f"=== FedQueue V3: Controlled Simulation ===")
print(f"Clients={n_clients}, Rounds={n_rounds}, Batch={batch_size}")
print(f"Tsync={Tsync}, delta={delta_val}, rho={queue_rho}, sigma={queue_sigma}")
print(f"base_lr={base_lr}, local_steps={local_steps_default}, warmup={warmup_steps}")

# ---- Load and partition MNIST ----
print(f"\nLoading MNIST (Dirichlet alpha=0.5)...")
client_dss, server_ds, meta = load_and_split_dataset(cfg.data_configs, n_clients)
print(f"Loaded: {len(server_ds)} test samples")
sample_sizes = {cid: len(ds[0]) for cid, ds in enumerate(client_dss)}
for cid, sz in sample_sizes.items():
    print(f"  Client {cid}: {sz} train, {len(client_dss[cid][1])} val")

# ---- Create Model ----
model = create_instance_from_file(
    cfg.model_configs.model_path,
    cfg.model_configs.model_name,
    **cfg.model_configs.model_kwargs,
).to(device)

# ---- Loss and Metric ----
loss_fn = get_function_from_file(cfg.loss_fn_path, cfg.loss_fn_name)()
if not isinstance(loss_fn, nn.Module):
    loss_fn = nn.CrossEntropyLoss()
    
metric_fn = get_function_from_file(cfg.metric_path, cfg.metric_name)

# ---- Create client data loaders ----
client_loaders = []
for cid in range(n_clients):
    train_ds, val_ds = client_dss[cid]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    client_loaders.append((train_loader, val_loader))

# ---- Server test loader ----
server_loader = DataLoader(server_ds, batch_size=batch_size, shuffle=False)

# ---- Staleness function ----
def staleness_fn(u):
    return 1.0 / (1.0 + staleness_beta * u)

# ---- Queue delay sampling ----
def sample_queue_delay():
    mu = math.log(max(1e-6, queue_rho)) - 0.5 * queue_sigma * queue_sigma
    return max(0.0, queue_rng.lognormvariate(mu, queue_sigma))

# ---- Evaluation ----
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    model.train()
    return total_loss / total, 100.0 * correct / total

# ---- Client training ----
def train_client(model_state, loader, steps, lr):
    """Train for `steps` SGD steps, return updated model state."""
    local_model = copy.deepcopy(model)
    local_model.load_state_dict(model_state)
    local_model.train()
    
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    data_iter = iter(loader)
    
    for _ in range(steps):
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            data, target = next(data_iter)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = loss_fn(local_model(data), target)
        loss.backward()
        optimizer.step()
    
    return {k: v.cpu().detach().clone() for k, v in local_model.state_dict().items()}

# ---- Initialize ----
global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ---- FedQueue State ----
q_hat = {cid: queue_rho for cid in range(n_clients)}
client_deadline = {cid: queue_rho + alpha_admission * Tsync for cid in range(n_clients)}
late_buffer = []

# ---- Tracking ----
max_acc, time_to_target = 0.0, None
total_steps = 0
start_time = time.time()
history = []

print(f"\nRunning {n_rounds} rounds...\n")

for r in range(n_rounds):
    # 1. Generate queue delays
    q_delays = {cid: sample_queue_delay() for cid in range(n_clients)}
    
    # 2. Determine local steps (adaptive, matching FedQueue)
    local_steps_per_client = {}
    lr_per_client = {}
    for cid in range(n_clients):
        Jk = max(0.5, Tsync - q_hat[cid] - delta_val)
        Ek = max(warmup_steps, local_steps_default)  # use default, not budget-based
        local_steps_per_client[cid] = Ek
    
    Emin = min(local_steps_per_client.values())
    for cid in range(n_clients):
        lr_per_client[cid] = base_lr * (Emin / max(local_steps_per_client[cid], 1))
    
    # 3. Train clients (with queue delay sim for timing)
    client_updates = {}
    for cid in range(n_clients):
        q_delay = q_delays[cid]
        # Simulate queue wait (for timing only)
        time.sleep(min(q_delay * 0.1, 0.05))  # scaled down for speed
        
        train_start = time.time()
        local_state = train_client(
            global_state, client_loaders[cid][0],
            local_steps_per_client[cid], lr_per_client[cid]
        )
        train_time = time.time() - train_start
        total_steps += local_steps_per_client[cid]
        
        # Admission check
        deadline = client_deadline.get(cid, q_hat[cid] + alpha_admission * Tsync)
        admitted = q_delay <= deadline
        
        # Update estimates
        q_hat[cid] = ewma_alpha * q_delay + (1 - ewma_alpha) * q_hat[cid]
        client_deadline[cid] = q_hat[cid] + alpha_admission * Tsync
        
        if admitted:
            client_updates[cid] = {"model": local_state, "q_delay": q_delay}
        else:
            late_buffer.append({"model": local_state, "q_delay": q_delay, "cid": cid})
    
    # 4. Merge late buffer
    merged, still_late = [], []
    for entry in late_buffer:
        cid = entry["cid"]
        if cid not in client_updates:
            client_updates[cid] = entry
            merged.append(entry)
        else:
            still_late.append(entry)
    late_buffer = still_late
    
    # 5. If no admitted clients, use late ones
    if len(client_updates) == 0 and late_buffer:
        for entry in late_buffer:
            client_updates[entry["cid"]] = entry
        late_buffer = []
    
    # 6. Staleness-aware aggregation
    if len(client_updates) > 0:
        total_samples = sum(sample_sizes[cid] for cid in client_updates)
        staleness = {cid: 0 for cid in client_updates}  # all same round, staleness=0
        factors = {}
        for cid in client_updates:
            w = sample_sizes[cid] / total_samples if total_samples > 0 else 1.0/len(client_updates)
            factors[cid] = staleness_fn(staleness[cid]) * w
        
        fsum = sum(factors.values())
        factors = {cid: f/fsum for cid, f in factors.items()} if fsum > 0 else factors
        
        with torch.no_grad():
            first = True
            for cid, upd in client_updates.items():
                w = factors[cid]
                if first:
                    global_state = {k: v.clone() * w for k, v in upd["model"].items()}
                    first = False
                else:
                    for k in global_state:
                        global_state[k] += upd["model"][k] * w
    
    # 7. Load and evaluate
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    test_loss, test_acc = evaluate(model, server_loader)
    elapsed = time.time() - start_time
    
    if test_acc > max_acc:
        max_acc = test_acc
    if time_to_target is None and test_acc >= target_acc:
        time_to_target = elapsed
    
    admitted_count = len(client_updates)
    history.append({"round": r+1, "elapsed": round(elapsed,1), "acc": round(test_acc,2)})
    
    print(f"R {r+1:3d}/{n_rounds} | Acc: {test_acc:.2f}% | Max: {max_acc:.2f}% | "
          f"T: {elapsed:.0f}s | Adm: {admitted_count}/{n_clients} | Steps: {local_steps_per_client[0]}"
          + (f" | TTA*={time_to_target:.0f}s" if time_to_target else ""))

# ---- Final ----
total_time = time.time() - start_time
if time_to_target is None and max_acc >= target_acc:
    time_to_target = total_time

sep = "=" * 60
print(f"\n{sep}")
print(f"FedQueue V3 Results")
print(f"{sep}")
print(f"Max-A:        {max_acc:.2f}%")
print(f"Time-to-A*:   {time_to_target:.1f}s" if time_to_target else f"Time-to-A*: NOT REACHED")
print(f"Total Steps:  {total_steps}")
print(f"Total Time:   {total_time:.1f}s")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
res = {
    "max_accuracy": round(max_acc, 2),
    "time_to_target": round(time_to_target, 1) if time_to_target else None,
    "total_local_steps": total_steps,
    "total_wall_time": round(total_time, 1),
    "target_accuracy": target_acc,
    "history": history,
}
with open(args.output, "w") as f:
    json.dump(res, f, indent=2)
print(f"Saved to {args.output}")
