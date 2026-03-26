import argparse
import os, sys
import pickle
import sys
import time
import warnings

# Compatibility shim for newer torchrl versions
try:
    import torchrl.data.tensor_specs as _ts
    if not hasattr(_ts, 'CompositeSpec'):
        _ts.CompositeSpec = _ts.Composite
        _ts.BoundedTensorSpec = _ts.Bounded
        _ts.UnboundedContinuousTensorSpec = _ts.UnboundedContinuous
        _ts.UnboundedDiscreteTensorSpec = _ts.UnboundedDiscrete
        import torchrl.data as _td
        _td.CompositeSpec = _ts.Composite
        _td.BoundedTensorSpec = _ts.Bounded
        _td.UnboundedContinuousTensorSpec = _ts.UnboundedContinuous
        _td.UnboundedDiscreteTensorSpec = _ts.UnboundedDiscrete
except Exception:
    pass
from collections import OrderedDict
import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from utils import get_dataloader


def apply_2opt_batch(actions, locs, demands=None, capacities=None, max_iters=10):
    """
    Apply 2-opt + Or-opt improvement for CVRP.
    args:
        actions: (B, seq_len) - sequences with 0=depot as route separators
        locs: (B, N+1, 2) - locations (0=depot)
        demands: (B, N+1) - demand per node (0=depot has 0)
        capacities: (B,) - vehicle capacity
    Returns: improved actions tensor
    """
    import numpy as np
    B = actions.shape[0]
    device = actions.device
    actions_np = actions.cpu().numpy().astype(np.int32)
    locs_np = locs.cpu().float().numpy()
    demands_np = demands.cpu().float().numpy() if demands is not None else None
    caps_np = capacities.cpu().float().numpy() if capacities is not None else None
    improved = actions_np.copy()

    def dist(a, b, pts):
        d = pts[a] - pts[b]
        return np.sqrt(d[0]*d[0] + d[1]*d[1])

    for b in range(B):
        seq = actions_np[b]
        pts = locs_np[b]

        # Parse into routes: list of lists of customer node indices
        routes = []
        curr_route = []
        for node in seq:
            if node == 0:
                if curr_route:
                    routes.append(curr_route)
                    curr_route = []
            else:
                curr_route.append(int(node))
        if curr_route:
            routes.append(curr_route)

        def route_length(route):
            if not route:
                return 0.0
            cost = dist(0, route[0], pts)
            for i in range(len(route)-1):
                cost += dist(route[i], route[i+1], pts)
            cost += dist(route[-1], 0, pts)
            return cost

        def route_demand(route):
            if demands_np is None:
                return 0.0
            return sum(demands_np[b][n] for n in route)

        cap = caps_np[b] if caps_np is not None and caps_np[b] > 0 else 1e9

        # Intra-route 2-opt
        for ri, route in enumerate(routes):
            if len(route) <= 2:
                continue
            best = route[:]
            best_cost = route_length(best)
            for _ in range(max_iters):
                changed = False
                n = len(best)
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                        c = route_length(cand)
                        if c < best_cost - 1e-7:
                            best = cand
                            best_cost = c
                            changed = True
                if not changed:
                    break
            routes[ri] = best

        # Or-opt: try relocating a single node from one route to another
        improved_overall = True
        n_passes = 10
        for _ in range(n_passes):
            improved_overall = False
            for ri in range(len(routes)):
                for ci in range(len(routes[ri])):
                    node = routes[ri][ci]
                    node_dem = demands_np[b][node] if demands_np is not None else 0.0
                    # Cost of removing node from route ri
                    r = routes[ri]
                    n = len(r)
                    prev_n = r[ci-1] if ci > 0 else 0
                    next_n = r[ci+1] if ci < n-1 else 0
                    
                    if n == 1:
                        cost_remove = route_length(r) - dist(0, node, pts) - dist(node, 0, pts)
                    elif ci == 0:
                        cost_remove = -dist(0, r[0], pts) - dist(r[0], r[1], pts) + dist(0, r[1], pts)
                    elif ci == n-1:
                        cost_remove = -dist(r[-2], r[-1], pts) - dist(r[-1], 0, pts) + dist(r[-2], 0, pts)
                    else:
                        cost_remove = -dist(r[ci-1], r[ci], pts) - dist(r[ci], r[ci+1], pts) + dist(r[ci-1], r[ci+1], pts)

                    # Try inserting node into each other route
                    for rj in range(len(routes)):
                        if rj == ri and len(routes[ri]) == 1:
                            continue
                        if rj == ri:
                            continue
                        # Check capacity
                        if demands_np is not None and route_demand(routes[rj]) + node_dem > cap + 1e-6:
                            continue
                        rj_route = routes[rj]
                        m = len(rj_route)
                        # Try each insertion position
                        for pos in range(m + 1):
                            if pos == 0:
                                cost_insert = dist(0, node, pts) + dist(node, rj_route[0], pts) - dist(0, rj_route[0], pts)
                            elif pos == m:
                                cost_insert = dist(rj_route[-1], node, pts) + dist(node, 0, pts) - dist(rj_route[-1], 0, pts)
                            else:
                                cost_insert = dist(rj_route[pos-1], node, pts) + dist(node, rj_route[pos], pts) - dist(rj_route[pos-1], rj_route[pos], pts)
                            
                            if cost_remove + cost_insert < -1e-7:
                                # Apply the move
                                new_ri = routes[ri][:ci] + routes[ri][ci+1:]
                                new_rj = routes[rj][:pos] + [node] + routes[rj][pos:]
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved_overall = True
                                break
                        if improved_overall:
                            break
                    if improved_overall:
                        break
                if improved_overall:
                    break

        # Reconstruct sequence
        new_seq = []
        for route in routes:
            if route:
                new_seq.extend(route)
                new_seq.append(0)
        
        # Pad/trim to original length
        orig_len = len(seq)
        if len(new_seq) < orig_len:
            new_seq.extend([0] * (orig_len - len(new_seq)))
        elif len(new_seq) > orig_len:
            new_seq = new_seq[:orig_len]
        
        improved[b] = np.array(new_seq, dtype=np.int32)
    
    return torch.tensor(improved, dtype=actions.dtype, device=device)
from envs import MTVRPEnv
from models import RouteFinderBase
from models import RouteFinderPolicy, LoRAPolicy, MultiLoRAPolicy, CadaPolicy, CadaLoRAPolicy, CadaMultiLoRAPolicy


# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


class Logger(object):
    def __init__(self, file_name, stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'ab', buffering=0)
        self.log_disable = False
    def write(self, message):
        self.terminal.write(str(message))
        if not self.log_disable:
            self.log.write(str(message).encode("utf-8"))
    def flush(self):
        self.terminal.flush()
        if not self.log_disable:
            self.log.flush()
    def close(self):
        self.log.close()
    def disable_log(self):
        self.log_disable = True
        self.close()



def test(
        policy,
        td,
        env,
        num_augment=8,
        augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
        num_starts=None,
        device="cuda",
):
    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                            100
                            * (-max_aug_reward - torch.abs(costs_bks))
                            / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out



def load_model_weights(policy, path, device, strict=True):
    _policy_weights = torch.load(path, map_location=torch.device("cpu"), weights_only=False)['state_dict']
    policy_weights = {}
    for name, weight in _policy_weights.items():
        assert name.split('.')[0] == 'policy'
        policy_weights[name.lstrip('policy.')] = weight
    policy.load_state_dict(policy_weights, strict=strict)        
    policy = policy.to(device).eval()
    return policy




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data'
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='logs'
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint}",
    )

    parser.add_argument('--model_name', type=str, default='rf_base')
    parser.add_argument('--lora_rank', type=int, nargs='+')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_use_gate', type=int, default=1)
    parser.add_argument('--lora_act_func', type=str, default='sigmoid')

    parser.add_argument('--lora_n_experts', type=int, default=4)
    parser.add_argument('--lora_top_k', type=int, default=4)
    parser.add_argument('--lora_temperature', type=float, default=1.0)
    parser.add_argument('--lora_use_trainable_layer', type=int, default=1)
    parser.add_argument('--lora_use_dynamic_topK', type=int, default=0)
    parser.add_argument('--lora_use_basis_variants', type=int, default=0)
    parser.add_argument('--lora_use_basis_variants_as_input', type=int, default=0)
    parser.add_argument('--lora_use_linear', type=int, default=0)
    parser.add_argument('--num_augment', type=int, default=8)
    parser.add_argument('--augment_fn', type=str, default='dihedral8')



    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()
    opts.lora_use_gate = bool(opts.lora_use_gate)
    opts.lora_use_trainable_layer = bool(opts.lora_use_trainable_layer)
    opts.lora_use_dynamic_topK = bool(opts.lora_use_dynamic_topK)
    opts.lora_use_basis_variants = bool(opts.lora_use_basis_variants)
    opts.lora_use_basis_variants_as_input = bool(opts.lora_use_basis_variants_as_input)
    opts.lora_use_linear = bool(opts.lora_use_linear)

    log_file_name = f"test_{opts.size}_{opts.model_name}_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.txt"
    os.makedirs(opts.log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(opts.log_path, log_file_name), sys.stdout)
    sys.stderr = Logger(os.path.join(opts.log_path, log_file_name), sys.stderr)

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        # list recursively all npz files in data/
        data_paths = []
        for root, _, files in os.walk(opts.dataset_path):
            for file in files:
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    if opts.remove_mixed_backhaul and "m" in root and opts.size <= 100:
                        continue
                    if str(opts.size) in file:
                        if file == f"{opts.size}.npz":
                            data_paths.append(os.path.join(root, file))

        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

        ordered_tasks = [
            "cvrp", "vrptw", "ovrp", "vrpl",
            "vrpb", "ovrptw", "vrpbl", "vrpbltw",
            "vrpbtw", "vrpltw", "ovrpb", "ovrpbl",
            "ovrpbltw", "ovrpbtw", "ovrpl", "ovrpltw",
        ]
        ordered_paths = [f"{opts.dataset_path}/{task}/test/{opts.size}.npz" for task in ordered_tasks]
        data_paths = [ordered_p for ordered_p in ordered_paths if ordered_p in data_paths]


    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if opts.model_name == 'rf_base':
        policy = RouteFinderPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            }
        )
    elif opts.model_name == 'cada_base':
        policy = CadaPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
        )
    elif opts.model_name == 'rf_lora':
        policy = LoRAPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_use_gate=opts.lora_use_gate,
            lora_act_func=opts.lora_act_func,
        )
    elif opts.model_name == 'cada_lora':
        policy = CadaLoRAPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_use_gate=opts.lora_use_gate,
            lora_act_func=opts.lora_act_func,
        )
    elif opts.model_name == 'rf_multilora':
        assert opts.lora_act_func in ['softmax','softplus','sigmoid']
        policy = MultiLoRAPolicy(
            normalization='rms',
            encoder_use_prenorm=True,
            encoder_use_post_layers_norm=True,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_act_func=opts.lora_act_func,
            lora_n_experts=opts.lora_n_experts,
            lora_top_k=opts.lora_top_k,
            lora_temperature=opts.lora_temperature,
            lora_use_trainable_layer=opts.lora_use_trainable_layer,
            lora_use_dynamic_topK=opts.lora_use_dynamic_topK,
            lora_use_basis_variants=opts.lora_use_basis_variants,
            lora_use_basis_variants_as_input=opts.lora_use_basis_variants_as_input,
            lora_use_linear=opts.lora_use_linear,
        )
    elif opts.model_name == 'cada_multilora':
        assert opts.lora_act_func in ['softmax','softplus','sigmoid']
        policy = CadaMultiLoRAPolicy(
            normalization="rms",
            encoder_use_prenorm=False,
            encoder_use_post_layers_norm=False,
            parallel_gated_kwargs={
                "mlp_activation": "silu"
            },
            attn_sparse_ratio=0.5,
            sparse_applied_to_score=True,
            lora_rank=opts.lora_rank,
            lora_alpha=opts.lora_alpha,
            lora_act_func=opts.lora_act_func,
            lora_n_experts=opts.lora_n_experts,
            lora_top_k=opts.lora_top_k,
            lora_temperature=opts.lora_temperature,
            lora_use_trainable_layer=opts.lora_use_trainable_layer,
            lora_use_dynamic_topK=opts.lora_use_dynamic_topK,
            lora_use_basis_variants=opts.lora_use_basis_variants,
            lora_use_basis_variants_as_input=opts.lora_use_basis_variants_as_input,
            lora_use_linear=opts.lora_use_linear,
        )
    else:
        raise NotImplementedError

    policy = load_model_weights(policy, opts.checkpoint, device, strict=False)

    env = MTVRPEnv()
    results = {}
    for dataset in data_paths:
        print("\n")

        td_test = env.load_data(dataset)  # this also adds the bks cost
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in dataloader:
            td_test = env.reset(batch).to(device)
            o = test(policy, td_test, env, device=device, num_augment=opts.num_augment, augment_fn=opts.augment_fn)
            res.append(o)
        
        out = {}
        out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
        gaps = [o["gap_to_bks"] for o in res]
        if all(isinstance(g, torch.Tensor) for g in gaps):
            out["gap_to_bks"] = torch.cat(gaps)
        else:
            out["gap_to_bks"] = None
        
        # 2-opt post-processing on best actions
        if all(o.get("best_aug_actions", None) is not None for o in res):
            try:
                td_test_full = env.load_data(dataset)
                dataloader_full = get_dataloader(td_test_full, batch_size=opts.batch_size)
                improved_rewards = []
                improved_gaps = []
                for i, (batch_o, batch_td) in enumerate(zip(res, dataloader_full)):
                    best_actions = batch_o["best_aug_actions"]  # (B, seq_len)
                    td_b = env.reset(batch_td).to(device)
                    # Apply 2-opt
                    improved_actions = apply_2opt_batch(best_actions, td_b["locs"], demands=td_b.get("demand_linehaul", None), capacities=td_b.get("vehicle_capacity", None) if td_b.get("vehicle_capacity", None) is not None and td_b.get("vehicle_capacity", None).dim() > 0 else None, max_iters=3)
                    # Recompute reward with environment
                    new_reward = env._get_reward(td_b, improved_actions)
                    # Compare with original best
                    orig_reward = batch_o["max_aug_reward"]  # (B,)
                    better_mask = new_reward > orig_reward
                    final_reward = torch.where(better_mask, new_reward, orig_reward)
                    improved_rewards.append(final_reward)
                    if out["gap_to_bks"] is not None:
                        bks = td_b.get("costs_bks", None)
                        if bks is not None:
                            new_gap = 100 * (-final_reward - torch.abs(bks)) / torch.abs(bks)
                            improved_gaps.append(new_gap)
                if improved_rewards:
                    out["max_aug_reward"] = torch.cat(improved_rewards)
                if improved_gaps:
                    out["gap_to_bks"] = torch.cat(improved_gaps)
            except Exception as e:
                pass  # fall back to non-2opt if any error

        inference_time = time.time() - start

        dataset_name = dataset.split("/")[-3].split(".")[0].upper()
        if out["gap_to_bks"] is not None:
            gap_str = f"{out['gap_to_bks'].mean().item():.3f}%"
        else:
            gap_str = "N/A (no BKS)"
        print(
            f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {gap_str} | Inference time: {inference_time:.3f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = out["gap_to_bks"].mean().item() if out["gap_to_bks"] is not None else None
        results[dataset_name]["inference_time"] = inference_time


    if opts.save_results:
        # Save results with checkpoint name under results/main/
        checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))


