import argparse
import os, sys
import pickle
import time
import warnings

import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from test import Logger, load_model_weights

from utils import get_dataloader
from envs import MTVRPEnv, MTVRPGenerator

from models import RouteFinderBase
from models import RouteFinderPolicy, LoRAPolicy, MultiLoRAPolicy, CadaPolicy, CadaLoRAPolicy, CadaMultiLoRAPolicy


# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


from tensordict import TensorDict
from math import ceil
import vrplib




# Utils function
def normalize_coord(coord:torch.Tensor) -> torch.Tensor: # if we scale x and y separately, aren't we losing the relative position of the points? i.e. we mess with the distances.
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_scaled = (x - x_min) / (x_max - x_min) 
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled 


def evaluate(policy, td, env,
             num_augment=8,
             num_starts=None,
             ):
    
    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                augment = StateAugmentation(
                    num_augment=num_augment,
                    augment_fn='dihedral8',
                    first_aug_identity=True,
                    feats=None,
                )
                td = augment(td)

            # Evaluate policy
            out = policy(
                td, env, phase="test", num_starts=n_start, return_actions=True
            )

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
                        {"best_multistart_actions": gather_by_index(actions, max_idxs, dim=max_idxs.dim())}
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})
                    
            return out







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
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
    parser.add_argument("--device", type=str, default="cuda")


    parser.add_argument('--model_name', type=str, default='rf_base')
    parser.add_argument('--lora_rank', type=int, nargs='+')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_use_gate', type=int, default=1.0)
    parser.add_argument('--lora_act_func', type=str, default='sigmoid')

    parser.add_argument('--lora_n_experts', type=int, default=4)
    parser.add_argument('--lora_top_k', type=int, default=4)
    parser.add_argument('--lora_temperature', type=float, default=1.0)
    parser.add_argument('--lora_use_trainable_layer', type=int, default=1)
    parser.add_argument('--lora_use_dynamic_topK', type=int, default=0)
    parser.add_argument('--lora_use_basis_variants', type=int, default=0)


    opts = parser.parse_args()
    opts.lora_use_gate = bool(opts.lora_use_gate)
    opts.lora_use_trainable_layer = bool(opts.lora_use_trainable_layer)
    opts.lora_use_dynamic_topK = bool(opts.lora_use_dynamic_topK)
    opts.lora_use_basis_variants = bool(opts.lora_use_basis_variants)

    log_file_name = f"test_vrplib_{opts.model_name}_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}.txt"    
    os.makedirs(opts.log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(opts.log_path, log_file_name), sys.stdout)
    sys.stderr = Logger(os.path.join(opts.log_path, log_file_name), sys.stderr)

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

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
        )
    else:
        raise NotImplementedError

    policy = load_model_weights(policy, opts.checkpoint, device, strict=False)

    generator = MTVRPGenerator(num_loc=100, variant_preset="all")
    env = MTVRPEnv(generator, check_solution=False)
    td_test = env.reset(env.generator(32))
    # Test the model
    with torch.amp.autocast("cuda"):
        with torch.inference_mode():
            out = policy(td_test.clone().to(device), env, phase="test", decode_type="greedy", return_actions=True)
            actions = out['actions'].cpu().detach()
            rewards = out['reward'].cpu().detach()

    # Ensure you have downloaded vrplib under vrplib/
    # Initialize the instances dictionary
    instances = {}
    # Walk through the vrplib directory recursively
    for root, dirs, files in sorted(os.walk(opts.dataset_path)):
        for file in files:
            if file.endswith('.vrp'):
                # Initialize the dictionary for this instance
                instance_name = file[:-4]  # Remove the '.vrp' extension
                instances[instance_name] = {"solution": None}  # Create entry for instance
                # Print the file for verification
                instances[instance_name]["data"] = os.path.join(root, file)  # Save the VRP file path
                instances[instance_name]["solution"] = os.path.join(root, file[:-4] + '.sol')  # Save the solution file path
                # ensure the solution file exists
                assert os.path.exists(instances[instance_name]["solution"]), f"Solution file not found for {instance_name}"
                #print(f"Found VRP file: {file}")
    


    print('\n')
    results = []
    for instance in instances:
        problem = vrplib.read_instance(instances[instance]["data"])

        if problem.get("node_coord", None) is None:
            print(f"Skipping {instance} as it does not have node_coord")
            continue
        coords = torch.tensor(problem['node_coord']).float()
        coords_norm = normalize_coord(coords)

        original_capacity = problem['capacity']
        demand = torch.tensor(problem['demand'][1:]).float() / original_capacity
        original_capacity = torch.tensor(original_capacity)[None]         

        # Make instance
        td_instance = TensorDict({
            "locs": coords_norm,
            "demand_linehaul": demand,
            "capacity_original": original_capacity,
        },
        batch_size=[])[None]

        td_reset = env.reset(td_instance).to(device)
        
        start = time.time()
        actions = evaluate(policy, td_reset.clone(), env)["best_aug_actions"]
        inference_time = time.time() - start        

        # Obtain reward from the environment with new locs
        td_reset["locs"] = coords[None] # unnormalized
        reward = env.get_reward(td_reset, actions)
        
        # Load the optimal cost
        solution = vrplib.read_solution(instances[instance]["solution"])
        optimal_cost = solution['cost'] # note that this cost is somehow slightly lower than the one calculated from the distance matrix
        
        # Calculate the gap and print
        cost = ceil(-reward.item())
        gap = (cost - optimal_cost) / optimal_cost
        print(f'Problem: {instance:<15} Cost: {cost:<10} Optimal Cost: {optimal_cost:<10}\t Gap: {gap:.3%} Inference Time: {inference_time:.0}')
        
        results.append({
            "instance": instance,
            "cost": cost,
            "optimal_cost": optimal_cost,
            "gap": gap,
            "inference_time": inference_time,
        })



    print('\n')
    gaps_names = {}
    for instance in results:
        name = instance['instance'][0]
        # if name == 'X':
        #     num_locs = instance['instance'].split('-')[1][1:]
        #     if int(num_locs) < 252:
        #         name = 'X<251'
        #     elif int(num_locs) >= 252 and int(num_locs) <= 501:
        #         name = 'X251-501'
        #     else:
        #         name = 'X501-1000'
        if name not in gaps_names:
            # if X, then we divide between < 251 and >= 251
            gaps_names[name] = []
        gaps_names[name].append(instance['gap'])
            
    # Calculate the average gap for each group
    average_gaps = {name: sum(gaps) / len(gaps) for name, gaps in gaps_names.items()}
    for name, gap in average_gaps.items():
        print(f'Group {name}: {(gap):.3%} %')


    print('\n')
    gaps_names = {}
    for instance in results:
        name = instance['instance'][0]
        if name == 'X':
            num_locs = instance['instance'].split('-')[1][1:]
            if int(num_locs) < 252:
                name = 'X<251'
            elif int(num_locs) >= 252 and int(num_locs) <= 501:
                name = 'X251-501'
            else:
                name = 'X501-1000'
        if name not in gaps_names:
            # if X, then we divide between < 251 and >= 251
            gaps_names[name] = []
        gaps_names[name].append(instance['gap'])
            
    # Calculate the average gap for each group
    average_gaps = {name: sum(gaps) / len(gaps) for name, gaps in gaps_names.items()}
    for name, gap in average_gaps.items():
        print(f'Group {name}: {(gap):.3%} %')
        
            

    


