"""
Q11: FPS measurement for TD3 methods
Measures training throughput (FPS) for each algorithm.
Run from PAVE_Merge root: python td3/tests/measure_fps.py --env lunar --alg vanilla
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modules.controller import train_vanilla, train_caps, train_grad, train_pave, train_asap
from modules.envs import make_lunar_env, make_walker_env
from modules.params import env_args, alg_args

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, choices=["lunar", "walker"])
parser.add_argument("--alg", type=str, required=True, choices=["vanilla", "caps", "grad", "pave", "asap"])
parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to run")
parser.add_argument("--device", type=str, default="auto")
args = parser.parse_args()

env_funcs = {"lunar": make_lunar_env, "walker": make_walker_env}
alg_funcs = {
    "vanilla": train_vanilla,
    "caps": train_caps,
    "grad": train_grad,
    "pave": train_pave,
    "asap": train_asap,
}

save_dir = f"./td3/results/pths_fps/{args.env}/"
log_dir = f"./td3/results/tensorboard_fps/{args.env}/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

seed = 42
total_timesteps = args.timesteps

print(f"=== FPS Measurement: {args.alg} on {args.env}, {total_timesteps} steps ===")
print(f"Device: {args.device}")

start_time = time.time()
alg_funcs[args.alg](
    seed, total_timesteps, save_dir, log_dir,
    env_funcs[args.env], env_args[args.env],
    alg_args[args.alg][args.env], args.device
)
elapsed = time.time() - start_time

fps = total_timesteps / elapsed
print(f"\n{'='*50}")
print(f"Algorithm: {args.alg}")
print(f"Environment: {args.env}")
print(f"Total timesteps: {total_timesteps}")
print(f"Elapsed time: {elapsed:.1f}s")
print(f"FPS: {fps:.1f}")
print(f"{'='*50}")
