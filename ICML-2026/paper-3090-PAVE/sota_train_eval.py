#!/usr/bin/env python3
"""Train and evaluate PAVE+TD3 on Pendulum-v1 for SOTA optimization."""
import sys, os, argparse, json, math, time
import numpy as np

os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/td3/tests")
sys.path.insert(0, "/repo/td3")

from modules.envs import make_pendulum_env
from modules.action_extractor import calculate_smoothness_np
from modules.controller import train_pave
from modules.params import env_timestep, env_args

EVAL_SEEDS = [857751, 968229, 423337, 499844, 985365,
              713160, 643903, 235098, 197317, 212049]


def evaluate_model(model_path, n_episodes=10):
    """Evaluate a trained PAVE model."""
    from models.custom_td3 import CustomTD3
    env = make_pendulum_env()()

    all_returns = []
    all_smoothness = []

    for i in range(n_episodes):
        eval_seed = EVAL_SEEDS[i % len(EVAL_SEEDS)]
        model = CustomTD3.load(model_path, env=env)
        obs, _ = env.reset(seed=eval_seed)
        actions = []
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        sm = calculate_smoothness_np(np.array(actions))
        all_returns.append(total_reward)
        all_smoothness.append(sm)

    env.close()
    return {
        "Cumulative Return": float(np.mean(all_returns)),
        "Cumulative Return std": float(np.std(all_returns)),
        "Smoothness Score": float(np.mean(all_smoothness)),
        "Smoothness Score std": float(np.std(all_smoothness)),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_lamS", type=float, default=2.0)
    parser.add_argument("--grad_lamT", type=float, default=0.005)
    parser.add_argument("--grad_lamC", type=float, default=2.0)
    parser.add_argument("--grad_sigma", type=float, default=0.01)
    parser.add_argument("--grad_delta", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="PAVE loss warmup steps (0=disabled)")
    parser.add_argument("--grad_clip_max_norm", type=float, default=0.0,
                        help="Critic gradient clip max norm (0=disabled)")
    parser.add_argument("--use_huber_loss", type=int, default=0,
                        help="Use Huber loss for critic TD (1=enabled)")
    parser.add_argument("--lr_min", type=float, default=1e-5,
                        help="Minimum LR for cosine schedule")
    parser.add_argument("--lr_max", type=float, default=1e-3,
                        help="Maximum LR for cosine schedule")
    parser.add_argument("--use_cosine_lr", type=int, default=0,
                        help="Use cosine LR schedule (1=enabled)")
    parser.add_argument("--seed", type=int, default=178132)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str,
                        default="/repo/td3/results/pths/pendulum_sota/")
    parser.add_argument("--log_dir", type=str,
                        default="/repo/td3/results/tensorboard_logs/pendulum_sota/")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    pave_args = {
        "grad_lamS": args.grad_lamS,
        "grad_lamT": args.grad_lamT,
        "grad_lamC": args.grad_lamC,
        "grad_sigma": args.grad_sigma,
        "grad_delta": args.grad_delta,
        "warmup_steps": args.warmup_steps,
        "grad_clip_max_norm": args.grad_clip_max_norm,
        "use_huber_loss": bool(args.use_huber_loss),
    }

    env_name = "pendulum"
    save_dir = args.save_dir
    log_dir = args.log_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Build env_args with cosine LR if enabled
    env_args_copy = dict(env_args[env_name])
    if args.use_cosine_lr:
        import math as _math
        lr_min, lr_max = args.lr_min, args.lr_max
        env_args_copy["learning_rate"] = (
            lambda p: lr_min + 0.5 * (lr_max - lr_min)
            * (1 + _math.cos(_math.pi * (1.0 - p)))
        )

    lamS, lamT, lamC = args.grad_lamS, args.grad_lamT, args.grad_lamC
    sig, delta = args.grad_sigma, args.grad_delta
    flags = []
    if args.warmup_steps > 0:
        flags.append(f"warmup={args.warmup_steps}")
    if args.use_cosine_lr:
        flags.append("cosineLR")
    if args.use_huber_loss:
        flags.append("huber")
    if args.grad_clip_max_norm > 0:
        flags.append(f"gradclip={args.grad_clip_max_norm}")
    flag_str = (" [" + ",".join(flags) + "]") if flags else ""

    print(f"[TRAIN] PAVE+TD3: lamS={lamS} lamT={lamT} lamC={lamC} "
          f"sigma={sig} delta={delta} seed={args.seed}{flag_str}")

    t0 = time.time()
    train_pave(
        seed=args.seed,
        total_time_steps=args.timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        mkenv_func=make_pendulum_env,
        env_args=env_args_copy,
        alg_args=pave_args,
        device=args.device,
    )
    train_time = time.time() - t0
    print(f"[TRAIN] Done in {train_time:.1f}s")

    # Find trained model
    suffix = f"_S{lamS}_T{lamT}_C{lamC}_sig{sig}_del{delta}"
    model_path = None
    for d in sorted(os.listdir(save_dir), reverse=True):
        if d.startswith(f"pave_td3{suffix}"):
            candidate = os.path.join(save_dir, d, "final.zip")
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None:
        print(f"[ERROR] Model not found for suffix: {suffix}")
        sys.exit(1)

    print(f"[EVAL] {model_path}")
    t0 = time.time()
    metrics = evaluate_model(model_path, n_episodes=args.n_eval_episodes)
    eval_time = time.time() - t0
    print(f"[EVAL] Done in {eval_time:.1f}s")

    re_mean = metrics["Cumulative Return"]
    re_std = metrics["Cumulative Return std"]
    sm_mean = metrics["Smoothness Score"]
    sm_std = metrics["Smoothness Score std"]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Cumulative Return: {re_mean:.2f} +/- {re_std:.2f}")
    print(f"  Smoothness Score:  {sm_mean:.4f} +/- {sm_std:.4f}")
    print(sep)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVE] {args.output_json}")


if __name__ == "__main__":
    main()
