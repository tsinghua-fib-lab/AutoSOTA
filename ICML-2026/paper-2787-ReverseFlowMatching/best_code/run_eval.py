#!/usr/bin/env python3
"""Run RFM evaluation with config overrides and parse metrics."""
import time, subprocess, sys, json, glob, os


def main():
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []

    log_base = "/autosota_cache/paper2787-logs"

    cmd = [sys.executable, "main.py", "--config", "configs/rfm.yaml"]
    cmd += [
        "--override", "env.name=walker-run",
        "--override", "algo=rfm",
        "--override", "seed=1",
        "--override", "logger.wandb_project=",
        "--override", "logger.debug=true",
        "--override", "logger.log_dir=%s" % log_base,
    ]
    for arg in overrides:
        cmd += ["--override", arg]

    print("Running: %s" % " ".join(cmd), flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/autosota_cache/venv-paper2787/lib/python3.10/site-packages:%s" % env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "osmesa"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_COMPILATION_CACHE_DIR"] = "/autosota_cache/jax_cache"

    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3500, env=env)
    elapsed = time.time() - start

    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)

    wall_time = elapsed / 60.0
    print("TRAINING_WALL_TIME_MINUTES=%.2f" % wall_time)

    # Find and parse metrics.jsonl
    metrics_dir = "%s/walker-run/rfm" % log_base
    run_dirs = sorted(glob.glob("%s/*/metrics.jsonl" % metrics_dir))
    if run_dirs:
        with open(run_dirs[-1]) as f:
            lines = [json.loads(l) for l in f.readlines() if l.strip()]
            if lines:
                last = lines[-1]
                print("FINAL_EPISODE_REWARD=%.2f" % last["episode_reward"])
                print("FINAL_STEP=%s" % last["step"])
                best_reward = max(l["episode_reward"] for l in lines)
                print("BEST_EPISODE_REWARD=%.2f" % best_reward)
    else:
        print("FINAL_EPISODE_REWARD=N/A")

    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
