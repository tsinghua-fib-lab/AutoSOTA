#!/usr/bin/env python3
"""Parse eval log and record score."""
import subprocess, sys, re, json


def parse_log(log_path):
    """Extract metrics from eval log output."""
    with open(log_path) as f:
        content = f.read()

    training_time = None
    final_reward = None
    best_reward = None
    final_step = None

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("TRAINING_WALL_TIME_MINUTES="):
            training_time = float(line.split("=")[1])
        elif line.startswith("FINAL_EPISODE_REWARD="):
            val = line.split("=")[1]
            if val != "N/A":
                final_reward = float(val)
        elif line.startswith("BEST_EPISODE_REWARD="):
            val = line.split("=")[1]
            if val != "N/A":
                best_reward = float(val)
        elif line.startswith("FINAL_STEP="):
            val = line.split("=")[1]
            if val != "N/A":
                final_step = int(val)

    return {
        "training_time": training_time,
        "final_reward": final_reward,
        "best_reward": best_reward,
        "final_step": final_step,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: record_from_log.py <log_path> --iter N --idea-id ID --title TITLE [--is-best true/false]")
        sys.exit(1)

    log_path = sys.argv[1]

    # Parse additional args
    args = {}
    i = 2
    while i < len(sys.argv):
        key = sys.argv[i]
        if key.startswith("--"):
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                args[key[2:]] = sys.argv[i + 1]
                i += 2
            else:
                args[key[2:]] = "true"
                i += 1
        else:
            i += 1

    metrics = parse_log(log_path)

    training_time = metrics["training_time"]
    final_reward = metrics["final_reward"]

    if training_time is None:
        print("ERROR: Could not parse TRAINING_WALL_TIME_MINUTES from log")
        status = "failed"
        primary = 0.0
        metrics_json = "{}"
        notes = "Failed to parse training time from eval output."
    elif final_reward is None:
        print("ERROR: Could not parse FINAL_EPISODE_REWARD from log")
        status = "failed"
        primary = 0.0
        metrics_json = "{}"
        notes = "Failed to parse episode reward from eval output."
    else:
        status = "success"
        primary = training_time
        metrics_dict = {
            "training_time_minutes": training_time,
            "final_episode_reward": final_reward,
        }
        if metrics["best_reward"] is not None:
            metrics_dict["best_episode_reward"] = metrics["best_reward"]
        if metrics["final_step"] is not None:
            metrics_dict["final_step"] = metrics["final_step"]
        metrics_json = json.dumps(metrics_dict)
        notes = args.get("notes", "")

    iter_num = args.get("iter", "?")
    idea_id = args.get("idea-id", "unknown")
    title = args.get("title", "untitled")
    is_best = args.get("is-best", "")

    cmd = [
        "/tools/record_score.sh",
        "--scores", "/autosota_artifacts/paper-2787/sota/scores.jsonl",
        "--iter", str(iter_num),
        "--idea-id", str(idea_id),
        "--title", str(title),
        "--status", status,
        "--primary", str(primary),
        "--metrics", metrics_json,
        "--notes", notes,
    ]
    if is_best:
        cmd += ["--is-best", str(is_best)]

    print(f"Recording: status={status} primary={primary} reward={final_reward}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


if __name__ == "__main__":
    main()
