#!/usr/bin/env python3
"""Training experiment wrapper for paper 5267 SOTA optimization."""
import subprocess, sys, json, os, re
from pathlib import Path
from datetime import datetime

def run(cmd, timeout_min=150):
    print(f"[{datetime.now()}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/repo",
                          timeout=timeout_min * 60)
    stdout = result.stdout
    stderr = result.stderr
    if stderr:
        print("STDERR:", stderr[:2000])
    
    metrics = {}
    for line in stdout.split("\n"):
        nums = re.findall(r"0\.\d{3,4}", line)
        if len(nums) >= 4:
            metrics["gsm8k_accuracy"] = float(nums[0])
            metrics["aqua_accuracy"] = float(nums[1])
            metrics["mawps_accuracy"] = float(nums[2])
            metrics["svamp_accuracy"] = float(nums[3])
            if len(nums) >= 5:
                metrics["math10k_average"] = float(nums[4])
            break
    
    summary_csv = list(Path("/repo/LLM-Adapters/experiment").rglob("summary.csv"))
    if summary_csv:
        import pandas as pd
        df = pd.read_csv(summary_csv[-1])
        for col in df.columns:
            if col.lower() in ["gsm8k", "aqua", "mawps", "svamp"]:
                metrics[f"{col.lower()}_accuracy"] = float(df[col].iloc[0])
            elif col.lower() == "average":
                metrics["math10k_average"] = float(df[col].iloc[0])
    
    train_logs = list(Path("/repo/LLM-Adapters/trained_models").rglob("train_log.jsonl"))
    if train_logs:
        eps_vals = []
        with open(train_logs[-1]) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "eps_spent" in rec:
                        eps_vals.append(float(rec["eps_spent"]))
                except Exception:
                    pass
        if eps_vals:
            metrics["epsilon_spent"] = eps_vals[-1]
    
    return {
        "success": result.returncode == 0 and len(metrics) >= 4,
        "metrics": metrics,
        "stdout_tail": stdout[-2000:] if stdout else "",
    }

if __name__ == "__main__":
    cmd = sys.argv[1:]
    if not cmd:
        print("Usage: python3 training_experiment.py <command args...>")
        sys.exit(1)
    result = run(cmd)
    print(json.dumps(result, indent=2))
