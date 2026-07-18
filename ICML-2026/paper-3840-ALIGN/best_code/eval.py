#!/usr/bin/env python3
"""Evaluation script for ALIGN infinite-horizon donation game.
Reproduces: Cooperation Ratio, Image Score, Reward Per Round, Discounted Return, Gini Coefficient.
"""
import subprocess, os, glob, json, numpy as np, sys, time

def run_eval():
    print("=== ALIGN Donor Game Evaluation ===")
    print("Model: deepseek-chat (DeepSeek V3)")
    print("Settings: n_agents=9, infinite horizon, discount=0.99, cost=1, benefit=5, gossip=true")
    print()
    
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["DEEPSEEK_API_KEY"] = env.get("DEEPSEEK_API_KEY", "")
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    
    # Clear old logs
    log_dir = "/repo/logs/donor_horizon_infinite_gossip_True_greedy_False"
    old_logs = glob.glob(f"{log_dir}/*.json")
    for old in old_logs:
        try:
            os.remove(old)
        except:
            pass
    
    result = subprocess.run(
        ["python3", "main.py"],
        cwd="/repo",
        env=env,
        capture_output=True,
        text=True,
        timeout=600
    )
    
    if result.returncode != 0:
        print(f"ERROR: Experiment failed")
        print(result.stderr[-1000:])
        sys.exit(1)
    
    log_files = glob.glob(f"{log_dir}/*.json")
    log_files.sort(key=os.path.getmtime, reverse=True)
    if not log_files:
        print("ERROR: No log file found")
        sys.exit(1)
    
    log_path = log_files[0]
    print(f"Log: {log_path}")
    
    with open(log_path) as f:
        data = json.load(f)
    
    for ep_key, ep_data in data.items():
        if not ep_key.startswith("episode"):
            continue
        interactions = ep_data.get("interaction", {})
        n_rounds = len(interactions)
        
        agents_data = {}
        for r_key, r_data in interactions.items():
            donor = r_data["donor_name"]
            recip = r_data["recipient_name"]
            donation = r_data["donation"]
            benefit = r_data["received_benefit"]
            
            for name in [donor, recip]:
                if name not in agents_data:
                    agents_data[name] = {"donations": [], "rewards": []}
            
            agents_data[donor]["donations"].append(donation)
            agents_data[donor]["rewards"].append(-donation)
            agents_data[recip]["rewards"].append(benefit)
        
        n_agents = len(agents_data)
        rounds_per_agent = n_agents - 1
        
        coop_count = sum(1 for r in interactions.values() if r.get("donation", 0) > 0)
        coop_ratio = coop_count / n_rounds
        
        image_scores = [sum(1 if d > 0 else -1 for d in a["donations"]) for a in agents_data.values()]
        agent_rewards = [sum(a["rewards"]) for a in agents_data.values()]
        reward_per_round = np.mean([r / rounds_per_agent for r in agent_rewards])
        
        discount = 0.99
        disc_returns = [sum((discount ** i) * r for i, r in enumerate(a["rewards"])) for a in agents_data.values()]
        
        G = np.array(disc_returns)
        n = len(G)
        if np.sum(G) > 0:
            gini = np.sum(np.abs(G[:, None] - G[None, :])) / (2 * n * np.sum(G))
        else:
            gini = 0.0
        
        print()
        print(f"Cooperation Ratio: {coop_ratio:.4f}")
        print(f"Image Score: {np.mean(image_scores):.4f}")
        print(f"Reward Per Round: {reward_per_round:.4f}")
        print(f"Discounted Return: {np.mean(disc_returns):.4f}")
        print(f"Gini Coefficient: {gini:.4f}")
        print()
        print("=== Evaluation Complete ===")

if __name__ == "__main__":
    run_eval()
