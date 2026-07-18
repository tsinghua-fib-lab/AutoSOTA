#!/bin/bash
# Evaluation script for ALIGN infinite-horizon donation game
# Reproduces Cooperation Ratio, Image Score, Reward Per Round, Discounted Return, Gini Coefficient

set -e

cd /repo

# Proxy fix: remove SOCKS5 proxy to avoid connection drops
unset ALL_PROXY all_proxy

export WANDB_MODE=disabled
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be set}"

echo "=== ALIGN Donor Game Evaluation ==="
echo "Model: deepseek-chat (DeepSeek V3)"
echo "Settings: n_agents=9, infinite horizon, discount=0.99, cost=1, benefit=5, gossip=true"
echo ""

python3 main.py 2>/dev/null

# Find the latest log file
LATEST_LOG=$(find /repo/logs/donor_horizon_infinite_gossip_True_greedy_False/ -name "*.json" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2)

if [ -z "$LATEST_LOG" ]; then
    echo "ERROR: No log file found"
    exit 1
fi

echo "Log file: $LATEST_LOG"
echo ""

python3 -c "
import json, numpy as np

with open() as f:
    data = json.load(f)

for ep_key, ep_data in data.items():
    if not ep_key.startswith(episode):
        continue
    interactions = ep_data.get(interaction, {})
    n_rounds = len(interactions)
    
    agents_data = {}
    for r_key, r_data in interactions.items():
        donor = r_data[donor_name]
        recip = r_data[recipient_name]
        donation = r_data[donation]
        benefit = r_data[received_benefit]
        
        for name in [donor, recip]:
            if name not in agents_data:
                agents_data[name] = {donations: [], rewards: []}
        
        agents_data[donor][donations].append(donation)
        agents_data[donor][rewards].append(-donation)
        agents_data[recip][rewards].append(benefit)
    
    n_agents = len(agents_data)
    rounds_per_agent = n_agents - 1
    
    coop_count = sum(1 for r in interactions.values() if r.get(donation, 0) > 0)
    coop_ratio = coop_count / n_rounds
    
    image_scores = [sum(1 if d > 0 else -1 for d in a[donations]) for a in agents_data.values()]
    agent_rewards = [sum(a[rewards]) for a in agents_data.values()]
    reward_per_round = np.mean([r / rounds_per_agent for r in agent_rewards])
    
    discount = 0.99
    disc_returns = [sum((discount ** i) * r for i, r in enumerate(a[rewards])) for a in agents_data.values()]
    
    G = np.array(disc_returns)
    n = len(G)
    gini = np.sum(np.abs(G[:, None] - G[None, :])) / (2 * n * np.sum(G)) if np.sum(G) > 0 else 0.0
    
    print(Cooperation
