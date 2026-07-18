import numpy as np
import wandb
from omegaconf import OmegaConf

# do this before training loop starts 
def init_log(cfg, is_test): 
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name = cfg.llm.model.replace("/", "_")  # Replace slashes to avoid issues with wandb
    game_name = cfg.experiment.env.game_name
    discount_factor = cfg.experiment.env.discount_factor
    if is_test:
        project_name = f"TEST_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    # elif cfg.experiment.agents.insert_greedy_agent:
    elif cfg.experiment.agents.insert_greedy_agent:
        project_name = f"{game_name}_WithGreedy_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    else:
        project_name = f"{game_name}_Concur_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    group_name = f"model_{model_name}_gossip_{cfg.experiment.agents.is_gossip}_discount_factor_{discount_factor}_eqknow_{cfg.experiment.agents.use_equilibrium_knowledge}"
    trial_name = cfg.metadata.trial_timestamp
    run = wandb.init(
        entity="ALIGN",
        project=project_name,
        group=group_name,
        name=trial_name,
        dir=cfg.metadata.save_dir,
        config=cfg_dict, # Track hyperparameters and run metadata.
        )
    return run

def get_std_err(data): 
    std_dev = np.std(data)  # standard deviation (default ddof=0)
    std_err = std_dev / np.sqrt(len(data))
    return std_err

def compute_dis_cum_reward(rewards, discount_factor):
    """
    Calculate the discounted cumulative reward for an agent.
    """
    discounted_cumulative_reward = 0
    for idx, reward in enumerate(rewards):
        discounted_cumulative_reward += (discount_factor ** idx) * reward
    return discounted_cumulative_reward

def compute_gini_coefficient(x):
    """Compute the Gini coefficient of discounted cumulative rewards between agents."""
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    gini = 0.5 * rmad
    return gini

def logging_metrics(agents, discount_factor):
    all_total_investments = []
    all_avg_investment_ratios = []
    all_total_returned_amounts = []
    all_avg_returned_ratios = []
    all_avg_rewards = []
    all_discounted_cumulative_rewards = []
    for agent_idx, agent in enumerate(agents):
        total_investment = np.sum(agent.investments)
        avg_investment_ratio = np.mean(agent.investment_ratios) if len(agent.investment_ratios) > 0 else 0
        total_returned_amount = np.sum(agent.returned_amounts)
        avg_returned_ratio = np.mean(agent.returned_ratios) if len(agent.returned_ratios) > 0 else 0
        avg_reward = np.mean(agent.rewards) if len(agent.rewards) > 0 else 0
        discounted_cumulative_reward = compute_dis_cum_reward(agent.rewards, discount_factor)

        all_total_investments.append(total_investment)
        all_avg_investment_ratios.append(avg_investment_ratio)
        all_total_returned_amounts.append(total_returned_amount)
        all_avg_returned_ratios.append(avg_returned_ratio)
        all_avg_rewards.append(avg_reward)
        all_discounted_cumulative_rewards.append(discounted_cumulative_reward)
        wandb.log({
            f"Agent {agent_idx} Total Investment": total_investment,
            f"Agent {agent_idx} Avg Investment Ratio": avg_investment_ratio,
            f"Agent {agent_idx} Total Returned Amount": total_returned_amount,
            f"Agent {agent_idx} Avg Returned Ratio": avg_returned_ratio,
            f"Agent {agent_idx} Avg Reward": avg_reward,
            f"Agent {agent_idx} Discounted Cumulative Reward": discounted_cumulative_reward
        })
    wandb.log({"Average Total Investment": np.mean(all_total_investments)})
    wandb.log({"Average Avg Investment Ratio": np.mean(all_avg_investment_ratios)})
    wandb.log({"Average Total Returned Amount": np.mean(all_total_returned_amounts)})
    wandb.log({"Average Avg Returned Ratio": np.mean(all_avg_returned_ratios)})
    wandb.log({"Average Avg Reward": np.mean(all_avg_rewards)})
    wandb.log({"Average Discounted Cumulative Reward": np.mean(all_discounted_cumulative_rewards)})

    gini_coefficient = compute_gini_coefficient(np.array(all_discounted_cumulative_rewards))
    wandb.log({"Gini Coefficient of Discounted Cumulative Rewards": gini_coefficient})


# must be closed once all episodes are done 
def close_log(run): 
    run.finish()