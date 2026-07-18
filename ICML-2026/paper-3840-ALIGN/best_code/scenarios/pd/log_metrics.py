import numpy as np
import wandb
from omegaconf import OmegaConf

# do this before training loop starts 
def init_log(cfg, is_test): 
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name = cfg.llm.model.replace("/", "_")  # Replace slashes to avoid issues with wandb
    if is_test:
        project_name = f"TEST_{cfg.experiment.env.game_name}_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    elif cfg.experiment.agents.insert_greedy_agent:
        project_name = f"ROBUSSTNESS_{cfg.experiment.env.game_name}_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    else:
        project_name = f"{cfg.experiment.env.game_name}_N{cfg.experiment.agents.num}_horizon_{cfg.experiment.env.horizon}"
    group_name = f"model_{model_name}_gossip_{cfg.experiment.agents.is_gossip}_eqknow_{cfg.experiment.agents.use_equilibrium_knowledge}"
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

def logging_metrics(all_agent_cooperation_ratio, all_agent_return, all_agent_discounted_return, all_agent_image_score): 
    #Average Metrics
    avg_cooperation_ratio = np.mean(all_agent_cooperation_ratio)
    avg_return = np.mean(all_agent_return) #avg return across all agents for 1 episode
    avg_discount_return = np.mean(all_agent_discounted_return) #avg discounted return across all
    avg_image_score = np.mean(all_agent_image_score)

    # Standard Error Metrics
    std_err_cooperation = get_std_err(all_agent_cooperation_ratio)
    std_err_returns = get_std_err(all_agent_return)
    std_err_discounted_returns = get_std_err(all_agent_discounted_return)
    std_err_image_score = get_std_err(all_agent_image_score)

    wandb.log({"Average Cooperation Ratio": avg_cooperation_ratio, 
               "Average Return": avg_return, "Average Discounted Return": avg_discount_return, 
               "Average Image Score": avg_image_score,
               "Std Error for Cooperation Ratios": std_err_cooperation,
               "Std Error for Returns": std_err_returns, "Std Error for Discounted Returns": std_err_discounted_returns, 
               "Std Error for Image Score": std_err_image_score})
    for agent_idx in range(len(all_agent_cooperation_ratio)):
        wandb.log({f"Agent {agent_idx} Avg Cooperation Ratio": all_agent_cooperation_ratio[agent_idx], f"Agent {agent_idx} Return": all_agent_return[agent_idx], f"Agent {agent_idx} Discounted Return": all_agent_discounted_return[agent_idx], f"Agent {agent_idx} Image Score": all_agent_image_score[agent_idx]})

# must be closed once all episodes are done 
def close_log(run): 
    run.finish()