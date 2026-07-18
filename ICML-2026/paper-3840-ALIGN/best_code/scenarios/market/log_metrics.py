import numpy as np
import wandb
from omegaconf import OmegaConf


def init_log(cfg, is_test):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_name = cfg.llm.model.replace("/", "_") # Replace slashes to avoid issues with wandb
    game_name = cfg.experiment.env.game_name
    discount_factor = cfg.experiment.env.discount_factor

    if is_test:
        project_name = f"TEST_{game_name}_horizon_{cfg.experiment.env.horizon}"
    else:
        project_name = f"{game_name}_horizon_{cfg.experiment.env.horizon}"

    group_name = (
        f"model_{model_name}_gossip_{cfg.experiment.agents.is_gossip}"
    )
    trial_name = cfg.metadata.trial_timestamp

    run = wandb.init(
        entity="ALIGN",
        project=project_name,
        group=group_name,
        name=trial_name,
        dir=cfg.metadata.save_dir,
        config=cfg_dict,
    )
    return run


def close_log(run):
    run.finish()


def get_std_err(data):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return 0.0
    std_dev = np.std(data)  # ddof=0
    std_err = std_dev / np.sqrt(len(data))
    return float(std_err)


def compute_avg_reward(agent):
    rewards = np.asarray(getattr(agent, "rewards", []), dtype=float)
    return float(np.mean(rewards)) if rewards.size else 0.0


def compute_dis_cum_reward(agent, discount_factor):
    rewards = np.asarray(getattr(agent, "rewards", []), dtype=float)
    if rewards.size == 0:
        return 0.0
    gamma = float(discount_factor)
    disc = 0.0
    g = 1.0
    for r in rewards:
        disc += g * float(r)
        g *= gamma
    return float(disc)

def compute_deal_rate(round_infos):
    if len(round_infos) == 0:
        return 0.0
    deals = sum(1 for ri in round_infos if ri.get("buyer_action") != "none")
    return float(deals / len(round_infos))


def compute_pair_proportions(round_infos, conditional_on_deal=False):
    keys = ["Hc", "Hs", "Lc", "Ls"]
    counts = {k: 0 for k in keys}
    denom = 0

    for ri in round_infos:
        sa = ri.get("seller_action")
        ba = ri.get("buyer_action")

        if conditional_on_deal:
            if ba == "none":
                continue
            denom += 1
        else:
            denom += 1

        if ba in ("c", "s") and sa in ("H", "L"):
            counts[f"{sa}{ba}"] += 1

    if denom == 0:
        return {k: 0.0 for k in keys}
    return {k: float(v / denom) for k, v in counts.items()}


def compute_avg_episode_reward_seller(round_infos):
    if len(round_infos) == 0:
        return 0.0
    return float(np.mean([float(ri.get("seller_reward", 0.0)) for ri in round_infos]))


def compute_avg_episode_reward_buyer(round_infos):
    if len(round_infos) == 0:
        return 0.0
    return float(np.mean([float(ri.get("buyer_reward", 0.0)) for ri in round_infos]))


def compute_avg_episode_reward_all(round_infos):
    if len(round_infos) == 0:
        return 0.0
    welfare = [float(ri.get("seller_reward", 0.0)) + float(ri.get("buyer_reward", 0.0)) for ri in round_infos]
    # per-agent-per-round average
    return float(np.mean(welfare) / 2.0)


def compute_welfare_per_round(round_infos):
    if len(round_infos) == 0:
        return 0.0
    welfare = [float(ri.get("seller_reward", 0.0)) + float(ri.get("buyer_reward", 0.0)) for ri in round_infos]
    return float(np.mean(welfare))


def compute_discounted_welfare(round_infos, discount_factor):
    if len(round_infos) == 0:
        return 0.0
    gamma = float(discount_factor)
    disc = 0.0
    g = 1.0
    for ri in round_infos:
        disc += g * (float(ri.get("seller_reward", 0.0)) + float(ri.get("buyer_reward", 0.0)))
        g *= gamma
    return float(disc)


# -----------------------------
# Main logging function (episode)
# -----------------------------
def logging_metrics_market(
    sellers,
    buyers,
    round_infos: list,
    discount_factor: float,
):
    """
    Call once per episode.
    - sellers, buyers: lists of agent objects
    - round_infos: list of per-round logs for the episode
    """

    # ---- episode-level outcome metrics ----
    deal_rate = compute_deal_rate(round_infos)
    pair_props = compute_pair_proportions(round_infos, conditional_on_deal=False)
    pair_props_deal = compute_pair_proportions(round_infos, conditional_on_deal=True)

    avg_ep_seller_reward = compute_avg_episode_reward_seller(round_infos)
    avg_ep_buyer_reward = compute_avg_episode_reward_buyer(round_infos)
    avg_ep_all_reward = compute_avg_episode_reward_all(round_infos)

    welfare_per_round = compute_welfare_per_round(round_infos)
    discounted_welfare = compute_discounted_welfare(round_infos, discount_factor)

    # ---- per-agent metrics ----
    seller_avg_rewards = [compute_avg_reward(a) for a in sellers]
    buyer_avg_rewards = [compute_avg_reward(a) for a in buyers]

    seller_disc_returns = [compute_dis_cum_reward(a, discount_factor) for a in sellers]
    buyer_disc_returns = [compute_dis_cum_reward(a, discount_factor) for a in buyers]

    all_avg_rewards = seller_avg_rewards + buyer_avg_rewards
    all_disc_returns = seller_disc_returns + buyer_disc_returns

    # requested aggregates
    avg_reward_seller = float(np.mean(seller_avg_rewards)) if len(seller_avg_rewards) else 0.0
    avg_reward_buyer = float(np.mean(buyer_avg_rewards)) if len(buyer_avg_rewards) else 0.0
    avg_reward_all_agents = float(np.mean(all_avg_rewards)) if len(all_avg_rewards) else 0.0

    avg_disc_return_seller = float(np.mean(seller_disc_returns)) if len(seller_disc_returns) else 0.0
    avg_disc_return_buyer = float(np.mean(buyer_disc_returns)) if len(buyer_disc_returns) else 0.0
    avg_disc_return_all_agents = float(np.mean(all_disc_returns)) if len(all_disc_returns) else 0.0

    # standard errors (helpful in plots)
    se_avg_reward_seller = get_std_err(seller_avg_rewards)
    se_avg_reward_buyer = get_std_err(buyer_avg_rewards)
    se_avg_reward_all = get_std_err(all_avg_rewards)

    se_disc_return_seller = get_std_err(seller_disc_returns)
    se_disc_return_buyer = get_std_err(buyer_disc_returns)
    se_disc_return_all = get_std_err(all_disc_returns)

    # ---- wandb.log: episode-level ----
    wandb.log({
        # deals + outcome distribution
        "Deal Rate": deal_rate,
        "Pair Prop (Hc)": pair_props["Hc"],
        "Pair Prop (Hs)": pair_props["Hs"],
        "Pair Prop (Lc)": pair_props["Lc"],
        "Pair Prop (Ls)": pair_props["Ls"],
        "Pair Prop | Deal (Hc)": pair_props_deal["Hc"],
        "Pair Prop | Deal (Hs)": pair_props_deal["Hs"],
        "Pair Prop | Deal (Lc)": pair_props_deal["Lc"],
        "Pair Prop | Deal (Ls)": pair_props_deal["Ls"],

        # episode reward summaries (from rounds)
        "Avg Episode Seller Reward": avg_ep_seller_reward,
        "Avg Episode Buyer Reward": avg_ep_buyer_reward,
        "Avg Episode All Reward": avg_ep_all_reward,
        "Welfare Per Round": welfare_per_round,
        "Discounted Welfare": discounted_welfare,

        # requested per-agent aggregates
        "Avg Reward Seller (agents)": avg_reward_seller,
        "Avg Reward Buyer (agents)": avg_reward_buyer,
        "Avg Reward All Agents": avg_reward_all_agents,
        "Avg Discounted Return Seller": avg_disc_return_seller,
        "Avg Discounted Return Buyer": avg_disc_return_buyer,
        "Avg Discounted Return All Agents": avg_disc_return_all_agents,

        # standard errors
        "StdErr Avg Reward Seller": se_avg_reward_seller,
        "StdErr Avg Reward Buyer": se_avg_reward_buyer,
        "StdErr Avg Reward All": se_avg_reward_all,
        "StdErr Disc Return Seller": se_disc_return_seller,
        "StdErr Disc Return Buyer": se_disc_return_buyer,
        "StdErr Disc Return All": se_disc_return_all,
    })

    # ---- wandb.log: per-agent (like donor game) ----
    for idx, a in enumerate(sellers):
        wandb.log({
            f"Seller {idx} Avg Reward": seller_avg_rewards[idx],
            f"Seller {idx} Discounted Return": seller_disc_returns[idx],
        })
    for idx, a in enumerate(buyers):
        wandb.log({
            f"Buyer {idx} Avg Reward": buyer_avg_rewards[idx],
            f"Buyer {idx} Discounted Return": buyer_disc_returns[idx],
        })