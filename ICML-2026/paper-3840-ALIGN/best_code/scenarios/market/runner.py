import json
from itertools import product

import numpy as np
from omegaconf import OmegaConf

from scenarios.market.env import ProductChoiceMarketEnv
from scenarios.market.prompt import rulePrompt
from scenarios.market.log_metrics import init_log, logging_metrics_market, close_log

from scenarios.market.agent import (
    BuyerBaselineAgent,
    BuyerGossipAgent,
    SellerGossipAgent,
    SellerBaselineAgent,
)



class ProductChoiceMarketRunner:
    """
    Product-choice Market runner (multi-seller / multi-buyer).

    Key properties:
    - Roles do NOT switch: sellers are always sellers; buyers are always buyers.
    - Each episode runs exactly one interaction for every (seller, buyer) pair.
    - If gossip enabled: ONLY buyers publish messages; BOTH sides can read public log.
    """

    def __init__(self, cfg, client, log_path: str):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path

        self.env = ProductChoiceMarketEnv(cfg)

        self.is_gossip = cfg.experiment.agents.is_gossip
        self.use_equilibrium_knowledge = cfg.experiment.agents.use_equilibrium_knowledge

        self.discount_factor = cfg.experiment.env.discount_factor
        self.horizon = cfg.experiment.env.horizon

        self.num_agents = cfg.experiment.agents.num
        self.num_sellers = self.num_agents // 2
        self.num_buyers = self.num_agents // 2

        # One episode = all seller-buyer pairs play once (finite schedule)
        self.horizon_length = self.num_sellers * self.num_buyers if self.horizon == "finite" else np.inf

        self.sellers, self.buyers = self.init_agents()


        # Build shared rules prompt (numbers, not formulas)
        self.rules = rulePrompt(horizon=self.horizon, is_gossip=self.is_gossip).substitute(
            discount_factor=self.discount_factor,
            horizon_length=self.horizon_length,
            # payoff matrix numbers (from env)
            seller_Hc_reward=self.env.payoff_matrix[("H", "c")][0],
            buyer_Hc_reward=self.env.payoff_matrix[("H", "c")][1],
            seller_Hs_reward=self.env.payoff_matrix[("H", "s")][0],
            buyer_Hs_reward=self.env.payoff_matrix[("H", "s")][1],
            seller_Lc_reward=self.env.payoff_matrix[("L", "c")][0],
            buyer_Lc_reward=self.env.payoff_matrix[("L", "c")][1],
            seller_Ls_reward=self.env.payoff_matrix[("L", "s")][0],
            buyer_Ls_reward=self.env.payoff_matrix[("L", "s")][1],
        ).strip()

    # ---------------------------
    # init
    # ---------------------------
    def init_agents(self):
        if self.is_gossip:
            seller_cls = SellerGossipAgent
            buyer_cls = BuyerGossipAgent
        else:
            seller_cls = SellerBaselineAgent
            buyer_cls = BuyerBaselineAgent

        # first half if sellers, second half is buyers
        sellers = [
            seller_cls(
                client=self.client,
                agent_id=f"agent_{i}",
                cfg=self.cfg,
                log_path=self.log_path,
                horizon_length=self.horizon_length,
                env=self.env,
            )
            for i in range(self.num_agents // 2)
        ]
        buyers = [
            buyer_cls(
                client=self.client,
                agent_id=f"agent_{i}",
                cfg=self.cfg,
                log_path=self.log_path,
                horizon_length=self.horizon_length,
                env=self.env,
            )
            for i in range(self.num_agents // 2, self.num_agents)
        ]
        return sellers, buyers

    def all_pairs_schedule(self, shuffle: bool = False):
        pairs = list(product(self.sellers, self.buyers))  # all (seller, buyer)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(pairs)
        return pairs

    # ---------------------------
    # run
    # ---------------------------
    def run_simulation(self, is_test: bool):
        run = init_log(self.cfg, is_test)

        scenario_data = {"config": OmegaConf.to_container(self.cfg, resolve=True)}
        episode_logs = {}
        episode_data = {}
        episode_round_infos = []

        # Public log (buyers write; everyone can read)
        historical_messages = []

        # Reset env + agent episode buffers
        self.env.reset(self.sellers, self.buyers)

        schedule = self.all_pairs_schedule(shuffle=True)

        for round_index, (seller, buyer) in enumerate(schedule, start=1):
            # ---- seller chooses quality ----
            if self.is_gossip:
                seller_justification, seller_action = seller.sell(
                    rules=self.rules,
                    buyer=buyer,
                    historical_messages=historical_messages,  # seller can read public log
                )
            else:
                seller_justification, seller_action = seller.sell(
                    rules=self.rules,
                    buyer=buyer,
                )
            assert seller_action in ("H", "L"), f"Invalid seller_action: {seller_action}"

            # ---- buyer chooses purchase/refuse ----
            if self.is_gossip:
                buyer_justification, buyer_action = buyer.buy(
                    rules=self.rules,
                    seller=seller,
                    historical_messages=historical_messages,  # buyer can read public log
                )
            else:
                buyer_justification, buyer_action = buyer.buy(
                    rules=self.rules,
                    seller=seller,
                )
            assert buyer_action in ("c", "s", "none"), f"Invalid buyer_action: {buyer_action}"

            # ---- env payoff ----
            seller_reward, buyer_reward = self.env.step(
                seller_action=seller_action,
                buyer_action=buyer_action,
            )

            # Store rewards only (no resources tracked)
            seller.rewards.append(float(seller_reward))
            buyer.rewards.append(float(buyer_reward))

            # Optional action logs
            if hasattr(seller, "actions"):
                seller.actions.append({"round": round_index, "seller_action": seller_action})
            if hasattr(buyer, "actions"):
                buyer.actions.append({"round": round_index, "buyer_action": buyer_action})

            # ---- buyer gossip (ONLY buyer publishes) ----
            gossip_pack = None
            if self.is_gossip:
                g_just, g_tone, g_msg = buyer.gossip(
                    rules=self.rules,
                    seller=seller,
                    seller_action=seller_action,
                    buyer_action=buyer_action,
                    seller_reward=seller_reward,
                    buyer_reward=buyer_reward,
                    historical_messages=historical_messages,
                )
                gossip_pack = {
                    "buyer_gossip_justification": g_just,
                    "tone": g_tone,
                    "gossip": g_msg,
                }
                
                historical_messages.append(
                    {
                        "round": round_index,
                        "seller": seller.name,
                        "buyer": buyer.name,
                        "tone": g_tone,
                        "message": g_msg,
                    }
                )

            # ---- round log ----
            round_info = {
                "round": round_index,
                "seller_name": seller.name,
                "buyer_name": buyer.name,
                "seller_action": seller_action,
                "buyer_action": buyer_action,
                "seller_reward": float(seller_reward),
                "buyer_reward": float(buyer_reward),
                "seller_justification": seller_justification,
                "buyer_justification": buyer_justification,
            }
            if gossip_pack is not None:
                round_info.update(gossip_pack)

            episode_data[f"round_{round_index}"] = round_info
            episode_round_infos.append(round_info)

            # Update STM if you keep it
            if hasattr(seller, "update_stm"):
                seller.update_stm(round_index, round_info)
            if hasattr(buyer, "update_stm"):
                buyer.update_stm(round_index, round_info)

            print(f"Round {round_index}/{len(schedule)}")
            print(f"Seller: {seller.name}, action: {seller_action}, reward: {seller_reward}")
            print(f"Buyer: {buyer.name}, action: {buyer_action}, reward: {buyer_reward}")
            if self.is_gossip:
                print(f"Gossip from {buyer.name} to public: {g_msg} (tone: {g_tone})")


        # ---- metrics ----
        logging_metrics_market(
            sellers=self.sellers,
            buyers=self.buyers,
            round_infos=episode_round_infos,
            discount_factor=self.discount_factor,
        )

        episode_logs["interaction"] = episode_data
        scenario_data["episode_1"] = episode_logs

        close_log(run)

        with open(self.log_path, "w") as f:
            json.dump(scenario_data, f, indent=4)

        print(f"Simulation completed. Logs saved to {self.log_path}")