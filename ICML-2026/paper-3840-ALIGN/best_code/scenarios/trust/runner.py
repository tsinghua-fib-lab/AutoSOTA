from scenarios.trust.agent import BaselineAgent, GossipAgent
from scenarios.trust.env import TrustGameEnv
from scenarios.trust.prompt import rulePrompt
from scenarios.trust.log_metrics import *
import numpy as np
from itertools import combinations
import json

class TrustGameRunner:
    def __init__(self, cfg, client, log_path):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path
        self.env = TrustGameEnv(cfg)
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.use_equilibrium_knowledge = cfg.experiment.agents.use_equilibrium_knowledge
        self.discount_factor = cfg.experiment.env.discount_factor
        self.investment_multiplier = cfg.experiment.env.investment_multiplier
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = cfg.experiment.env.horizon_length if self.horizon == "finite" else np.inf
        self.agents = self.init_agents()
        self.rules = rulePrompt(horizon=self.cfg.experiment.env.horizon, is_gossip=self.is_gossip).substitute(initial_resources=cfg.experiment.env.initial_resources, investment_multiplier=cfg.experiment.env.investment_multiplier, discount_factor=self.discount_factor, horizon_length=self.horizon_length).strip()

    def init_agents(self):
        if self.is_gossip:
            agent_class = GossipAgent
        else:
            agent_class = BaselineAgent
        agents = [agent_class(client=self.client, agent_id=f"agent_{i}", cfg=self.cfg, log_path=self.log_path, horizon_length=self.horizon_length)
            for i in range(self.cfg.experiment.agents.num)
        ]
        return agents
    
    def round_robin_donor_game(self, agents):
        """
        Return an ordered list of (donor, recipient) pairs such that

        1. Each unordered pair of agents appears exactly once.
        2. For any given agent, successive appearances alternate
        between donor (D) and recipient (R).
        3. Each agent is donor ⌊(n-1)/2⌋ or ⌈(n-1)/2⌉ times.

        One pair plays per round —­ the list order *is* the round order.
        Raises RuntimeError if no schedule exists (rare for n ≤ 10).
        """
        n = len(agents)
        plays_per_agent  = n - 1
        donor_low, donor_high = plays_per_agent // 2, (plays_per_agent + 1) // 2

        # --- all unordered pairs produced in one line with *combinations* ---
        remaining_pairs = list(combinations(agents, 2))      # ← ★ here ★

        # bookkeeping
        donor_cnt, recip_cnt = {a: 0 for a in agents}, {a: 0 for a in agents}
        last_role           = {a: None for a in agents}
        used_pairs, schedule = set(), []

        # depth-first search with back-tracking — minimalist but guaranteed
        def dfs():
            if len(schedule) == len(remaining_pairs):
                return True

            for a, b in remaining_pairs:
                if (a, b) in used_pairs:
                    continue

                for donor, recip in ((a, b), (b, a)):      # orient the pair
                    # quotas
                    if donor_cnt[donor]   >= donor_high or recip_cnt[recip] >= donor_high:
                        continue
                    # role alternation
                    if last_role[donor] == 'D' or last_role[recip] == 'R':
                        continue

                    # commit
                    used_pairs.add((a, b))
                    schedule.append((donor, recip))
                    donor_cnt[donor]   += 1
                    recip_cnt[recip]   += 1
                    prev_d, prev_r      = last_role[donor], last_role[recip]
                    last_role[donor]    = 'D'
                    last_role[recip]    = 'R'

                    if dfs():
                        return True

                    # back-track
                    used_pairs.remove((a, b))
                    schedule.pop()
                    donor_cnt[donor]   -= 1
                    recip_cnt[recip]   -= 1
                    last_role[donor]    = prev_d
                    last_role[recip]    = prev_r
            return False

        if not dfs():
            raise RuntimeError("No valid schedule under the requested constraints.")
        return schedule

    def run_simulation(self, is_test):
        """
        run simulation
        """
        run = init_log(self.cfg, is_test)
        # Only one episode for trust game
        episode_data = {}
        episode_data["config"] = OmegaConf.to_container(self.cfg, resolve=True)
        historical_messages = []
        self.env.reset(self.agents)
        all_pairs_schedule = self.round_robin_donor_game(self.agents)
        for round_index, pair in enumerate(all_pairs_schedule):
            print(f"Round {round_index + 1}")
            investor, responder = pair
            resources_before_investment = {"investor": investor.resources, "responder": responder.resources}

            if self.is_gossip:
                investor_justification, investment = investor.invest(self.rules, responder, historical_messages)
            else:
                investor_justification, investment = investor.invest(self.rules, responder)
            # transfer investment from str to float if needed
            if isinstance(investment, str):
                investment = float(investment)
            assert investment <= investor.resources, "Investment amount is invalid."
            investment_ratio = investment/investor.resources if investor.resources > 0 else 0
            benefit = investment * self.cfg.experiment.env.investment_multiplier

            if self.is_gossip:
                # responder choose returns
                responder_justification, returned_amount = responder.respond(self.rules, investor, investment, investment_ratio, benefit, historical_messages)
            else:
                responder_justification, returned_amount = responder.respond(self.rules, investor, investment, investment_ratio, benefit)
            if isinstance(returned_amount, str):
                returned_amount = float(returned_amount)
            assert returned_amount <= benefit, "Returned amount is invalid."
            returned_ratio = returned_amount / (investment * self.investment_multiplier) if investment > 0 else 0
            
            print(f"Investor: {investor.name}, Investment: {investment},\n Justification: {investor_justification}")
            print(f"Responder: {responder.name}, Returned Amount: {returned_amount},\n Justification: {responder_justification}")
            # Gossip Phase
            if self.is_gossip:
                # investor gossip after observing investment and returned amount
                investor_gossip_justification, investor_tone, investor_message = investor.investor_gossip(self.rules, responder, investment, investment_ratio, benefit, returned_amount, returned_ratio, historical_messages)
                # responder gossip after observing investment and returned amount
                responder_gossip_justification, responder_tone, responder_message = responder.responder_gossip(self.rules, investor, investment, investment_ratio, benefit, returned_amount, returned_ratio, historical_messages)
                print(f"Investor: {investor.name}, Selected Tone: {investor_tone}, Gossip: {investor_message},\n Investor's Justification: {investor_gossip_justification}\n")
                print(f"Responder: {responder.name}, Selected Tone: {responder_tone}, Gossip: {responder_message},\n Responder's Justification: {responder_gossip_justification}\n")
                message_summary_investor = {"round": {round_index+1}, "investor": investor.name, "responder": responder.name, f"message from {investor.name}": investor_message}
                message_summary_responder = {"round": {round_index+1}, "investor": investor.name, "responder": responder.name, f"message from {responder.name}": responder_message}
                historical_messages.append(message_summary_investor)
                historical_messages.append(message_summary_responder)
                cur_round_info = {"investor_name": investor.name, "responder_name": responder.name, "resources_before_investment": resources_before_investment, "investment": investment, "investment_ratio": investment_ratio, "investor_justification": investor_justification, "returned_amount": returned_amount, "returned_ratio": returned_ratio, "responder_justification": responder_justification, "investor_tone":investor_tone, "investor_gossip": investor_message, "investor_gossip_justification": investor_gossip_justification, "responder_tone":responder_tone, "responder_gossip": responder_message, "responder_gossip_justification": responder_gossip_justification} # Update the trajectory(STM) of the players with this current round info 
            else:
                cur_round_info = {"investor_name": investor.name, "responder_name": responder.name, "resources_before_investment": resources_before_investment, "investment": investment, "investment_ratio": investment_ratio, "investor_justification": investor_justification, "returned_amount": returned_amount, "returned_ratio": returned_ratio, "responder_justification": responder_justification}
            episode_data[f"round_{round_index+1}"] = cur_round_info # Log data per round
            investor.update_stm(round_index+1, cur_round_info)
            responder.update_stm(round_index+1, cur_round_info)
            self.env.step(investor, responder, investment, investment_ratio, returned_amount, returned_ratio)
        
        # Logging metrics at the end of the episode
        logging_metrics(self.agents, self.discount_factor)
        close_log(run)
        with open(f'{self.log_path}', "w") as f:
            json.dump(episode_data, f, indent=4)
        print(f"Simulation completed. Logs saved to {self.log_path}")