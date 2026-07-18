from scenarios.donor.agent import BaselineAgent, GossipAgent, GreedyAgent
from scenarios.donor.env import DonorGameEnv
from scenarios.donor.prompt import rulePrompt
from scenarios.donor.utility import *
from scenarios.donor.log_metrics import *
import numpy as np
from itertools import combinations
import json

class DonorGameRunner:
    def __init__(self, cfg, client, log_path):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path
        self.env = DonorGameEnv(cfg)
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.discount_factor = cfg.experiment.env.discount_factor
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = self.cfg.experiment.agents.num * (self.cfg.experiment.agents.num - 1) / 2 if self.horizon == "finite" else np.inf
        self.agents = self.init_agents()
        self.rules = rulePrompt(horizon=self.cfg.experiment.env.horizon, is_gossip=self.is_gossip).substitute(initial_resources=cfg.experiment.env.initial_resources, cooperationGain=cfg.experiment.env.cooperationGain, termination_prob=cfg.experiment.env.termination_prob, discount_factor=self.discount_factor, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, horizon_length=self.horizon_length).strip()

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
        scenario_data = {}
        scenario_data["config"] = OmegaConf.to_container(self.cfg, resolve=True)
        for episode in range(self.cfg.experiment.env.num_episodes):
            episode_logs = {}
            episode_data = {}
            historical_messages = []
            self.env.reset(self.agents)
            all_pairs_schedule = self.round_robin_donor_game(self.agents)
            resources_start = [agent.resources for agent in self.agents]

            for round_index, pair in enumerate(all_pairs_schedule):
                import time as _time
                if round_index > 0:
                    _time.sleep(1.0)  # Rate limiting delay between rounds
                print(f"Round {round_index + 1}")
                donor, recipient = pair
                # for donor, recipient in round_pairings:
                resources_before_donation = {"donor": donor.resources, "recipient": recipient.resources}

                if self.is_gossip:
                    donor_justification, donor_action = donor.donate(self.rules, recipient, historical_messages)
                else:
                    donor_justification, donor_action = donor.donate(self.rules, recipient)

                assert donor_action in ["cooperate", "defect"], "Invalid action taken by donor."
                if donor_action == "defect":
                    donation = 0
                    received_benefit = 0
                else: # cooperate
                    donation = self.cfg.experiment.env.cost
                    received_benefit = self.cfg.experiment.env.benefit
                print(f"Donor: {donor.name}, Recipient: {recipient.name}, Action : {donor_action}, Donation: {donation} \n")
            
                print(f"Donor's Justification: {donor_justification} \n")
                
                assert donation <= donor.resources, "Donation amount is invalid."
                donation_ratio = compute_donation_ratio(donation, donor.resources)

                if self.is_gossip:
                    recipient_justification, recipient_tone, recipient_message = recipient.gossip(self.rules, donor, donation, donation_ratio, received_benefit, historical_messages)
                    print(f"Recipient: {recipient.name}, Selected Tone: {recipient_tone}, Gossip: {recipient_message},\n Recipient's Justification: {recipient_justification}\n")
                    message_summary = {"round": {round_index+1}, "donor": donor.name, "recipient": recipient.name, f"message from {recipient.name}": recipient_message}
                    historical_messages.append(message_summary)
                    cur_round_info = {"donor_name": donor.name, "recipient_name": recipient.name, "resources_before_donation": resources_before_donation, "donation": donation, "donation_ratio": donation_ratio, "donor_justification": donor_justification, "received_benefit": received_benefit, "recipient_justification": recipient_justification, "tone":recipient_tone, "gossip": recipient_message} # Update the trajectory(STM) of the players with this current round info 
                else:
                    cur_round_info = {"donor_name": donor.name, "recipient_name": recipient.name, "resources_before_donation": resources_before_donation, "donation": donation, "donation_ratio": donation_ratio, "donor_justification": donor_justification, "received_benefit": received_benefit}
                episode_data[f"round_{round_index+1}"] = cur_round_info # Log data per round
                donor.update_stm(round_index+1, cur_round_info)
                recipient.update_stm(round_index+1, cur_round_info)
                self.env.step(donor, recipient, donation, received_benefit)
                # Update reward signals and donations made
                donor.donations.append(donation)
                donor.donation_ratios.append(donation_ratio)
                recipient.benefits.append(received_benefit)
                donor.rewards.append(-donation)
                recipient.rewards.append(received_benefit)

            for k, agent in enumerate(self.agents):
                donations = agent.donations
                donation_ratios = agent.donation_ratios
                rewards = agent.rewards
                benefits = agent.benefits
                assert len(donations) == len(donation_ratios) == len(benefits) == len(rewards)/2, "Mismatch in lengths of donations, donation ratios, rewards, and benefits."
                for step in range(len(donations)):
                    wandb.log({f"Agent {k} Donation Per Step": donations[step], f"Agent {k} Donation Ratio Per Step": donation_ratios[step], f"Agent {k} Reward Per Step": rewards[step], f"Agent {k} Benefit Per Step": benefits[step]})

            # Compute metrics
            avg_donation_all = [compute_avg_donation(agent) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            avg_donation_ratios_all = [compute_avg_donation_ratio(agent) for agent in self.agents]
            returns_all = [compute_return(agent, resources_start[agent_idx]) for agent_idx, agent in enumerate(self.agents)]
            discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, self.discount_factor) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            image_score_all = [compute_image_score(agent) for agent in self.agents]

            # Log data per episode
            episode_logs["interaction"] = episode_data
            scenario_data[f"episode_{episode+1}"] = episode_logs
            # Log metrics per episode    
            logging_metrics(avg_donation_all, avg_donation_ratios_all, returns_all, discounted_cumulative_rewards_all, image_score_all)
        close_log(run)
        # print(scenario_data)
        with open(f'{self.log_path}', "w") as f:
            json.dump(scenario_data, f, indent=4)
        print(f"Simulation completed. Logs saved to {self.log_path}")


class DonorGameRunnerWithGreedyAgent:
    def __init__(self, cfg, client, log_path):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path
        self.env = DonorGameEnv(cfg)
        self.insert_greedy_agent = cfg.experiment.agents.get("insert_greedy_agent", False)
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.discount_factor = cfg.experiment.env.discount_factor
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = self.cfg.experiment.agents.num * (self.cfg.experiment.agents.num - 1) / 2 if self.horizon == "finite" else np.inf
        self.agents = self.init_agents()
        self.rules = rulePrompt(horizon=self.cfg.experiment.env.horizon, is_gossip=self.is_gossip).substitute(initial_resources=cfg.experiment.env.initial_resources, cooperationGain=cfg.experiment.env.cooperationGain, termination_prob=cfg.experiment.env.termination_prob, discount_factor=self.discount_factor, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, horizon_length=self.horizon_length).strip()

    def init_agents(self):
        if self.is_gossip:
            agent_class = GossipAgent
        else:
            agent_class = BaselineAgent
        
        if self.insert_greedy_agent:
            greedy_agent_class = GreedyAgent
            agents = [agent_class(client=self.client, agent_id=f"agent_{i}", cfg=self.cfg, log_path=self.log_path, horizon_length=self.horizon_length)
                for i in range(self.cfg.experiment.agents.num - 1)
            ]
            agents.append(greedy_agent_class())
        else:
            agents = [agent_class(client=self.client, agent_id=f"agent_{i}", cfg=self.cfg, log_path=self.log_path, horizon_length=self.horizon_length)
                for i in range(self.cfg.experiment.agents.num)
            ]
        return agents

    def schedule_vs_newcomer(self):
        greedy_agent = self.agents[-1] # Assuming the greedy agent is the last in the list
        pool = self.agents[:-1]
        schedule = []
        role_flag = True # True means greedy is donor, False means greedy is recipient
        for opponent in pool:
            if role_flag:
                schedule.append((greedy_agent, opponent))   # greedy is donor
            else:
                schedule.append((opponent, greedy_agent))   # greedy is recipient
            role_flag = not role_flag           # alternate for next match
        return schedule

    def run_simulation(self, is_test):
        """
        run simulation
        """
        run = init_log(self.cfg, is_test)
        scenario_data = {}
        scenario_data["config"] = OmegaConf.to_container(self.cfg, resolve=True)
        for episode in range(self.cfg.experiment.env.num_episodes):
            episode_logs = {}
            episode_data = {}
            historical_messages = []
            self.env.reset(self.agents)
            all_pairs_schedule = self.schedule_vs_newcomer()
            resources_start = [agent.resources for agent in self.agents]

            for round_index, pair in enumerate(all_pairs_schedule):
                import time as _time
                if round_index > 0:
                    _time.sleep(1.0)  # Rate limiting delay between rounds
                print(f"Round {round_index + 1}")
                donor, recipient = pair
                # for donor, recipient in round_pairings:
                resources_before_donation = {"donor": donor.resources, "recipient": recipient.resources}

                if self.insert_greedy_agent and isinstance(donor, GreedyAgent):
                    donor_action = donor.donate()
                    donor_justification = ""
                else:
                    if self.is_gossip:
                        donor_justification, donor_action = donor.donate(self.rules, recipient, historical_messages)
                    else:
                        donor_justification, donor_action = donor.donate(self.rules, recipient)

                assert donor_action in ["cooperate", "defect"], "Invalid action taken by donor."
                if donor_action == "defect":
                    donation = 0
                    received_benefit = 0
                else: # cooperate
                    donation = self.cfg.experiment.env.cost
                    received_benefit = self.cfg.experiment.env.benefit
                print(f"Donor: {donor.name}, Recipient: {recipient.name}, Action : {donor_action}, Donation: {donation} \n")
            
                print(f"Donor's Justification: {donor_justification} \n")
                
                assert donation <= donor.resources, "Donation amount is invalid."
                donation_ratio = compute_donation_ratio(donation, donor.resources)

                if self.is_gossip:
                    if self.insert_greedy_agent and isinstance(recipient, GreedyAgent):
                        recipient_message = recipient.gossip()
                        recipient_justification = ""
                        recipient_tone = ""
                    else:
                        recipient_justification, recipient_tone, recipient_message = recipient.gossip(self.rules, donor, donation, donation_ratio, received_benefit, historical_messages)
                    print(f"Recipient: {recipient.name}, Selected Tone: {recipient_tone}, Gossip: {recipient_message},\n Recipient's Justification: {recipient_justification}\n")
                    message_summary = {"round": {round_index+1}, "donor": donor.name, "recipient": recipient.name, f"message from {recipient.name}": recipient_message}
                    historical_messages.append(message_summary)
                    cur_round_info = {"donor_name": donor.name, "recipient_name": recipient.name, "resources_before_donation": resources_before_donation, "donation": donation, "donation_ratio": donation_ratio, "donor_justification": donor_justification, "received_benefit": received_benefit, "recipient_justification": recipient_justification, "tone":recipient_tone, "gossip": recipient_message} # Update the trajectory(STM) of the players with this current round info 
                else:
                    cur_round_info = {"donor_name": donor.name, "recipient_name": recipient.name, "resources_before_donation": resources_before_donation, "donation": donation, "donation_ratio": donation_ratio, "donor_justification": donor_justification, "received_benefit": received_benefit}
                episode_data[f"round_{round_index+1}"] = cur_round_info # Log data per round
                donor.update_stm(round_index+1, cur_round_info)
                recipient.update_stm(round_index+1, cur_round_info)
                self.env.step(donor, recipient, donation, received_benefit)
                # Update reward signals and donations made
                donor.donations.append(donation)
                donor.donation_ratios.append(donation_ratio)
                recipient.benefits.append(received_benefit)
                donor.rewards.append(-donation)
                recipient.rewards.append(received_benefit)

            # Wandb Logging for the episode
            greedy_agent = self.agents[-1]
            donations = greedy_agent.donations
            donation_ratios = greedy_agent.donation_ratios
            rewards = greedy_agent.rewards
            benefits = greedy_agent.benefits
            assert len(donations) == len(donation_ratios) == len(benefits) == len(rewards)/2, "Mismatch in lengths of donations, donation ratios, rewards, and benefits."
            for step in range(len(donations)):
                wandb.log({f"Greedy Agent Donation Per Step": donations[step], f"Greedy Agent Donation Ratio Per Step": donation_ratios[step], f"Greedy Agent Reward Per Step": rewards[step], f"Greedy Agent Benefit Per Step": benefits[step]})

            # Compute metrics
            avg_donation_all = [compute_avg_donation(agent) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            avg_donation_ratios_all = [compute_avg_donation_ratio(agent) for agent in self.agents]
            returns_all = [compute_return(agent, resources_start[agent_idx]) for agent_idx, agent in enumerate(self.agents)]
            discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, self.discount_factor) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            image_score_all = [compute_image_score(agent) for agent in self.agents]
            episode_logs["interaction"] = episode_data
            scenario_data[f"episode_{episode+1}"] = episode_logs
            # Log metrics per episode    
            logging_metrics(avg_donation_all, avg_donation_ratios_all, returns_all, discounted_cumulative_rewards_all, image_score_all)
        close_log(run)
        # print(scenario_data)
        with open(f'{self.log_path}', "w") as f:
            json.dump(scenario_data, f, indent=4)
        print(f"Simulation completed. Logs saved to {self.log_path}")