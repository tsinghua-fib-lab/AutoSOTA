from scenarios.pd.agent import BaselineAgent, GossipAgent, GreedyAgent
from scenarios.pd.env import PDEnv
from scenarios.pd.prompt import rulePrompt
from scenarios.pd.utility import *
from scenarios.pd.log_metrics import *
import numpy as np
from itertools import combinations
import json

class PDRunner:
    def __init__(self, cfg, client, log_path):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path
        self.env = PDEnv(cfg)
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.discount_factor = cfg.experiment.env.discount_factor
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = self.cfg.experiment.agents.num * (self.cfg.experiment.agents.num - 1) / 2 if self.horizon == "finite" else np.inf
        self.agents = self.init_agents()
        self.rules = rulePrompt(horizon=self.cfg.experiment.env.horizon, is_gossip=self.is_gossip).substitute(discount_factor=self.discount_factor, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, horizon_length=self.horizon_length).strip()

    def init_agents(self):
        if self.is_gossip:
            agent_class = GossipAgent
        else:
            agent_class = BaselineAgent
        agents = [agent_class(client=self.client, agent_id=f"agent_{i}", cfg=self.cfg, log_path=self.log_path, horizon_length=self.horizon_length)
            for i in range(self.cfg.experiment.agents.num)
        ]
        return agents

    def round_robin_pd_game(self, agents):
        """
        Return an ordered list of unordered pairs (a, b) such that:
        1) Each unordered pair appears exactly once.
        2) The order spreads appearances across rounds (round-robin 'circle' method).

        One pair plays per round — the list order IS the round order.
        """
        agents = list(agents)
        n = len(agents)
        if n < 2:
            return []

        # If odd, insert a bye (None) so the circle method works cleanly
        players = agents[:]
        if n % 2 == 1:
            players.append(None)
            n += 1

        rounds = []
        for _ in range(n - 1):
            # Pair first half with reversed second half
            pairs_this_round = []
            for i in range(n // 2):
                a = players[i]
                b = players[-(i + 1)]
                if a is not None and b is not None:
                    pairs_this_round.append((a, b))
            rounds.append(pairs_this_round)

            # Rotate everything except the first element
            players = [players[0]] + [players[-1]] + players[1:-1]

        # Flatten rounds to a single pair-per-round schedule
        schedule = [pair for rnd in rounds for pair in rnd]
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
            # rounds = self.round_robin_donor_game(self.agents)
            all_pairs_schedule = self.round_robin_pd_game(self.agents)

            for round_index, pair in enumerate(all_pairs_schedule):
                actions = []
                action_justifications = []
                for agent_id in range(2):
                    if self.is_gossip:
                        action_justification, action = pair[agent_id].act(self.rules, pair[1-agent_id], historical_messages)
                    else:
                        action_justification, action = pair[agent_id].act(self.rules, pair[1-agent_id])
                    assert action in ["C", "D"], "Invalid action taken by agent {}.".format(agent_id)
                    actions.append(action)
                    action_justifications.append(action_justification)
                rewards = self.env.step(actions)
                print(f"Round {round_index+1}: Player 1: {pair[0].name}, Action: {actions[0]}, Player 2: {pair[1].name}, Action: {actions[1]}, Rewards: {rewards}\n")
                
                if self.is_gossip:
                    tones = []
                    messages = []
                    message_justifications = []
                    for agent_id in range(2):
                        gossip_justification, tone, message = pair[agent_id].gossip(self.rules, pair[1-agent_id], actions[1-agent_id], historical_messages)
                        messages.append(message)
                        tones.append(tone)
                        message_justifications.append(gossip_justification)
                    message_summary = {"round": {round_index+1}, "player_1": pair[0].name, "player_2": pair[1].name, f"message from {pair[0].name}": messages[0], f"message from {pair[1].name}": messages[1]}
                    historical_messages.append(message_summary)
                    cur_round_info = {"player_1": pair[0].name, "player_2": pair[1].name, "action_1": actions[0], "action_2": actions[1], "reward_1": rewards[0], "reward_2": rewards[1], "action_justification_1": action_justifications[0], "action_justification_2": action_justifications[1], "tone_1": tones[0], "tone_2": tones[1], "message_1": messages[0], "message_2": messages[1], "gossip_justification_1": message_justifications[0], "gossip_justification_2": message_justifications[1]} # Update the trajectory(STM) of the players with this current round info
                else:
                    cur_round_info = {"player_1": pair[0].name, "player_2": pair[1].name, "action_1": actions[0], "action_2": actions[1], "reward_1": rewards[0], "reward_2": rewards[1], "action_justification_1": action_justifications[0], "action_justification_2": action_justifications[1]} # Update the trajectory(STM) of the players with this current round info
                episode_data[f"round_{round_index+1}"] = cur_round_info # Log data per round

                for idx, agent in enumerate(pair):
                    agent.update_stm(round_index+1, cur_round_info)
                    agent.actions.append(actions[idx])
                    agent.rewards.append(rewards[idx])
            
            for k, agent in enumerate(self.agents):
                rewards = agent.rewards
                actions_bits = [1 if action == "C" else 0 for action in agent.actions]
                for step in range(len(actions_bits)):
                    wandb.log({f"Agent {k} Action Per Step": actions_bits[step], f"Agent {k} Reward Per Step": rewards[step]})

            # Compute metrics
            returns_all = discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, 1.0) for agent in self.agents]
            discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, self.discount_factor) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            image_score_all = [compute_image_score(agent) for agent in self.agents]
            cooperation_ratio_all = [compute_cooperation_ratio(agent) for agent in self.agents]

            episode_logs["interaction"] = episode_data
            scenario_data[f"episode_{episode+1}"] = episode_logs
            # Log metrics per episode    
            logging_metrics(cooperation_ratio_all, returns_all, discounted_cumulative_rewards_all, image_score_all)
        close_log(run)
        # print(scenario_data)
        with open(f'{self.log_path}', "w") as f:
            json.dump(scenario_data, f, indent=4)
        print(f"Simulation completed. Logs saved to {self.log_path}")


class PDRunnerrWithGreedyAgent:
    def __init__(self, cfg, client, log_path):
        self.cfg = cfg
        self.client = client
        self.log_path = log_path
        self.env = PDEnv(cfg)
        self.insert_greedy_agent = cfg.experiment.agents.insert_greedy_agent
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.discount_factor = cfg.experiment.env.discount_factor
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = self.cfg.experiment.agents.num * (self.cfg.experiment.agents.num - 1) / 2 if self.horizon == "finite" else np.inf
        self.agents = self.init_agents()
        self.rules = rulePrompt(horizon=self.cfg.experiment.env.horizon, is_gossip=self.is_gossip).substitute(discount_factor=self.discount_factor, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, horizon_length=self.horizon_length).strip()

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
        for opponent in pool:
            schedule.append((greedy_agent, opponent))   # greedy is donor
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
            greedy_agent = self.agents[-1]

            for round_index, pair in enumerate(all_pairs_schedule):
                assert isinstance(greedy_agent, GreedyAgent)

                actions = [greedy_agent.act()]
                action_justifications = [""]  # Greedy agent does not provide justification
                if self.is_gossip:
                    action_justification, action = pair[1].act(self.rules, greedy_agent, historical_messages)
                else:
                    action_justification, action = pair[1].act(self.rules, greedy_agent)
                assert action in ["C", "D"], "Invalid action taken by agent"
                actions.append(action)
                action_justifications.append(action_justification)
                rewards = self.env.step(actions)
                print(f"Round {round_index+1}: Player 1: {pair[0].name}, Action: {actions[0]}, Player 2: {pair[1].name}, Action: {actions[1]}, Rewards: {rewards}\n")
                
                if self.is_gossip:
                    tones = [""]
                    messages = [greedy_agent.gossip()]
                    message_justifications = [""]  # Greedy agent does not provide justification
                    gossip_justification, tone, message = pair[1].gossip(self.rules, greedy_agent, actions[0], historical_messages)
                    messages.append(message)
                    tones.append(tone)
                    message_justifications.append(gossip_justification)
                    message_summary = {"round": {round_index+1}, "player_1": pair[0].name, "player_2": pair[1].name, f"message from {pair[0].name}": messages[0], f"message from {pair[1].name}": messages[1]}
                    historical_messages.append(message_summary)
                    cur_round_info = {"player_1": pair[0].name, "player_2": pair[1].name, "action_1": actions[0], "action_2": actions[1], "reward_1": rewards[0], "reward_2": rewards[1], "action_justification_1": action_justifications[0], "action_justification_2": action_justifications[1], "tone_1": tones[0], "tone_2": tones[1], "message_1": messages[0], "message_2": messages[1], "gossip_justification_1": message_justifications[0], "gossip_justification_2": message_justifications[1]} # Update the trajectory(STM) of the players with this current round info
                else:
                    cur_round_info = {"player_1": pair[0].name, "player_2": pair[1].name, "action_1": actions[0], "action_2": actions[1], "reward_1": rewards[0], "reward_2": rewards[1], "action_justification_1": action_justifications[0], "action_justification_2": action_justifications[1]} # Update the trajectory(STM) of the players with this current round info
                episode_data[f"round_{round_index+1}"] = cur_round_info # Log data per round

                for idx, agent in enumerate(pair):
                    agent.update_stm(round_index+1, cur_round_info)
                    agent.actions.append(actions[idx])
                    agent.rewards.append(rewards[idx])

            for step in range(len(greedy_agent.rewards)):
                wandb.log({f"Greedy Agent Reward Per Step": greedy_agent.rewards[step]})

            for k, agent in enumerate(self.agents):
                rewards = agent.rewards
                actions_bits = [1 if action == "C" else 0 for action in agent.actions]
                for step in range(len(actions_bits)):
                    wandb.log({f"Agent {k} Action Per Step": actions_bits[step], f"Agent {k} Reward Per Step": rewards[step]})

            # Compute metrics
            returns_all = discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, 1.0) for agent in self.agents]
            discounted_cumulative_rewards_all = [compute_dis_cum_reward(agent, self.discount_factor) for agent in self.agents] # This should be appended to the agent's long-term memory as a feedback signal
            image_score_all = [compute_image_score(agent) for agent in self.agents]
            cooperation_ratio_all = [compute_cooperation_ratio(agent) for agent in self.agents]

            episode_logs["interaction"] = episode_data
            scenario_data[f"episode_{episode+1}"] = episode_logs
            # Log metrics per episode    
            logging_metrics(cooperation_ratio_all, returns_all, discounted_cumulative_rewards_all, image_score_all)
        close_log(run)
        # print(scenario_data)
        with open(f'{self.log_path}', "w") as f:
            json.dump(scenario_data, f, indent=4)
        print(f"Simulation completed. Logs saved to {self.log_path}")