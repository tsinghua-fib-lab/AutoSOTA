from scenarios.pd.prompt import actionPrompt, gossipPrompt
from scenarios.pd.utility import ActionResponse, GossipResponse
import json

class BaselineAgent:
    def __init__(self, client, agent_id, cfg, log_path, horizon_length):
        self.name = cfg.experiment.agents[f"{agent_id}"].name
        self.client = client
        self.cfg = cfg
        self.log_path = log_path
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.use_equilibrium_knowledge = cfg.experiment.agents.use_equilibrium_knowledge
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = horizon_length
    
    def action_policy_llm(self, rule_prompt, action_prompt):
        response_class = ActionResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": action_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.player_action # get the donation amount from the response
    
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": action_prompt}
                ],
                response_format={'type': 'json_schema',
                                 "schema": response_class.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return response_content['justification'], response_content['player_action']

        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, action_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['player_action']
        
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": action_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["player_action"]

    def act(self, rules, recipient): # for donor
        """ Handle the donation process for the agent """
        action_prompt = actionPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(player_name=self.name, opponent_name=recipient.name, stm=self.stm, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, discount_factor=self.cfg.experiment.env.discount_factor, horizon_length=self.horizon_length).strip()
        justification, player_action = self.action_policy_llm(rules, action_prompt)
        return justification, player_action
    
    def update_stm(self, round_idx, round_info):
        if self.name == round_info["player_1"]: # agent role is player 1
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["player_2"]}. This round, I chose to { 'cooperate' if round_info["action_1"] == 'C' else 'defect' } and {round_info["player_2"]} chose to { 'cooperate' if round_info["action_2"] == 'C' else 'defect' }. As a result, I received a reward of {round_info["reward_1"]} and {round_info["player_2"]} received a reward of {round_info["reward_2"]}. This is my justification for my action: "{round_info["action_justification_1"]}".
            """
        elif self.name == round_info["player_2"]: # agent role is player 2
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["player_1"]}. This round, {round_info["player_1"]} chose to { 'cooperate' if round_info["action_1"] == 'C' else 'defect' } and I chose to { 'cooperate' if round_info["action_2"] == 'C' else 'defect' }. As a result, {round_info["player_1"]} received a reward of {round_info["reward_1"]} and I received a reward of {round_info["reward_2"]}. This is my justification for my action: "{round_info["action_justification_2"]}".
            """    
        self.stm.append(round_context)

class GossipAgent(BaselineAgent):
    def __init__(self, client, agent_id, cfg, log_path, horizon_length):
        super().__init__(client, agent_id, cfg, log_path, horizon_length)
        
    def gossip_policy_llm(self, rule_prompt, recipient_prompt): 
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
            model=self.cfg.llm.model,
            messages=[
                {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                {"role": "user", "content": recipient_prompt}
            ],
            response_format=GossipResponse,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.tone, response.choices[0].message.parsed.gossip # get the gossip from the response
        
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": recipient_prompt}
                ],
                response_format={'type': 'json_schema',
                                 "schema": GossipResponse.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return response_content['justification'], response_content['tone'], response_content['gossip']

        elif self.cfg.llm.api == "gemini": 
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, recipient_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GossipResponse,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['tone'], response_content['gossip']
        
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": recipient_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["tone"],structured_response["gossip"]

    def act(self, rules, recipient, historical_messages): # for donor
        """ Handle the donation process for the agent """
        action_prompt = actionPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(player_name=self.name, opponent_name=recipient.name, stm=self.stm, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, discount_factor=self.cfg.experiment.env.discount_factor, historical_messages=historical_messages, horizon_length=self.horizon_length).strip()
        justification, player_action = self.action_policy_llm(rules, action_prompt)
        return justification, player_action
    
    def gossip(self, rules, opponent, opponent_action, historical_messages): # for player
        """ Handle the gossip process for the agent """
        gossip_prompt = gossipPrompt(horizon=self.horizon, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(player_name=self.name, opponent_name=opponent.name, opponent_action=opponent_action, historical_messages=historical_messages, stm=self.stm, discount_factor=self.cfg.experiment.env.discount_factor, horizon_length=self.horizon_length).strip()
        justification, tone, gossip_response = self.gossip_policy_llm(rules, gossip_prompt)
        print("Tone selected: ", tone)
        print("Gossip response: ", gossip_response)
        return justification, tone, gossip_response
    
    def update_stm(self, round_idx, round_info):
        if self.name == round_info["player_1"]: # agent role is player 1
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["player_2"]}. This round, I chose to { 'cooperate' if round_info["action_1"] == 'C' else 'defect' } and {round_info["player_2"]} chose to { 'cooperate' if round_info["action_2"] == 'C' else 'defect' }. After executing our actions, I broadcasted a message about {round_info["player_2"]}: "{round_info["message_1"]}", and {round_info["player_2"]} broadcasted a message about me: "{round_info["message_2"]}". As a result, I received a reward of {round_info["reward_1"]} and {round_info["player_2"]} received a reward of {round_info["reward_2"]}. This is my justification for my action: "{round_info["action_justification_1"]}", and this is my justification for my message: "{round_info["gossip_justification_1"]}".
            """
        elif self.name == round_info["player_2"]: # agent role is player 2
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["player_1"]}. This round, {round_info["player_1"]} chose to { 'cooperate' if round_info["action_1"] == 'C' else 'defect' } and I chose to { 'cooperate' if round_info["action_2"] == 'C' else 'defect' }. After executing our actions, {round_info["player_1"]} broadcasted a message about me: "{round_info["message_1"]}", and I broadcasted a message about {round_info["player_1"]}: "{round_info["message_2"]}". As a result, {round_info["player_1"]} received a reward of {round_info["reward_1"]} and I received a reward of {round_info["reward_2"]}. This is my justification for my action: "{round_info["action_justification_2"]}", and this is my justification for my message: "{round_info["gossip_justification_2"]}".
            """
        self.stm.append(round_context)


class GreedyAgent:
    def __init__(self):
        self.name = "Max"

    def act(self):
        return "D"

    def gossip(self):
        return "No comment"

    def update_stm(self, round_idx, round_info):
        pass