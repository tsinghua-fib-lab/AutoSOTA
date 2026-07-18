from scenarios.donor.prompt import donationPrompt, gossipPrompt
from scenarios.donor.utility import GossipResponse, BinaryDonationResponse
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
    
    def action_policy_llm(self, rule_prompt, donation_prompt):
        response_class = BinaryDonationResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": donation_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.donor_action # get the donation amount from the response
    
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": donation_prompt}
                ],
                response_format={'type': 'json_schema',
                                 "schema": response_class.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return response_content['justification'], response_content['donor_action']

        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, donation_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['donor_action']
        
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": donation_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            for _retry in range(3):
                try:
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise ValueError("empty")
                    structured_response = json.loads(content)
                    return structured_response["justification"], structured_response["donor_action"]
                except (json.JSONDecodeError, KeyError, ValueError):
                    if _retry < 2:
                        import time as _t
                        _t.sleep(2)
                        response = self.client.chat.completions.create(
                            model=self.cfg.llm.model,
                            messages=[
                                {"role": "system", "content": rule_prompt},
                                {"role": "user", "content": donation_prompt}
                            ],
                            response_format={'type': 'json_object'}
                        )
                    else:
                        return "", "defect"

    def donate(self, rules, recipient): # for donor
        """ Handle the donation process for the agent """
        donation_prompt = donationPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(donor_name=self.name, recipient_name=recipient.name, donor_resources=self.resources, recipient_resources=recipient.resources, stm=self.stm, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, termination_prob=self.cfg.experiment.env.termination_prob, discount_factor=self.cfg.experiment.env.discount_factor, horizon_length=self.horizon_length).strip()
        justification, donor_action = self.action_policy_llm(rules, donation_prompt)
        return justification, donor_action
    
    def update_stm(self, round_idx, round_info):
        resources_before_donation = round_info["resources_before_donation"]
        donation = round_info["donation"]
        benefit = round_info["received_benefit"]
        donation_ratio = round_info["donation_ratio"]
        if self.name == round_info["donor_name"]: # agent role is donor 
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["recipient_name"]}. Before interaction, I had {resources_before_donation["donor"]} and {round_info["recipient_name"]} had {resources_before_donation["recipient"]}. This round, I took the role of donor and {round_info["recipient_name"]} was the recipient. I chose to pay a cost of {donation} ({donation_ratio} of my current stock) to donate {benefit} units of the valuable resource to {round_info["recipient_name"]}. This is my justification for the donation: "{round_info["donor_justification"]}".
            """
            self.stm.append(round_context)
        else: # agent role is recipient
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["donor_name"]}. Before interaction, I had {resources_before_donation["recipient"]} and {round_info["donor_name"]} had {resources_before_donation["donor"]}. This round, I took the role of recipient and {round_info["donor_name"]} was the donor. {round_info["donor_name"]} chose to pay a cost of {donation} ({donation_ratio} of my current stock) to donate {benefit} units of the valuable resource to me.
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
            for _retry in range(3):
                try:
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise ValueError("empty")
                    structured_response = json.loads(content)
                    return structured_response["justification"], structured_response["tone"], structured_response["gossip"]
                except (json.JSONDecodeError, KeyError, ValueError):
                    if _retry < 2:
                        import time as _t
                        _t.sleep(2)
                        response = self.client.chat.completions.create(
                            model=self.cfg.llm.model,
                            messages=[
                                {"role": "system", "content": rule_prompt},
                                {"role": "user", "content": recipient_prompt}
                            ],
                            response_format={'type': 'json_object'}
                        )
                    else:
                        return "", "neutral", "No comment"

    def donate(self, rules, recipient, historical_messages): # for donor
        """ Handle the donation process for the agent """
        donation_prompt = donationPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(donor_name=self.name, recipient_name=recipient.name, donor_resources=self.resources, recipient_resources=recipient.resources, stm=self.stm, cost=self.cfg.experiment.env.cost, benefit=self.cfg.experiment.env.benefit, termination_prob=self.cfg.experiment.env.termination_prob, discount_factor=self.cfg.experiment.env.discount_factor, historical_messages=historical_messages, horizon_length=self.horizon_length).strip()
        justification, donor_action = self.action_policy_llm(rules, donation_prompt)
        return justification, donor_action
    
    def gossip(self, rules, donor, donation, donation_ratio, received_benefit, historical_messages): # for recipient
        """ Handle the gossip process for the agent """
        gossip_prompt = gossipPrompt(horizon=self.horizon, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(donor_name=donor.name, recipient_name=self.name, donor_resources=donor.resources, recipient_resources=self.resources,donation=donation, donation_ratio=donation_ratio, benefit=received_benefit, historical_messages=historical_messages, stm=self.stm, discount_factor=self.cfg.experiment.env.discount_factor, horizon_length=self.horizon_length, termination_prob=self.cfg.experiment.env.termination_prob).strip()
        justification, tone, gossip_response = self.gossip_policy_llm(rules, gossip_prompt)
        print("Tone selected: ", tone)
        print("Gossip response: ", gossip_response)
        return justification, tone, gossip_response
    
    def update_stm(self, round_idx, round_info):
        resources_before_donation = round_info["resources_before_donation"]
        donation = round_info["donation"]
        donation_ratio = round_info["donation_ratio"]
        benefit = round_info["received_benefit"]
        if self.name == round_info["donor_name"]: # agent role is donor 
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["recipient_name"]}. Before interaction, I had {resources_before_donation["donor"]} and {round_info["recipient_name"]} had {resources_before_donation["recipient"]}. This round, I took the role of donor and {round_info["recipient_name"]} was the recipient. I chose to pay a cost of {donation} ({donation_ratio} of my current stock) to donate {benefit} units of the valuable resource to {round_info["recipient_name"]}. After observing that, {round_info["recipient_name"]} spread a message about me: "{round_info["gossip"]}". This is my justification for the donation: "{round_info["donor_justification"]}".
            """
            self.stm.append(round_context)
        else: # agent role is recipient
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["donor_name"]}. Before interaction, I had {resources_before_donation["recipient"]} and {round_info["donor_name"]} had {resources_before_donation["donor"]}. This round, I took the role of recipient and {round_info["donor_name"]} was the donor. {round_info["donor_name"]} chose to pay a cost of {donation} ({donation_ratio} of my current stock) to donate {benefit} units of the valuable resource to me. After observing that, I spread a message about {round_info["donor_name"]}: "{round_info["gossip"]}". This is my justification for the message: "{round_info["recipient_justification"]}".
            """
            self.stm.append(round_context)


class GreedyAgent:
    def __init__(self):
        self.name = "Max"

    def donate(self):
        return "defect"
    
    def gossip(self):
        return "No comment"
    
    def update_stm(self, round_idx, round_info):
        pass