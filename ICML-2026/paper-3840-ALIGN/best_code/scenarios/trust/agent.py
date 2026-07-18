from scenarios.trust.prompt import investorPrompt, responderPrompt, investorGossipPrompt, responderGossipPrompt
from scenarios.trust.utility import InvestmentResponse, ReturnResponse, InvestorGossipResponse, ResponderGossipResponse, extract_json
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
        self.discount_factor = cfg.experiment.env.discount_factor

    def invest_policy_llm(self, rule_prompt, investment_prompt):
        response_class = InvestmentResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": investment_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.investor_action # get the investment amount from the response
        elif self.cfg.llm.api == 'together':
            if self.cfg.llm.model == "deepseek-reasoner":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investment_prompt}
                    ],
                    reasoning={"enabled": True},
                    )
            elif self.cfg.llm.model == "deepseek-chat":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investment_prompt}
                    ],
                    reasoning={"enabled": False},
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investment_prompt}
                    ],
                    response_format={'type': 'json_schema',
                                    "schema": response_class.model_json_schema()}
                )
            try:
                response_content = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                response_content = json.loads(extract_json(response.choices[0].message.content))
            return response_content['justification'], response_content['investor_action']
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, investment_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['investor_action']
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": investment_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["investor_action"]
    
    def respond_policy_llm(self, rule_prompt, return_prompt):
        response_class = ReturnResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": return_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.responder_action # get the return amount from the response
        elif self.cfg.llm.api == 'together':
            if self.cfg.llm.model == "deepseek-reasoner":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": return_prompt}
                    ],
                    reasoning={"enabled": True},
                    )
            elif self.cfg.llm.model == "deepseek-chat":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": return_prompt}
                    ],
                    reasoning={"enabled": False},
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": return_prompt}
                    ],
                    response_format={'type': 'json_schema',
                                    "schema": response_class.model_json_schema()}
                )
            try:
                response_content = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                response_content = json.loads(extract_json(response.choices[0].message.content))
            return response_content['justification'], response_content['responder_action']
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, return_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['responder_action']
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": return_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["responder_action"]
        
    def invest(self, rules, responder): # for investor action
        """ Handle the investment process for the agent """
        investment_prompt = investorPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(investor_name=self.name, responder_name=responder.name, investor_resources=self.resources, responder_resources=responder.resources, horizon_length=self.horizon_length, discount_factor=self.discount_factor, stm=self.stm).strip()
        justification, investor_action = self.invest_policy_llm(rules, investment_prompt)
        return justification, investor_action
    
    def respond(self, rules, investor, investment, investment_ratio, benefit): # for responder action
        """ Handle the return process for the agent """
        return_prompt = responderPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(responder_name=self.name, investor_name=investor.name, responder_resources=self.resources, investor_resources=investor.resources, investment=investment, investment_ratio=investment_ratio, benefit=benefit, horizon_length=self.horizon_length, discount_factor=self.discount_factor, stm=self.stm).strip()
        justification, responder_action = self.respond_policy_llm(rules, return_prompt)
        return justification, responder_action
    
    def update_stm(self, round_idx, round_info):
        if self.name == round_info["investor_name"]: # agent role is investor 
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["responder_name"]}. Before interaction, I had {round_info["resources_before_investment"]["investor"]} and {round_info["responder_name"]} had {round_info["resources_before_investment"]["responder"]}. This round, I took the role of investor and {round_info["responder_name"]} was the responder. I chose to invest {round_info["investment"]} ({round_info["investment_ratio"]} of my current stock) to {round_info["responder_name"]}. This is my justification for the investment: "{round_info["investor_justification"]}". The responder returned {round_info["returned_amount"]} ({round_info["returned_ratio"]} of their received amount) to me. 
            """
        else: # agent role is responder
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["investor_name"]}. Before interaction, I had {round_info["resources_before_investment"]["responder"]} and {round_info["investor_name"]} had {round_info["resources_before_investment"]["investor"]}. This round, I took the role of responder and {round_info["investor_name"]} was the investor. The investor chose to invest {round_info["investment"]} ({round_info["investment_ratio"]} of their current stock) to me. This is my justification for the investment: "{round_info["investor_justification"]}". I chose to return {round_info["returned_amount"]} ({round_info["returned_ratio"]} of my received amount) to the investor. This is my justification for the return: "{round_info["responder_justification"]}".
            """
        self.stm.append(round_context)


class GossipAgent(BaselineAgent):
    def __init__(self, client, agent_id, cfg, log_path, horizon_length):
        super().__init__(client, agent_id, cfg, log_path, horizon_length)

    def investor_gossip_policy_llm(self, rule_prompt, investor_gossip_prompt):
        response_class = InvestorGossipResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": investor_gossip_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.tone, response.choices[0].message.parsed.gossip # get the gossip from the response
        elif self.cfg.llm.api == 'together':
            if self.cfg.llm.model == "deepseek-reasoner":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investor_gossip_prompt}
                    ],
                    reasoning={"enabled": True},
                    )
            elif self.cfg.llm.model == "deepseek-chat":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investor_gossip_prompt}
                    ],
                    reasoning={"enabled": False},
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": investor_gossip_prompt}
                    ],
                    response_format={'type': 'json_schema',
                                    "schema": response_class.model_json_schema()}
                )
            try:
                response_content = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                response_content = json.loads(extract_json(response.choices[0].message.content))
            return response_content['justification'], response_content['tone'], response_content['gossip']
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, investor_gossip_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['tone'], response_content['gossip']
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": investor_gossip_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["tone"], structured_response["gossip"]
        
    def responder_gossip_policy_llm(self, rule_prompt, responder_gossip_prompt):
        response_class = ResponderGossipResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt}, # Instructions to the model that are prioritized ahead of user messages, following chain of command. Previously called the system prompt.
                    {"role": "user", "content": responder_gossip_prompt}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.tone, response.choices[0].message.parsed.gossip # get the gossip from the response
        elif self.cfg.llm.api == 'together':
            if self.cfg.llm.model == "deepseek-reasoner":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": responder_gossip_prompt}
                    ],
                    reasoning={"enabled": True},
                    )
            elif self.cfg.llm.model == "deepseek-chat":
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": responder_gossip_prompt}
                    ],
                    reasoning={"enabled": False},
                    )
            else:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {"role": "system", "content": rule_prompt},
                        {"role": "user", "content": responder_gossip_prompt}
                    ],
                    response_format={'type': 'json_schema',
                                    "schema": response_class.model_json_schema()}
                )
            try:
                response_content = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                response_content = json.loads(extract_json(response.choices[0].message.content))
            return response_content['justification'], response_content['tone'], response_content['gossip']
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, responder_gossip_prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['tone'], response_content['gossip']
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": responder_gossip_prompt}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["tone"], structured_response["gossip"]
        
    def investor_gossip(self, rules, responder, investment, investment_ratio, benefit, returned_amount, returned_ratio, historical_messages): # for investor gossip
        """ Handle the investor-side gossip process for the trust game """
        investor_gossip_prompt = investorGossipPrompt(horizon=self.horizon, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(investor_name=self.name, responder_name=responder.name, investor_resources=self.resources, responder_resources=responder.resources, investment=investment, investment_ratio=investment_ratio, benefit=benefit, returned_amount=returned_amount, returned_ratio=returned_ratio, horizon_length=self.horizon_length, discount_factor=self.discount_factor, historical_messages=historical_messages, stm=self.stm).strip()
        justification, tone, gossip_response = self.investor_gossip_policy_llm(rules, investor_gossip_prompt)
        return justification, tone, gossip_response
    
    def responder_gossip(self, rules, investor, investment, investment_ratio, benefit, returned_amount, returned_ratio, historical_messages): # for responder gossip
        """ Handle the responder-side gossip process for the trust game """
        responder_gossip_prompt = responderGossipPrompt(horizon=self.horizon, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(responder_name=self.name, investor_name=investor.name, responder_resources=self.resources, investor_resources=investor.resources, investment=investment, investment_ratio=investment_ratio, benefit=benefit, returned_amount=returned_amount, returned_ratio=returned_ratio, horizon_length=self.horizon_length, discount_factor=self.discount_factor, historical_messages=historical_messages, stm=self.stm).strip()
        justification, tone, gossip_response = self.responder_gossip_policy_llm(rules, responder_gossip_prompt)
        return justification, tone, gossip_response
    
    def invest(self, rules, responder, historical_messages):
        """ Handle the investment process for the agent when gossip is enabled"""
        investment_prompt = investorPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(investor_name=self.name, responder_name=responder.name, investor_resources=self.resources, responder_resources=responder.resources, stm=self.stm, historical_messages=historical_messages, horizon_length=self.horizon_length, discount_factor=self.discount_factor).strip()
        justification, investor_action = self.invest_policy_llm(rules, investment_prompt)
        return justification, investor_action
    
    def respond(self, rules, investor, investment, investment_ratio, benefit, historical_messages):
        """ Handle the return process for the agent when gossip is enabled"""
        return_prompt = responderPrompt(horizon=self.horizon, is_gossip=self.is_gossip, use_equilibrium_knowledge=self.use_equilibrium_knowledge).substitute(responder_name=self.name, investor_name=investor.name, responder_resources=self.resources, investor_resources=investor.resources, investment=investment, investment_ratio=investment_ratio, benefit=benefit, stm=self.stm, historical_messages=historical_messages, horizon_length=self.horizon_length, discount_factor=self.discount_factor).strip()
        justification, responder_action = self.respond_policy_llm(rules, return_prompt)
        return justification, responder_action
    
    def update_stm(self, round_idx, round_info):
        """ Update STM with gossip information """
        if self.name == round_info["investor_name"]: # agent role is investor 
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["responder_name"]}. Before interaction, I had {round_info["resources_before_investment"]["investor"]} and {round_info["responder_name"]} had {round_info["resources_before_investment"]["responder"]}. This round, I took the role of investor and {round_info["responder_name"]} was the responder. I chose to invest {round_info["investment"]} ({round_info["investment_ratio"]} of my current stock) to {round_info["responder_name"]}. This is my justification for the investment: "{round_info["investor_justification"]}". The responder returned {round_info["returned_amount"]} ({round_info["returned_ratio"]} of their received amount) to me. After the interaction, I broadcasted the following message about {round_info["responder_name"]}: "{round_info["investor_gossip"]}". This is my justification for the gossip: "{round_info["investor_gossip_justification"]}". The responder broadcasted the following message about me: "{round_info["responder_gossip"]}".
            """
        else:
            round_context = f""" In round {round_idx}, I, {self.name}, was matched with {round_info["investor_name"]}. Before interaction, I had {round_info["resources_before_investment"]["responder"]} and {round_info["investor_name"]} had {round_info["resources_before_investment"]["investor"]}. This round, I took the role of responder and {round_info["investor_name"]} was the investor. {round_info["investor_name"]} chose to invest {round_info["investment"]} ({round_info["investment_ratio"]} of my current stock) to me. This is my justification for the return: "{round_info["responder_justification"]}". I returned {round_info["returned_amount"]} ({round_info["returned_ratio"]} of the received amount) to {round_info["investor_name"]}. After the interaction, I broadcasted the following message about {round_info["investor_name"]}: "{round_info["responder_gossip"]}". This is my justification for the gossip: "{round_info["responder_gossip_justification"]}". The investor broadcasted the following message about me: "{round_info["investor_gossip"]}".
            """
        self.stm.append(round_context)