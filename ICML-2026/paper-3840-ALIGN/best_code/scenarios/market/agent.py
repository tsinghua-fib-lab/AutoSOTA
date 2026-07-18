from scenarios.market.prompt import sellerPrompt, buyerPrompt, buyerGossipPrompt
from scenarios.market.utility import SellerActionResponse, BuyerActionResponse, BuyerGossipResponse
import json

class SellerBaselineAgent:
    def __init__(self, client, agent_id, cfg, log_path, horizon_length, env):
        self.name = cfg.experiment.agents[f"{agent_id}"].name
        self.client = client
        self.cfg = cfg
        self.log_path = log_path
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.use_equilibrium_knowledge = cfg.experiment.agents.use_equilibrium_knowledge
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = horizon_length
        self.env = env
        self.stm = []

    def sell_policy_llm(self, rule_prompt, seller_prompt_text):
        response_class = SellerActionResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt},
                    {"role": "user", "content": seller_prompt_text}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.seller_action
        
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": seller_prompt_text}
                ],
                response_format={'type': 'json_schema',
                                 "schema": response_class.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return response_content['justification'], response_content['seller_action']
        
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, seller_prompt_text],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['seller_action']
        
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": seller_prompt_text}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["seller_action"]
        
    def sell(self, rules, buyer):
        """
        Seller chooses quality: H or L.
        """
        seller_prompt_text = sellerPrompt(
            horizon=self.horizon,
            is_gossip=self.is_gossip,
            use_equilibrium_knowledge=self.use_equilibrium_knowledge,
        ).substitute(
            seller_name=self.name,
            buyer_name=buyer.name,
            stm=self.stm,
            discount_factor=self.cfg.experiment.env.discount_factor,
            horizon_length=self.horizon_length,
            seller_Hc_reward=self.env.payoff_matrix[("H", "c")][0],
            seller_Hs_reward=self.env.payoff_matrix[("H", "s")][0],
            seller_Lc_reward=self.env.payoff_matrix[("L", "c")][0],
            seller_Ls_reward=self.env.payoff_matrix[("L", "s")][0],
        ).strip()

        justification, seller_action = self.sell_policy_llm(rules, seller_prompt_text)
        return justification, seller_action

    def update_stm(self, round_idx, round_info):
        """
        Suggested round_info:
          - buyer_name, buyer_action
          - seller_action, seller_reward
          - seller_justification
          - optional: gossip, gossip_tone
        """
        buyer_name = round_info["buyer_name"]
        seller_action = round_info["seller_action"]
        buyer_action = round_info["buyer_action"]
        seller_reward = round_info["seller_reward"]
        seller_justification = round_info.get("seller_justification", "")

        round_context = f""" In round {round_idx}, I, {self.name}, was matched with buyer {buyer_name}. I chose {seller_action}. The buyer chose {buyer_action}. My reward was {seller_reward}. This is my justification: "{seller_justification}".
        """
        self.stm.append(round_context)


class SellerGossipAgent(SellerBaselineAgent):
    """
    Sellers do NOT gossip in this market game.
    Kept for symmetry: seller can still review historical_messages when is_gossip=True.
    """
    def __init__(self, client, agent_id, cfg, log_path, horizon_length, env):
        super().__init__(client, agent_id, cfg, log_path, horizon_length, env)

    def sell(self, rules, buyer, historical_messages):
        """
        Seller chooses quality: H or L.
        """
        seller_prompt_text = sellerPrompt(
            horizon=self.horizon,
            is_gossip=self.is_gossip,
            use_equilibrium_knowledge=self.use_equilibrium_knowledge,
        ).substitute(
            seller_name=self.name,
            buyer_name=buyer.name,
            stm=self.stm,
            historical_messages=historical_messages,
            discount_factor=self.cfg.experiment.env.discount_factor,
            horizon_length=self.horizon_length,
            seller_Hc_reward=self.env.payoff_matrix[("H", "c")][0],
            seller_Hs_reward=self.env.payoff_matrix[("H", "s")][0],
            seller_Lc_reward=self.env.payoff_matrix[("L", "c")][0],
            seller_Ls_reward=self.env.payoff_matrix[("L", "s")][0],
        ).strip()
        justification, seller_action = self.sell_policy_llm(rules, seller_prompt_text)
        return justification, seller_action

    def update_stm(self, round_idx, round_info):
        """
        Suggested round_info:
          - buyer_name, buyer_action
          - seller_action, seller_reward
          - seller_justification
          - optional: gossip, gossip_tone
        """
        buyer_name = round_info["buyer_name"]
        seller_action = round_info["seller_action"]
        buyer_action = round_info["buyer_action"]
        seller_reward = round_info["seller_reward"]
        seller_justification = round_info.get("seller_justification", "")

        extra = f' After the round, the buyer posted a public message about me: "{round_info["gossip"]}".'

        round_context = f""" In round {round_idx}, I, {self.name}, was matched with buyer {buyer_name}. I chose {seller_action}. The buyer chose {buyer_action}. My reward was {seller_reward}. This is my justification: "{seller_justification}".{extra}
        """
        self.stm.append(round_context)


# ============================================================
# Buyer Agents (fixed role)
# ============================================================

class BuyerBaselineAgent:
    def __init__(self, client, agent_id, cfg, log_path, horizon_length, env):
        self.name = cfg.experiment.agents[f"{agent_id}"].name
        self.client = client
        self.cfg = cfg
        self.log_path = log_path
        self.is_gossip = cfg.experiment.agents.is_gossip
        self.use_equilibrium_knowledge = cfg.experiment.agents.use_equilibrium_knowledge
        self.horizon = cfg.experiment.env.horizon
        self.horizon_length = horizon_length
        self.env = env
        self.stm = []

    def buy_policy_llm(self, rule_prompt, buyer_prompt_text):
        response_class = BuyerActionResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt},
                    {"role": "user", "content": buyer_prompt_text}
                ],
                response_format=response_class,
            )
            return response.choices[0].message.parsed.justification, response.choices[0].message.parsed.buyer_action
        
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": buyer_prompt_text}
                ],
                response_format={'type': 'json_schema',
                                 "schema": response_class.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return response_content['justification'], response_content['buyer_action']
        
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, buyer_prompt_text],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return response_content['justification'], response_content['buyer_action']
        
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": buyer_prompt_text}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return structured_response["justification"], structured_response["buyer_action"]

    def buy(self, rules, seller, historical_messages=""):
        """
        Buyer chooses c / s / none.
        """
        buyer_prompt_text = buyerPrompt(
            horizon=self.horizon,
            is_gossip=self.is_gossip,
            use_equilibrium_knowledge=self.use_equilibrium_knowledge,
        ).substitute(
            buyer_name=self.name,
            seller_name=seller.name,
            stm=self.stm,
            historical_messages=historical_messages,
            discount_factor=self.cfg.experiment.env.discount_factor,
            horizon_length=self.horizon_length,
            buyer_Hc_reward=self.env.payoff_matrix[("H", "c")][1],
            buyer_Hs_reward=self.env.payoff_matrix[("H", "s")][1],
            buyer_Lc_reward=self.env.payoff_matrix[("L", "c")][1],
            buyer_Ls_reward=self.env.payoff_matrix[("L", "s")][1],
        ).strip()

        justification, buyer_action = self.buy_policy_llm(rules, buyer_prompt_text)
        return justification, buyer_action

    def update_stm(self, round_idx, round_info):
        """
        Suggested round_info:
          - seller_name, seller_action
          - buyer_action, buyer_reward
          - buyer_justification
          - optional: gossip, gossip_tone, gossip_justification
        """
        seller_name = round_info["seller_name"]
        seller_action = round_info["seller_action"]
        buyer_action = round_info["buyer_action"]
        buyer_reward = round_info["buyer_reward"]
        buyer_justification = round_info.get("buyer_justification", "")

        extra = ""
        if round_info.get("gossip", ""):
            extra = f' After the round, I posted a public message about {seller_name}: "{round_info["gossip"]}".'

        round_context = f""" In round {round_idx}, I, {self.name}, was matched with seller {seller_name}. The seller chose {seller_action}. I chose {buyer_action}. My reward was {buyer_reward}. This is my justification: "{buyer_justification}".{extra}
        """
        self.stm.append(round_context)


class BuyerGossipAgent(BuyerBaselineAgent):
    def __init__(self, client, agent_id, cfg, log_path, horizon_length, env):
        super().__init__(client, agent_id, cfg, log_path, horizon_length, env)

    def gossip_policy_llm(self, rule_prompt, gossip_prompt_text):
        response_class = BuyerGossipResponse
        if self.cfg.llm.api == 'openai' or self.cfg.llm.api == 'gemini-v2':
            response = self.client.beta.chat.completions.parse(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "developer", "content": rule_prompt},
                    {"role": "user", "content": gossip_prompt_text}
                ],
                response_format=response_class,
            )
            return (response.choices[0].message.parsed.justification,
                    response.choices[0].message.parsed.tone,
                    response.choices[0].message.parsed.gossip)
        elif self.cfg.llm.api == 'together':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": gossip_prompt_text}
                ],
                response_format={'type': 'json_schema',
                                 "schema": response_class.model_json_schema()}
            )
            response_content = json.loads(response.choices[0].message.content)
            return (response_content['justification'],
                    response_content['tone'],
                    response_content['gossip'])
        elif self.cfg.llm.api == "gemini":
            response = self.client.models.generate_content(
                model=self.cfg.llm.model,
                contents=[rule_prompt, gossip_prompt_text],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_class,
                }
            )
            response_content = json.loads(response.text)
            return (response_content['justification'],
                    response_content['tone'],
                    response_content['gossip'])
        elif self.cfg.llm.api == 'deepseek':
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": rule_prompt},
                    {"role": "user", "content": gossip_prompt_text}
                ],
                response_format={'type': 'json_object'}
            )
            structured_response = json.loads(response.choices[0].message.content)
            return (structured_response["justification"],
                    structured_response["tone"],
                    structured_response["gossip"])

    def gossip(self, rules, seller, seller_action, buyer_action, seller_reward, buyer_reward, historical_messages):
        """
        Only buyers gossip when is_gossip=True.
        """
        if not self.is_gossip:
            raise RuntimeError("gossip() called but is_gossip=False")

        gossip_prompt_text = buyerGossipPrompt(
            horizon=self.horizon,
            use_equilibrium_knowledge=self.use_equilibrium_knowledge,
        ).substitute(
            buyer_name=self.name,
            seller_name=seller.name,
            seller_action=seller_action,
            buyer_action=buyer_action,
            seller_reward=seller_reward,
            buyer_reward=buyer_reward,
            historical_messages=historical_messages,
            stm=self.stm,
            discount_factor=self.cfg.experiment.env.discount_factor,
            horizon_length=self.horizon_length,
        ).strip()

        justification, tone, gossip_response = self.gossip_policy_llm(rules, gossip_prompt_text)
        return justification, tone, gossip_response

    def update_stm(self, round_idx, round_info):
        seller_name = round_info["seller_name"]
        seller_action = round_info["seller_action"]
        buyer_action = round_info["buyer_action"]
        buyer_reward = round_info["buyer_reward"]
        buyer_justification = round_info.get("buyer_justification", "")

        gossip_msg = round_info.get("gossip", "")
        gossip_justification = round_info.get("gossip_justification", "")

        extra = ""
        if gossip_msg:
            extra = f' After the round, I posted a public message about {seller_name}: "{gossip_msg}". This is my justification for the message: "{gossip_justification}".'

        round_context = f""" In round {round_idx}, I, {self.name}, was matched with seller {seller_name}. The seller chose {seller_action}. I chose {buyer_action}. My reward was {buyer_reward}. This is my justification: "{buyer_justification}".{extra}
        """
        self.stm.append(round_context)