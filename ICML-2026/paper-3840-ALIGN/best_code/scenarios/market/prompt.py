import textwrap
from string import Template


def rulePrompt(horizon: str, is_gossip: bool) -> Template:
    """
    Shared rules prompt for the Transaction Market.

    Parameters
    ----------
    is_gossip : bool
        If True, include a public messaging channel.

    horizon : {"infinite", "finite"}
        - infinite: rounds continue indefinitely.
        - finite  : the game ends after exactly $horizon_length rounds
    """

    self_awareness = textwrap.dedent(f"""\
        ## Self-awareness
        You are a self-interested, far-sighted, rational agent playing the {horizon}-horizon Product-choice Market game.
        - Self-interested: your utility is only your own cumulative payoff.
        - Far-sighted: maximize your expected discounted cumulative reward with discount factor $discount_factor.
        - Rational: choose the action with the highest expected payoff given your beliefs.
    """)

    if horizon == "infinite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - Infinite-horizon: rounds continue indefinitely.
        """)
    elif horizon == "finite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - Finite-horizon: exactly $horizon_length rounds.
            - After round $horizon_length, play stops.
        """)
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    overview = textwrap.dedent(f"""\
        ## Overview of the Game
        - There are multiple sellers and multiple buyers.
        - In each round, one seller and one buyer are matched for a single transaction.
        - Seller chooses product quality: H (high) or L (low). High quality costs more to produce.
        - Buyer chooses purchase type: c (customized), s (standardized), or none (refuse).
        - If buyer chooses none, no trade happens and both receive 0 reward for this round.
        - Otherwise, rewards follow the payoff rules below.
        - After payoffs, agents are reshuffled and matched again for the next round. The same pair never meets twice across all rounds.
    """) + "\n" + game_length

    payoffs = textwrap.dedent("""\
        ## Payoff Matrix (Seller payoff, Buyer payoff)
        ```text
                        Buyer chooses c (customized)             Buyer chooses s (standardized)
        Seller chooses H   ($seller_Hc_reward, $buyer_Hc_reward)     ($seller_Hs_reward, $buyer_Hs_reward)
        Seller chooses L   ($seller_Lc_reward, $buyer_Lc_reward)     ($seller_Ls_reward, $buyer_Ls_reward)

        Buyer chooses none: (0, 0)
        ```
    """)

    info_flow = ""
    if is_gossip:
        info_flow = textwrap.dedent("""\
            ## Information flow
            - After each round, the buyer will write a public message about the seller. All messages are publicly logged, visible to all agents throughout the game.
            - Messages do not directly change anyone's payoffs.
            - Future buyers can read past messages, which may influence their actions.
        """)

    parts = [self_awareness, overview, payoffs, info_flow]
    prompt_text = "\n\n".join(p.strip() for p in parts if p)
    return Template(prompt_text)



def sellerPrompt(horizon: str, is_gossip: bool, use_equilibrium_knowledge: bool) -> Template:
    identity = textwrap.dedent("""\
        ## Seller Instructions
        ### Identity and Observation
        Your name is $seller_name. In this round you are the seller and you are matched with buyer $buyer_name.
    """)

    if horizon == "infinite":
        objective_line = (
            "maximize your expected discounted cumulative reward over the remaining rounds "
            "of an infinite-horizon game (discount factor $discount_factor)."
        )
    elif horizon == "finite":
        objective_line = (
            "maximize your expected discounted cumulative reward over the remaining rounds "
            "of a finite-horizon game (total length $horizon_length, discount factor $discount_factor)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {objective_line}
    """)

    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    if is_gossip:
        community = textwrap.dedent("""\
            ### Community Messages
            You can review the public log of all buyers' messages:
            $historical_messages
        """)
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your action is not directly observed by other agents. However, the buyer will post a public message about you. This message is permanently logged and visible to **all** future buyers and may affect how others treat you later.
        """)
    else:
        community = ""
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your action is not directly observed by other agents and no message will be broadcast.
        """)


    equilibrium_knowledge = ""
    if use_equilibrium_knowledge:
        equilibrium_knowledge = textwrap.dedent("""\
            ### Common Knowledge for Finding Subgame Perfect Equilibria
            - **Finite-horizon games:** Use backward induction.
            - Start from the last round and determine the optimal actions there.
            - Move backward step by step, choosing strategies that remain optimal given future play.

            - **Infinite-horizon games:** Use the one-shot deviation principle.
            - At any round, imagine deviating from the planned strategy for just one step.
            - Ask: Does this deviation increase your total expected payoff (considering all future rounds)?
            - If yes, the original strategy is not an equilibrium.
            - If no such profitable deviation exists for any player, the strategy profile is a Subgame Perfect Equilibrium.
        """)

    action_rule = textwrap.dedent("""\
        ### Seller Action Rule
        Choose product quality:
        - H: high quality
        - L: low quality

        Your payoff depends on the buyer's choice and your choice:
        - If buyer chooses c and you choose H: your reward = $seller_Hc_reward
        - If buyer chooses s and you choose H: your reward = $seller_Hs_reward
        - If buyer chooses c and you choose L: your reward = $seller_Lc_reward
        - If buyer chooses s and you choose L: your reward = $seller_Ls_reward
        - If buyer chooses none: your reward = 0

        Use the payoff matrix from the rules prompt when deciding.
    """)

    if use_equilibrium_knowledge==True:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "brief reason in 1-3 sentences",
            "seller_action": "exactly one of 'H' or 'L'"
        }
        ```
    """)

    sections = [identity, objective, memory, community, accountability, equilibrium_knowledge, action_rule, response_instruction]
    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)



def buyerPrompt(horizon: str, is_gossip: bool, use_equilibrium_knowledge: bool) -> Template:
    identity = textwrap.dedent("""\
        ## Buyer Instructions
        ### Identity and Observation
        Your name is $buyer_name. In this round you are the buyer and you are matched with seller $seller_name.
    """)

    if horizon == "infinite":
        objective_line = (
            "maximize your expected discounted cumulative reward over the remaining rounds "
            "of an infinite-horizon game (discount factor $discount_factor)."
        )
    elif horizon == "finite":
        objective_line = (
            "maximize your expected discounted cumulative reward over the remaining rounds "
            "of a finite-horizon game (total length $horizon_length, discount factor $discount_factor)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {objective_line}
    """)


    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    community = ""
    if is_gossip:
        community = textwrap.dedent("""\
        ### Community Messages
        You can review the public log about earlier buyer messages, track the past behavior of your current seller to judge their trustworthiness:
        $historical_messages                                    
        """)


    equilibrium_knowledge = ""
    if use_equilibrium_knowledge:
        equilibrium_knowledge = textwrap.dedent("""\
            ### Common Knowledge for Finding Subgame Perfect Equilibria
            - **Finite-horizon games:** Use backward induction.
            - Start from the last round and determine the optimal actions there.
            - Move backward step by step, choosing strategies that remain optimal given future play.

            - **Infinite-horizon games:** Use the one-shot deviation principle.
            - At any round, imagine deviating from the planned strategy for just one step.
            - Ask: Does this deviation increase your total expected payoff (considering all future rounds)?
            - If yes, the original strategy is not an equilibrium.
            - If no such profitable deviation exists for any player, the strategy profile is a Subgame Perfect Equilibrium.
        """)

    action_rule = textwrap.dedent("""\
        ### Action Rule
        Choose purchase type:
        - c: customized
        - s: standardized
        - none: refuse to buy

        Your payoff depends on the seller's choice and your choice:
        - If seller chooses H and you choose c: your reward = $buyer_Hc_reward
        - If seller chooses H and you choose s: your reward = $buyer_Hs_reward
        - If seller chooses L and you choose c: your reward = $buyer_Lc_reward
        - If seller chooses L and you choose s: your reward = $buyer_Ls_reward
        - If you choose none: your reward = 0

        Use the payoff matrix from the rules prompt when deciding.
    """)

    if use_equilibrium_knowledge==True:
        response = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)
    else:
        response = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)

    response += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "brief reason in 1-3 sentences",
            "buyer_action": "exactly one of 'c' or 's' or 'none'"
        }
        ```
    """)

    sections = [identity, objective, memory, community, equilibrium_knowledge, action_rule, response]
    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)



def buyerGossipPrompt(horizon: str, use_equilibrium_knowledge: bool) -> Template:
    """
    Build the buyer-side gossip prompt for the Transaction Market.
    """

    # ---------------------------------------------------------
    # Identity and observation
    # ---------------------------------------------------------
    identity = textwrap.dedent("""\
        ## Buyer Instructions
        ### Identity and Observation
        Your name is `$buyer_name`. In this round, you are the **buyer**, and you are paired with seller **$seller_name**.
        The seller chose quality `$seller_action` (H=high, L=low). You chose purchase type `$buyer_action` (c=customized, s=standardized, none=refuse).
        This round's realized payoffs were: seller payoff = `$seller_reward`, your payoff = `$buyer_reward`.
    """)

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------
    if horizon == "infinite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of an *infinite-horizon* game (discount factor `$discount_factor`)."
        )
    elif horizon == "finite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of a finite-horizon game (total length `$horizon_length`, discount factor `$discount_factor`)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")
    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {goal_line}
    """)

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------
    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    # ---------------------------------------------------------
    # Community Messages and Accountability notice
    # ---------------------------------------------------------
    community_messages = textwrap.dedent("""\
        ### Community Messages
        You can review the public log about earlier buyer messages, track the past behavior of your current seller to judge their trustworthiness:
        $historical_messages
    """)
    accountability = textwrap.dedent("""\
        ### Accountability Notice
        The seller's action is not directly observed by other agents. However, you will publish a message about it.
        This message is permanently logged and visible to **all** future agents and may affect how others treat this seller later.
    """)

    # ---------------------------------------------------------
    # Common Knowledge of Solving Subgame Perfect Equilibria
    # ---------------------------------------------------------
    equilibrium_knowledge = textwrap.dedent("""\
        ### Common Knowledge for Finding Subgame Perfect Equilibria
        - **Finite-horizon games:** Use backward induction.
        - Start from the last round and determine the optimal actions there.
        - Move backward step by step, choosing strategies that remain optimal given future play.

        - **Infinite-horizon games:** Use the one-shot deviation principle.
        - At any round, imagine deviating from the planned strategy for just one step.
        - Ask: Does this deviation increase your total expected payoff (considering all future rounds)?
        - If yes, the original strategy is not an equilibrium.
        - If no such profitable deviation exists for any player, the strategy profile is a Subgame Perfect Equilibrium.
    """)

    # ---------------------------------------------------------
    # Gossip Rule and Response Format
    # ---------------------------------------------------------
    gossip_rule = textwrap.dedent("""\
        ### Gossip Rule
        Now, choose exactly one tone from the list [**praising**, **neutral**, **mocking**, **complaint**, **criticism**]
        and write a public message about the seller's choice using the selected tone.
        - **praising**: express approval and positive evaluation
        - **neutral**: provide an objective, matter-of-fact assessment
        - **mocking**: use sarcastic ridicule
        - **complaint**: convey mild disappointment
        - **criticism**: deliver a harsh negative evaluation
    """)

    if use_equilibrium_knowledge:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** to judge whether the seller is trustworthy.
            - Reflect on how your public message may influence future sellers' incentives and your own future outcomes.
            - Ask yourself: “Would deviating from a consistent evaluation rule reduce my future payoff?”
            - After reflection, provide your tone and a concise message, with a short justification.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** to judge whether the seller is trustworthy.
            - Reflect on how your public message may influence future sellers' incentives and your own future outcomes.
            - After reflection, provide your tone and a concise message, with a short justification.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "a short explanation of how your choice follows from your reasoning",
            "tone": "one of {'praising', 'neutral', 'mocking', 'complaint', 'criticism'}",
            "gossip": "a concise public message to the community (less than 150 words)"
        }
        ```
    """)

    # ---------------------------------------------------------
    # Assemble and return
    # ---------------------------------------------------------
    if use_equilibrium_knowledge:
        sections = [
            identity, objective, memory, community_messages, accountability,
            equilibrium_knowledge, gossip_rule, response_instruction
        ]
    else:
        sections = [
            identity, objective, memory, community_messages, accountability,
            gossip_rule, response_instruction
        ]

    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)


if __name__ == "__main__":
    for horizon in ["finite", "infinite"]:
        for is_gossip in [True, False]:
            for use_equilibrium_knowledge in [True, False]:
                print("===")
                print(f"horizon={horizon}, is_gossip={is_gossip}, use_equilibrium_knowledge={use_equilibrium_knowledge}")
                print("--- Seller Prompt ---")
                seller_prompt = sellerPrompt(horizon, is_gossip, use_equilibrium_knowledge)
                print(seller_prompt.template)
                print("--- Buyer Prompt ---")
                buyer_prompt = buyerPrompt(horizon, is_gossip, use_equilibrium_knowledge)
                print(buyer_prompt.template)
                if is_gossip:
                    print("--- Buyer Gossip Prompt ---")
                    buyer_gossip_prompt = buyerGossipPrompt(horizon, use_equilibrium_knowledge)
                    print(buyer_gossip_prompt.template)