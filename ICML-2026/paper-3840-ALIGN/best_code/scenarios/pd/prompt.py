import os
import textwrap
from string import Template


def rulePrompt(horizon: str, is_gossip: bool) -> Template:
    """
    Build a rules prompt for the prisoner's dilemma game.

    Parameters: 
    is_gossip : bool
        If True, include the information-flow (messaging) section.

    horizon : {"infinite", "finite"}
        - infinite: rounds continue indefinitely.
        - finite  : the game ends after exactly $horizon_length rounds
    """

    # Self-awareness
    self_awareness = textwrap.dedent(f"""\
        ## Self-awareness
        You are a **self-interested, far-sighted, rational** agent, you are playing the {horizon}-horizon multi-round prisoner's dilemma game. 
        - **Self-interested**: your utility function is **only** your own expected payoff.
            - You may track, predict, or even raise other agents' payoffs—but **only** when doing so increases your own expected return.
            - Other agents' welfare carries no intrinsic weight; it matters solely through its effect on your future rewards.
        - **Far-sighted**: your goal is to maximize your **expected discounted cumulative reward** (discount factor `$discount_factor`) over all remaining rounds until the game ends, not just the immediate reward."
        - **Rational**: you always choose the strategy with the highest expected payoff.
    """)

    # Game overview (by horizon, gossip)
    player_action = (
            "either to **cooperate** (pay fitness cost `$cost`) "
            "or to **defect** (pay no cost)"
        )
    
    if is_gossip:
        player_second_action = "broadcast a public assessment of the opponent's choice; this message is permanently logged and visible to all agents."
    else:
        player_second_action = "do nothing (no message will be broadcast)"

    if horizon == "infinite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - This is an *infinite-horizon* prisoner's dilemma game.
            - Because rounds continue indefinitely without termination, the expected horizon is infinite.
        """)
    elif horizon == "finite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - This is a *finite-horizon* prisoner's dilemma game of exactly **$horizon_length** rounds.
            - After round $horizon_length, play stops; no future rewards are possible.
            - Because termination is fixed and independent of history, the expected horizon is exactly $horizon_length.""")
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    game_overview = textwrap.dedent(f"""\
        ## Overview of the Game
        You are playing a multi-round prisoner's dilemma game with {horizon} horizon.
        - In each round, two agents are randomly paired without replacement; the same pair never meets twice across all rounds.
        - Each player will choose {player_action}.
        - After both players act, each player will observe the opponent's action and then {player_second_action}.
        - After payoffs, agents are reshuffled and paired with new opponents in the next round.
    """) + "\n" + game_length

    
    # Reward structure
    reward = textwrap.dedent("""\
        ## Immediate Reward Per Round
        - If you **cooperate** and opponent **cooperates**: Your reward = **$benefit - $cost**, Opponent reward = **$benefit - $cost**
        - If you **cooperate** and opponent **defects**: Your reward = **-$cost**, Opponent reward = **$benefit**
        - If you **defect** and opponent **cooperates**: Your reward = **$benefit**, Opponent reward = **-$cost**
        - If you **defect** and opponent **defects**: Your reward = **0**, Opponent reward = **0**
        
        Note: The benefit is larger than the cost ($benefit > $cost), so mutual cooperation yields the highest total payoff.
    """)

    # Information flow (if gossip)
    info_flow = ""
    if is_gossip:
        info_flow = textwrap.dedent("""\
            ## Information flow
            - All messages are publicly logged, visible to all agents throughout the game.
            - Messages do not directly change anyone's payoffs.
            - Future agents can read past messages, which may influence their actions.
        """)

    # Integrate all parts into the final prompt
    parts = [self_awareness, game_overview, reward, info_flow]
    prompt_text = "\n\n".join(p.strip() for p in parts if p)

    return Template(prompt_text)


def actionPrompt(horizon: str, is_gossip: bool, use_equilibrium_knowledge: bool
) -> Template:
    """
    Build the action prompt for a single round of prisoner's dilemma.
    """

    # ---------------------------------------------------------
    # Identity and observation
    # ---------------------------------------------------------
    identity = textwrap.dedent("""\
        ## Player Instructions - Action Phase
        ### Identity and Observation
        Your name is `$player_name`. In this round, you are paired with opponent **$opponent_name**.
        
        This is the **action phase** where you choose to cooperate or defect.
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
    if is_gossip:
        community_messages = textwrap.dedent("""\
            ### Community Messages
            You can review the public log about earlier prisoner's dilemma broadcasts, track the past behavior of your current opponent to judge their trustworthiness:
            $historical_messages
        """)
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your action is not directly observed by other agents. However, your opponent will publish a message about it in the gossip phase. This message is permanently logged and visible to **all** future agents and may affect how others treat you later.
        """)
    else:
        community_messages = ""
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your action is not directly observed by other agents and no message will be broadcast.
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
    # Prisoner's Dilemma Rule and Response Format
    # ---------------------------------------------------------
    pd_rule = textwrap.dedent("""\
        ### Prisoner's Dilemma Rule
        Now, choose your action for this round.
    """)

    if use_equilibrium_knowledge==True:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above, judging if the opponent is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above, judging if the opponent is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)

    pd_rule += textwrap.dedent("""\
        You may choose either  
        - **C**: cooperate (pay fitness cost `$cost`)
        - **D**: defect (pay no cost)
        
        Your payoff depends on both your choice and your opponent's choice:
        - If both cooperate: you get `$benefit - $cost`
        - If you cooperate, opponent defects: you get `-$cost`
        - If you defect, opponent cooperates: you get `$benefit`
        - If both defect: you get `0`
    """)
    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "a short explanation of how your choice follows from reflecting on Subgame Perfect Equilibrium reasoning",
            "player_action": "exactly one of 'C' or 'D'"
        }
        ```
    """)

    # ----------------------------------------------------------
    # Assemble sections
    # ----------------------------------------------------------
    if use_equilibrium_knowledge==True:
        sections = [identity, objective, memory, community_messages, accountability, equilibrium_knowledge, pd_rule, response_instruction]
    else:
        sections = [identity, objective, memory, community_messages, accountability, pd_rule, response_instruction]
    full_prompt = "\n\n".join(s.strip() for s in sections if s)

    return Template(full_prompt)


def gossipPrompt(horizon: str, use_equilibrium_knowledge: bool) -> Template:
    """
    Build the gossip prompt for prisoner's dilemma.
    """

    # ---------------------------------------------------------
    # Identity and observation
    # ---------------------------------------------------------
    identity = textwrap.dedent("""\
        ## Player Instructions - Gossip Phase
        ### Identity and Observation
        Your name is `$player_name`. In this round, you are paired with opponent **$opponent_name**.
        Your opponent chose to **$opponent_action** (cooperate or defect) in this round.
        
        This is the **gossip phase** where you choose how to publicly assess your opponent's action.
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
        You can review the public log about earlier prisoner's dilemma broadcasts, track the past behavior of your current opponent to judge their trustworthiness:
        $historical_messages
    """)
    accountability = textwrap.dedent("""\
        ### Accountability Notice
        Your opponent's action is not directly observed by other agents. However, you will publish a message about it. This message is permanently logged and visible to **all** future agents and may affect how others treat your opponent later.
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
        Now, choose exactly one tone from the list [**praising**, **neutral**, **mocking**, **complaint**, **criticism**] and write a public message about your opponent's action using the selected tone.
        - **praising**: express gratitude and positive evaluation
        - **neutral**: provide an objective, matter-of-fact assessment
        - **mocking**: use sarcastic ridicule
        - **complaint**: convey mild disappointment
        - **criticism**: deliver a harsh negative evaluation
    """)

    if use_equilibrium_knowledge==True:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above, judging if the opponent is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory** and **Community Messages** provided above, judging if the opponent is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibria reasoning.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "a short explanation of how your choice follows from reflecting on Subgame
            Perfect Equilibrium reasoning",
            "tone": "one of {'praising', 'neutral', 'mocking', 'complaint', 'criticism'}",
            "gossip": "a concise public message to the community (less than 150 words)"
        }
        ```
    """)

    # ---------------------------------------------------------
    # Assemble and return
    # ---------------------------------------------------------
    if use_equilibrium_knowledge==True:
        sections = [identity, objective, memory, community_messages, accountability, equilibrium_knowledge, gossip_rule, response_instruction]
    else:
        sections = [identity, objective, memory, community_messages, accountability, gossip_rule, response_instruction]
    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)


if __name__ == "__main__":
    for is_gossip in [True, False]:
        for horizon in ["infinite", "finite"]:
                directory = f"prompt_templates/DonorGame/horizon{horizon}_gossip_{is_gossip}"
                os.makedirs(directory, exist_ok=True)
                with open(f"{directory}/rulePrompt.txt", "w") as f:
                    f.write(rulePrompt(horizon, is_gossip).safe_substitute().strip())
                with open(f"{directory}/actionPrompt.txt", "w") as f:
                    f.write(actionPrompt(horizon, is_gossip).safe_substitute().strip())
                with open(f"{directory}/gossipPrompt.txt", "w") as f:
                    f.write(gossipPrompt(horizon).safe_substitute().strip())
    print("Prompt templates saved successfully.")