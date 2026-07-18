import os
import textwrap
from string import Template

def rulePrompt(horizon: str, is_gossip: bool) -> Template:
    """
    Build a rules prompt for the (traditional) trust game.

    Parameters
    ----------
    horizon : {"infinite", "finite"}
        - infinite: rounds continue indefinitely.
        - finite  : the game ends after exactly $horizon_length rounds

    is_gossip : bool
        If True, include the information-flow (messaging) section.
    """

    # ---------------- Self-awareness ----------------
    self_awareness = textwrap.dedent(f"""\
        ## Self-awareness
        You are a **self-interested, far-sighted, rational** agent, you are playing the {horizon}-horizon multi-round **trust game**. 
        - **Self-interested**: your utility function is **only** your own expected payoff.
            - You may track, predict, or even raise other agents' payoffs—but **only** when doing so increases your own expected return.
            - Other agents' welfare carries no intrinsic weight; it matters solely through its effect on your future rewards.
        - **Far-sighted**: your goal is to maximize your **expected discounted cumulative reward** (discount factor `$discount_factor`) over all remaining rounds until the game ends, not just the immediate reward.
        - **Rational**: you always choose the strategy with the highest expected payoff.
    """)

    # ---------------- Game length ----------------
    if horizon == "infinite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - This is an *infinite-horizon* trust game.
            - Because rounds continue indefinitely without termination, the expected horizon is infinite.
        """)
    elif horizon == "finite":
        game_length = textwrap.dedent("""\
            ### Game Length
            - This is a *finite-horizon* trust game of exactly **$horizon_length** rounds.
            - After round $horizon_length, play stops; no future rewards are possible.
            - Because termination is fixed and independent of history, the expected horizon is exactly $horizon_length.
        """)
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    # ---------------- Gossip behaviour (symmetric) ----------------
    if is_gossip:
        postround_investor_action = (
            "observe the responder's return decision and then broadcast a public message about the responder's behavior this round"
        )
        postround_responder_action = (
            "observe the investor's investment decision and then broadcast a public message about the investor's behavior this round"
        )
        gossip_summary = (
            "Both agents send one public message per round (one from the investor and one from the responder). "
            "These two messages are permanently logged and visible to all agents."
        )
    else:
        postround_investor_action = (
            "observe the responder's return decision; no public message is sent"
        )
        postround_responder_action = (
            "observe the investor's investment decision; no public message is sent"
        )
        gossip_summary = (
            "No public gossip is allowed; agents only privately observe each other's actions."
        )

    # ---------------- Game overview (standard trust game) ----------------
    game_overview = textwrap.dedent(f"""\
        ## Overview of the Game
        You are playing a multi-round **trust game** with {horizon} horizon.

        - Every agent starts with **$initial_resources** resource units.
        - In each round, two agents are randomly paired without replacement as:
          - **Investor** (first mover)
          - **Responder** (second mover)
          The same pair never meets twice across all rounds.

        - Stage game per round:
          1. The investor observes their current resources and the responder's current resources.
          2. The investor chooses an **investment** amount `I` between **[0, current_resources]**.
          3. The investment `I` is multiplied by a factor `$investment_multiplier` and transferred to the responder.
          4. The responder chooses a **return** amount `R` between **[0, I * $investment_multiplier]** to send back to the investor.
          5. Both players' payoffs for the round are realized.
          6. Both players observe each other's actions in this round.
          7. The investor will {postround_investor_action}.
          8. The responder will {postround_responder_action}.

        - {gossip_summary}

        - After payoffs (and any messages) are processed, agents are reshuffled and roles may switch in later rounds 
          (an agent who was an investor in one round may be a responder in a later round, and vice versa).
    """) + "\n" + game_length

    # ---------------- Reward structure ----------------
    reward = textwrap.dedent("""\
        ## Immediate Reward Per Round (Standard Trust Game)
        Let `I` be the amount the investor chooses to invest,
        and let `R` be the amount the responder chooses to return.

        - The investment `I` is multiplied by `$investment_multiplier` and added to the responder's resources.
        - The responder then chooses a return amount `R` between **[0, I * $investment_multiplier]**.

        - **Investor reward** this round:
          - They lose the invested amount `I`, but receive `R` back.
          - Net payoff change from this round: **`-I + R`**.

        - **Responder reward** this round:
          - They gain the multiplied investment `I * $investment_multiplier`, but give back `R`.
          - Net payoff change from this round: **`I * $investment_multiplier - R`**.
    """)

    # ---------------- Information flow (gossip) ----------------
    info_flow = ""
    if is_gossip:
        info_flow = textwrap.dedent("""\
            ## Information flow and Gossip
            - At the end of each round, **after** both actions are taken and payoffs are realized:
              - The investor observes the responder's return decision.
              - The responder observes the investor's investment decision.
            - Then **each agent** can broadcast **one public message** about their coplayer's behavior in that round.
            - Thus, each round generates **two messages**:
              - one from the investor about the responder,
              - one from the responder about the investor.
            - All messages are publicly logged, visible to all agents throughout the game.
            - Messages do not directly change anyone's payoffs.
            - Future agents can read past messages, which may influence their future actions.
        """)

    parts = [self_awareness, game_overview, reward, info_flow]
    prompt_text = "\n\n".join(p.strip() for p in parts if p)

    return Template(prompt_text)


def investorPrompt(
    horizon: str,
    is_gossip: bool,
    use_equilibrium_knowledge: bool
) -> Template:
    """
    Build the investor-instruction prompt for a single round of the (standard) trust game.
    """

    # ---------------- Identity and observation ----------------
    identity = textwrap.dedent("""\
        ## Investor Instructions
        ### Identity and Observation
        Your name is `$investor_name`. In this round, you are the **investor** (first mover), and you are paired with responder **$responder_name**.
        You currently hold `$investor_resources` resource units; the matched responder **$responder_name** holds `$responder_resources`.
        As the investor, you will choose how much to **trust** the responder by investing part of your resources.
    """)

    # ---------------- Objective ----------------
    if horizon == "infinite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of an *infinite-horizon* trust game (discount factor `$discount_factor`)."
        )
    elif horizon == "finite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of a finite-horizon trust game (total length `$horizon_length`, discount factor `$discount_factor`)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {goal_line}
    """)

    # ---------------- Memory ----------------
    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    # ---------------- Community Messages & Accountability ----------------
    if is_gossip:
        community_messages = textwrap.dedent("""\
            ### Community Messages
            You can review the public log about earlier **trust-game gossip messages**, tracking the past behavior of your current responder to judge their trustworthiness:
            $historical_messages
        """)
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            At the end of each round, after you and your coplayer observe each other's actions, **both of you** can broadcast a public message about your coplayer's behavior.
            - Your coplayer (the responder) may broadcast a message about **you**.
            - You may also broadcast a message about **them** when asked.
            All such messages are permanently logged and visible to **all** future agents and may affect how others treat you later.
        """)
    else:
        community_messages = ""
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your investment decision and the responder's return are not publicly discussed; no gossip messages will be broadcast about this round.
        """)

    # ---------------- Subgame Perfect Equilibrium knowledge ----------------
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

    # ---------------- Trust Action Rule + Response ----------------
    trust_rule = textwrap.dedent("""\
        ### Trust Action Rule
        Now, choose your **investment** action as the investor for this round.

        You may invest any real amount `I` between **[0, $investor_resources]**
    """)

    if use_equilibrium_knowledge:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, judging if the responder is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibrium reasoning in the trust game.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, judging if the responder is trustworthy to avoid being exploited.
            - Reflect on what action will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it is strategically sensible in the trust game.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
          "justification": "a short explanation of how your choice follows from reflecting on strategic (and, if applicable, Subgame Perfect Equilibrium) reasoning in the trust game",
          "investor_action": "a real number between 0 and $investor_resources representing how much you invest"
        }
        ```
    """)

    # ---------------- Assemble ----------------
    if use_equilibrium_knowledge:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            equilibrium_knowledge, trust_rule, response_instruction
        ]
    else:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            trust_rule, response_instruction
        ]

    full_prompt = "\n\n".join(s.strip() for s in sections if s)
    return Template(full_prompt)


def responderPrompt(
    horizon: str,
    is_gossip: bool,
    use_equilibrium_knowledge: bool
) -> Template:
    """
    Build the responder-instruction prompt for choosing the return amount
    in a single round of the (standard) trust game.
    """

    # ---------------- Identity and observation ----------------
    identity = textwrap.dedent("""\
        ## Responder Instructions
        ### Identity and Observation
        Your name is `$responder_name`. In this round, you are the **responder** (second mover), and you are paired with investor **$investor_name**.
        Before the investment, you held `$responder_resources` resource units; the matched investor **$investor_name** held `$investor_resources`.
        The investor invested `$investment` (`$investment_ratio` of the investor's current stock), which was multiplied to `$benefit` units and transferred to you.
        You now choose how much to **return** to the investor in this round.
    """)

    # ---------------- Objective ----------------
    if horizon == "infinite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of an *infinite-horizon* trust game (discount factor `$discount_factor`)."
        )
    elif horizon == "finite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of a finite-horizon trust game (total length `$horizon_length`, discount factor `$discount_factor`)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {goal_line}
    """)

    # ---------------- Memory ----------------
    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    # ---------------- Community Messages & Accountability ----------------
    if is_gossip:
        community_messages = textwrap.dedent("""\
            ### Community Messages
            You can review the public log about earlier gossip in the **trust game**, tracking the past behavior of your current investor to judge their trustworthiness:
            $historical_messages
        """)
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            At the end of each round, after you and the investor observe each other's actions, **both of you** can broadcast a public message about your coplayer's behavior.
            - The investor may broadcast a message about **you**.
            - You may also broadcast a message about **them** when asked.
            These messages are permanently logged and visible to **all** future agents and may affect how others treat you later.
            When you choose how much to return now, you may anticipate the effect of future gossip on your long-run payoff.
        """)
    else:
        community_messages = ""
        accountability = textwrap.dedent("""\
            ### Accountability Notice
            Your return decision and the investor's investment are not publicly discussed; no gossip messages will be broadcast about this round.
        """)

    # ---------------- Subgame Perfect Equilibrium knowledge ----------------
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

    # ---------------- Return Action Rule + Response ----------------
    return_rule = textwrap.dedent("""\
        ### Return Action Rule
        Now, choose your **return amount** as the responder for this round.

        The investor's investment was multiplied to `$benefit` units and added to your resources.
        You may return any real amount `R` between [0, $benefit]**
    """)

    if use_equilibrium_knowledge:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory**, and (if available) **Community Messages** provided above, judging how your return choice today affects:
              - your immediate payoff, and
              - others' future treatment of you (especially under gossip).
            - Reflect on what return amount will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibrium reasoning in the trust game.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory**, and (if available) **Community Messages** provided above, judging how your return choice today affects:
              - your immediate payoff, and
              - others' future treatment of you (especially under gossip).
            - Reflect on what return amount will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it is strategically sensible in the trust game.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
          "justification": "a short explanation of how your choice follows from reflecting on strategic (and, if applicable, Subgame Perfect Equilibrium) reasoning in the trust game",
          "responder_action": "a real number between 0 and $benefit representing how much you return to the investor"
        }
        ```
    """)

    # ---------------- Assemble ----------------
    if use_equilibrium_knowledge:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            equilibrium_knowledge, return_rule, response_instruction
        ]
    else:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            return_rule, response_instruction
        ]

    full_prompt = "\n\n".join(s.strip() for s in sections if s)
    return Template(full_prompt)


def investorGossipPrompt(horizon: str, use_equilibrium_knowledge: bool) -> Template:
    """
    Build the investor-side gossip prompt for the trust game.

    The investor has already observed the responder's return and now gossips
    about the responder's behavior in this round.
    """

    # ---------------- Identity and observation ----------------
    identity = textwrap.dedent("""\
        ## Investor Gossip Instructions
        ### Identity and Observation
        Your name is `$investor_name`. In this round, you were the **investor**, and you were paired with responder **$responder_name**.

        - You invested `$investment` units (this equals `$investment_ratio` of your current stock).
        - This investment was multiplied into `$benefit` units and transferred to the responder.
        - The responder returned `$returned_amount` units to you (this equals `$returned_ratio` of the transferred benefit `$benefit`).

        You have fully observed:
        - how much you invested,
        - the multiplied amount you transferred,
        - and the responder's **actual return decision** in this round.
    """)

    # ---------------- Objective ----------------
    if horizon == "infinite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of an *infinite-horizon* trust game (discount factor `$discount_factor`)."
        )
    elif horizon == "finite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of a finite-horizon trust game (total length `$horizon_length`, discount factor `$discount_factor`)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {goal_line}
    """)

    # ---------------- Memory ----------------
    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    # ---------------- Community Messages & Accountability ----------------
    community_messages = textwrap.dedent("""\
        ### Community Messages
        You can review the public log about earlier gossip in the **trust game**, tracking the past behavior of your current responder and other agents:
        $historical_messages
    """)

    accountability = textwrap.dedent("""\
        ### Accountability Notice
        At the end of each round, after you and the responder observe each other's actions, **both of you** can broadcast a public message about your coplayer's behavior.
        - The responder can broadcast a message about **you**.
        - You will now broadcast a message about **them**.
        Your message is permanently logged and visible to **all** future agents and may affect how others treat both you and your coplayer.
    """)

    # ---------------- Subgame Perfect Equilibrium knowledge ----------------
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

    # ---------------- Gossip rule & response ----------------
    gossip_rule = textwrap.dedent("""\
        ### Gossip Rule (Investor)
        You have already **observed the responder's action** in this trust game round:

        - your own investment `$investment` (ratio `$investment_ratio` of your stock),
        - the multiplied benefit `$benefit` transferred to the responder,
        - the responder's returned amount `$returned_amount` (ratio `$returned_ratio` of `$benefit`).

        Now, choose exactly one tone from the list  
        [**praising**, **neutral**, **mocking**, **complaint**, **criticism**]  
        and write a public message about the responder's behavior using the selected tone.
        - **praising**: express gratitude and positive evaluation
        - **neutral**: provide an objective, matter-of-fact assessment
        - **mocking**: use sarcastic ridicule
        - **complaint**: convey mild disappointment
        - **criticism**: deliver a harsh negative evaluation
    """)

    if use_equilibrium_knowledge:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, judging how your gossip may influence:
              - others' beliefs about this responder (given `$investment`, `$benefit`, `$returned_amount`, `$returned_ratio`),
              - and your own future payoffs (through others' responses to reputations).
            - Reflect on what gossip tone and content will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibrium reasoning.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, especially the specific values of `$investment`, `$benefit`, `$returned_amount`, and `$returned_ratio`.
            - Judge how generous or exploitative the responder's behavior was and how your gossip may influence future play.
            - Reflect on what gossip tone and content will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it is strategically sensible in the trust game.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "a short explanation of how your choice follows from reflecting on strategic reasoning (and, if applicable, Subgame Perfect Equilibrium reasoning) in the trust game",
            "tone": "one of {'praising', 'neutral', 'mocking', 'complaint', 'criticism'}",
            "gossip": "a concise public message to the community (less than 150 words)"
        }
        ```
    """)

    # ---------------- Assemble ----------------
    if use_equilibrium_knowledge:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            equilibrium_knowledge, gossip_rule, response_instruction
        ]
    else:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            gossip_rule, response_instruction
        ]

    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)


def responderGossipPrompt(horizon: str, use_equilibrium_knowledge: bool) -> Template:
    """
    Build the responder-side gossip prompt for the trust game.

    The responder has already observed the investor's investment and their own return,
    and now gossips about the investor's behavior in this round.
    """

    # ---------------- Identity and observation ----------------
    identity = textwrap.dedent("""\
        ## Responder Gossip Instructions
        ### Identity and Observation
        Your name is `$responder_name`. In this round, you were the **responder** (second mover), and you were paired with investor **$investor_name**.

        - Before the investment, you held `$responder_resources` resource units;
          the matched investor **$investor_name** held `$investor_resources`.
        - The investor invested `$investment` units (this equals `$investment_ratio` of the investor's current stock).
        - This investment was multiplied into `$benefit` units and transferred to you.
        - You chose to return `$returned_amount` units to the investor
          (this equals `$returned_ratio` of the transferred benefit `$benefit`).

        You have fully observed:
        - the investor's investment decision,
        - the multiplied amount you received,
        - and your own actual return decision in this round.
    """)

    # ---------------- Objective ----------------
    if horizon == "infinite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of an *infinite-horizon* trust game (discount factor `$discount_factor`)."
        )
    elif horizon == "finite":
        goal_line = (
            "maximize your **expected discounted cumulative reward** over the remaining rounds of a finite-horizon trust game (total length `$horizon_length`, discount factor `$discount_factor`)."
        )
    else:
        raise ValueError("horizon must be 'infinite' or 'finite'")

    objective = textwrap.dedent(f"""\
        ### Objective
        Your sole objective is to {goal_line}
    """)

    # ---------------- Memory ----------------
    memory = textwrap.dedent("""\
        ### Memory
        You can recall your interaction history of past rounds. Here are the historical records:
        $stm
    """)

    # ---------------- Community Messages & Accountability ----------------
    community_messages = textwrap.dedent("""\
        ### Community Messages
        You can review the public log about earlier gossip in the **trust game**, tracking the past behavior of your current investor and other agents:
        $historical_messages
    """)

    accountability = textwrap.dedent("""\
        ### Accountability Notice
        At the end of each round, after you and the investor observe each other's actions, **both of you** can broadcast a public message about your coplayer's behavior.
        - The investor may broadcast a message about **you**.
        - You will now broadcast a message about **them**.
        Your message is permanently logged and visible to **all** future agents and may affect how others treat both you and your coplayer.
    """)

    # ---------------- Subgame Perfect Equilibrium knowledge ----------------
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

    # ---------------- Gossip rule & response ----------------
    gossip_rule = textwrap.dedent("""\
        ### Gossip Rule (Responder)
        You have already **observed your coplayer's action** in this trust game round:

        - the investor's investment `$investment` (ratio `$investment_ratio` of their stock),
        - the multiplied benefit `$benefit` you received,
        - your own returned amount `$returned_amount` (ratio `$returned_ratio` of `$benefit`).

        Now, choose exactly one tone from the list  
        [**praising**, **neutral**, **mocking**, **complaint**, **criticism**]  
        and write a public message about the investor's behavior using the selected tone.
        - **praising**: express gratitude and positive evaluation
        - **neutral**: provide an objective, matter-of-fact assessment
        - **mocking**: use sarcastic ridicule
        - **complaint**: convey mild disappointment
        - **criticism**: deliver a harsh negative evaluation
    """)

    if use_equilibrium_knowledge:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using the **Common Knowledge for Finding Subgame Perfect Equilibria** provided above.
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, judging how your gossip may influence:
              - others' beliefs about this investor (given `$investment`, `$benefit`, `$returned_amount`, `$returned_ratio`),
              - and your own future payoffs (through others' responses to reputations).
            - Reflect on what gossip tone and content will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it aligns with Subgame Perfect Equilibrium reasoning.
        """)
    else:
        response_instruction = textwrap.dedent("""\
            ### Response Guidelines
            - Reflect using **Identity and Observation**, **Memory**, and **Community Messages** provided above, especially the specific values of `$investment`, `$benefit`, `$returned_amount`, and `$returned_ratio`.
            - Judge how generous or exploitative the investor's behavior was and how your gossip may influence future play.
            - Reflect on what gossip tone and content will maximize your objective.
            - Ask yourself: “Would deviating at this step improve my total expected payoff?”
            - After reflection, provide your action and a short explanation of why it is strategically sensible in the trust game.
        """)

    response_instruction += textwrap.dedent("""\
        **Return JSON ONLY in this exact format**
        ```json
        {
            "justification": "a short explanation of how your choice follows from reflecting on strategic reasoning (and, if applicable, Subgame Perfect Equilibrium reasoning) in the trust game",
            "tone": "one of {'praising', 'neutral', 'mocking', 'complaint', 'criticism'}",
            "gossip": "a concise public message to the community (less than 150 words)"
        }
        ```
    """)

    # ---------------- Assemble ----------------
    if use_equilibrium_knowledge:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            equilibrium_knowledge, gossip_rule, response_instruction
        ]
    else:
        sections = [
            identity, objective, memory,
            community_messages, accountability,
            gossip_rule, response_instruction
        ]

    prompt_text = "\n\n".join(s.strip() for s in sections if s)
    return Template(prompt_text)