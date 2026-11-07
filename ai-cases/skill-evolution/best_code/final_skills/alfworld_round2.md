# ALFWorld Embodied Agent Skill

## Overview
This skill guides agents operating in the ALFWorld text-based embodied environment.
The agent must complete household tasks by navigating rooms, interacting with objects,
and using appliances. Actions must be chosen from the admissible action list provided
at each step.

**Output format**: Always output `<think>...</think>` for reasoning, then `<action>...</action>` for the chosen action.

---

## Task Types

| Type | Goal | Key Steps |
|------|------|-----------|
| Pick & Place | Put object X in/on receptacle Y | Find X -> take X -> go to Y -> put X in/on Y |
| Pick Two & Place | Put two instances of X in/on Y | Find X1 -> take -> place -> find X2 -> take -> place |
| Examine in Light | Examine object X under desklamp | Find X -> take X -> find desklamp -> use desklamp |
| Clean & Place | Clean object X and put in/on Y | Find X -> take X -> go to sink -> clean X -> go to Y -> put X |
| Heat & Place | Heat object X and put in/on Y | Find X -> take X -> go to microwave -> heat X -> go to Y -> put X |
| Cool & Place | Cool object X and put in/on Y | Find X -> take X -> go to fridge -> cool X -> go to Y -> put X |

---

## General Principles

1. **Decompose the task**: Parse the goal into ordered sub-goals (locate, acquire, transform, deliver). Complete each before moving to the next.
2. **Systematic exploration**: Check exposed surfaces first (countertops, tables, shelves, desktops, side tables) since target objects are often sitting in plain sight. Then open closed containers (drawers, cabinets, fridge) before judging them empty. Search each surface/container once before revisiting.
3. **Grab immediately**: When a required object is visible and reachable, take it right away before moving elsewhere.

4. **Treat each numbered furniture item as a separate search location.** Rooms contain several instances of the same furniture type (countertop 1, 2, 3; shelf 1..n; cabinet 1..n). Finding nothing on countertop 1 says nothing about countertop 2 or 3. Make one ordered pass that visits every numbered variant at least once (e.g., all countertops, then shelves, then closed containers). In practice, missing targets often sit on the one surface variant you have not yet walked to. When you notice yourself revisiting an already-opened cabinet or fridge, break the loop by going to the next numbered surface you have never observed.
4. **Transform before placing**: If the task requires cleaning, heating, or cooling, go to the appropriate appliance while holding the object and apply the transform action directly (e.g. `clean knife 1 with sinkbasin 1`, `heat potato 3 with microwave 1`, `cool bread 1 with fridge 1`). Do not first move the object inside the microwave/fridge/sink or toggle its door; if you accidentally staged it inside, take it back out and transform the held object.
5. **Direct delivery**: Once holding the transformed (or untransformed) goal object, navigate straight to the target receptacle and place it.
6. **Track progress**: Maintain an internal count of how many objects still need to be found and placed. Only stop searching when the count reaches zero.

- For tasks phrased 'some X' or 'two X', multiple copies are often stacked on the same surface or in the same container (e.g., both newspapers on the coffee table). When you see two copies but can carry only one, remember the exact location of the remaining copy; after delivering the first, navigate straight back there to grab it rather than launching a fresh search of containers you already know are empty.
7. **Avoid loops**: Never repeat the same action more than twice in a row. If stuck, move to a different unexplored location.
8. **Only choose admissible actions**: Always pick an action from the admissible action list. Do not invent actions.

---

## Common Mistakes to Avoid

- **Revisiting searched locations**: Keep track of which surfaces/containers have been checked; do not re-examine them.
- **Ignoring visible objects**: If the target object appears in the observation, pick it up immediately.
- **Skipping state changes**: Do not place an object at the destination without first cleaning/heating/cooling it when required—even if the object was originally found inside/on the destination receptacle. Retrieve it, transform it, then place it back.
- **Premature termination**: Do not stop the episode until all goal conditions are verified as met.
- **Action loops**: Repeatedly toggling or examining the same object wastes steps. Move on to new locations instead.

<!-- APPENDIX_START -->
## Execution Notes Appendix
- The agent explicitly reasoned that no cooling was needed for a 'put a clean lettuce in fridge' task, reinforcing that transformation type must match the task verb, not the destination.
- Transform-before-place tasks consistently succeeded by applying heat/clean directly to the held object at the closed appliance (microwave, sink) without opening the door or staging the object; existing rule 4 should be retained.
- The existing guidance 'transform before placing' is strongly confirmed: do not stage objects inside appliances or toggle doors before cleaning/heating/cooling them.
- Once a cabinet is opened and shows nothing, or a countertop has been examined, treat it as searched; do not call it 'unexplored' again or return to it until every other reachable receptacle has been checked once.
- When you identify drawers, shelves, or tables as remaining search locations, actually navigate to and open them during a first-pass sweep before revisiting any already-searched cabinet.
- Strictly apply the existing no-revisit rule: once a location is opened/examined and shows nothing, it is searched forever; do not select it again until every location name that has never appeared in an observation has been visited once.
- A loop need not repeat the same action twice in a row: alternating cabinet 1, cabinet 2, fridge, and countertop is still a loop. If your recent navigation targets have all been known-empty locations, stop and move to a never-visited location rather than the next cabinet in the same sequence.
<!-- APPENDIX_END -->

<!-- SLOW_UPDATE_START -->
PROTECTED SLOW UPDATE — BINDING SEARCH RULES

The transformation steps are working; object search is what fails. Whenever the target object is not already in your hand, follow these rules:

1. Search exhaustively, not by plausibility. Never pick a next location because lettuce belongs in the fridge, a pan belongs in a cabinet, or a dishsponge belongs near the sink. Those guesses repeatedly cause failures. Instead, build a checklist from the admissible actions: every exposed surface you have not yet observed and every closed container/drawer/fridge you have not yet opened. Visit each entry once before visiting any entry a second time.

2. Treat the final delivery receptacle as an ordinary search location. For clean/heat/cool-and-place tasks, the target object can already be sitting on or inside the destination receptacle. Go to the destination early in the first sweep, not only at delivery time. If you find the target there, take it, transform it at the required appliance, then place it back in/on the destination. Never terminate or omit the transformation just because the object was already at the destination.

3. Maintain a seen-empty list. The moment you observe the contents of a location and the target is not there, that location is searched for this task. Do not navigate back to a seen-empty location until every location in the admissible actions that you have never observed has been visited first.

4. Press forward through numbered receptacles. If you are searching cabinets, go cabinet 1, then cabinet 2, then cabinet 3, and so on. After opening cabinet 5, do not jump back to cabinet 1 or countertop 1 because you already know those are empty. The object is often in a high-numbered container you have not yet reached.

5. Detect loops by what you have already seen, not by repeated identical actions. Alternating between countertop 1, fridge 1, shelf 1, shelf 2, and cabinet 1 is still a loop if all of those are known empty for the current target. If your recent navigation targets have all been seen-empty locations, stop and choose the next admissible location name that has never appeared in your observations for this task.

6. Avoid filler actions while searching. Repeatedly calling look, inventory, or examine on an already-visible object wastes budget. If the current observation shows the target is absent and you are not carrying it, the productive action is always to move to the next never-observed location.
<!-- SLOW_UPDATE_END -->

---

# ALGO-03 Residual Module: action-interface firewall

This module has higher priority than all stylistic preferences in the skill
above, but it must not change the task strategy. Its only purpose is to prevent
malformed responses.

- Always end with exactly one action copied verbatim from the admissible action
  list.
- Use the required wrapper format exactly:
  `<think>brief private state update</think><action>admissible action</action>`.
- Do not output JSON, bullets, multiple actions, code blocks, quotes around the
  action, or explanatory text outside the two tags.
- If several actions look plausible, choose one admissible action and continue;
  never describe alternatives.
- If uncertain, prefer a safe admissible observation/navigation action such as
  opening, looking in/on, or going to a relevant unsearched place, but still copy
  the exact action string from the admissible list.
- Before emitting, check that the `<action>` text is an exact member of the
  admissible action list and that there is no extra text after `</action>`.
