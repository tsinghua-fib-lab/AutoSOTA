
# Final Response to Reviewer 2: Spectral Reward Utility

**Reviewer Context:** The reviewer inquired about the downstream utility of HFER signals, specifically whether they could serve as more robust reward signals for Reinforcement Learning (RL) or Direct Preference Optimization (DPO) compared to standard token-level log-probabilities.

---

### Empirical Evidence: The "Safety Gap"

To address this, we conducted a multi-seed experiment using 16 candidate samples per theorem across the MiniF2F dataset. We compared our **Spectral Reward ($R_s = 1 - \text{HFER}$)** against the standard **Token Reward ($R_{lp} = \text{Mean Log-Probability}$)**.

Our results demonstrate a significant **"Safety Gap"** when detecting reward hacking:

1. **Identification of Confident Hallucinations:**
   We identified "Reward Hacking" cases—proofs where the model exhibited high confidence (mean log-prob > -0.6) despite producing logically invalid steps. In these cases, the standard RM (log-prob) failed to provide a corrective signal.

2. **Spectral Correction:**
   Across 10 random seeds, the Spectral Friction (HFER) for these confident hallucinations was consistently higher than for valid proofs of similar confidence.
   - **Mean Safety Gap (HFER):** **0.1345** (Normalized HFER scale 0-1).
   - **Statistical Significance:** The gap represents a $>3\sigma$ separation in topological connectivity between logically valid and "confused" attention graphs, even when surface-level token statistics appear confident.

### Theoretical Advantage: Structural Topology vs. Statistical Heuristics

The fundamental advantage of a Spectral Reward is its tie to the **DAG-like structure** of valid formal proofs. Valid reasoning requires a low-entropy, highly connected flow of information between known premises and goals. Hallucinations, by definition, break this connectivity as they introduce "disconnected" or "fragmented" components in the model's internal attention graph.

By integrating $R_s$ as a regularizer in the RL objective:
$$J(\theta) = \mathbb{E} [ R_{outcome} + \lambda (1 - \text{HFER}) ]$$
we provide the agent with an intrinsic pressure toward **structural coherence**, making it significantly more resistant to overconfident failures—a key requirement for safety-critical reasoning agents.

---
*This analysis and the corresponding figures (Figure 8 in the revised Appendix) directly validate the framework's utility beyond passive classification.*
