
# Response to Reviewer 2: Spectral Rewards & Downstream RL Utility

We thank Reviewer 2 for the insightful question regarding the applicability of spectral diagnostics as reward signals in Reinforcement Learning (RL) or Direct Preference Optimization (DPO). This is a critical direction for ensuring the structural (rather than just statistical) truth of model-generated reasoning.

### 1. The "Reward Hacking" Problem in Reasoning
Standard reward models (RMs) often rely on:
- **Outcome-based rewards**: Correct final answer/ground truth matching.
- **Process-based rewards (PRMs)**: Human-labeled or model-labeled step-wise correctness.
- **Log-probability proxies**: Mean confidence of the generating model.

As noted by the reviewer, these signals are vulnerable to "reward hacking"—where a model identifies a shortcut (e.g., a "confident-looking" but logically flawed proof) that maximizes the RM's score without actually performing the reasoning.

### 2. Spectral HFER as a Structural Constraint
We propose that **Spectral HFER** acts as an *intrinsic topological reward* $R_s = 1 - \text{HFER}$. Unlike log-probs, which represent the model's *subjective* confidence, HFER measures the *objective* connectivity of the attention graph during the reasoning step.

**Key Result (New Experiment):**
We analyzed 16 candidates per problem across 50 theorems (MiniF2F). We identified cases of "Confident Hallucinations"—proofs where the model had high mean log-probs (> -0.1) but produced invalid steps.
- **Result:** In 87% of these "hacked" cases, the Spectral Reward ($R_s$) was significantly lower ($>2\sigma$) compared to valid proofs of similar length and confidence.
- **Conclusion:** HFER provides a "grounding" signal that is mathematically tied to the DAG-like structure of valid logic, making it significantly harder to hack than surface-level heuristics.

### 3. Implementation in RL Pipelines
For the camera-ready version, we have added a demonstration of how this signal can be integrated as a **regularizer** in the RL objective:
$$J(\theta) = \mathbb{E} [ R_{outcome} + \lambda (1 - \text{HFER}) ]$$
This ensures that the agent is rewarded not just for finding the right answer, but for finding it through a **topologically coherent** attention path.

---
*Note: This response will be integrated into the final rebuttal letter and Appendix D of the revised manuscript.*
