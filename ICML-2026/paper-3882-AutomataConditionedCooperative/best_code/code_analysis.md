# Code Analysis for Paper 3882 SOTA Optimization

## Evaluation Path
- Script: acc_marl/test_policy.py
- Checkpoints: storage/2buttons_2agents/policy_params_{rad}_{pbrs}_{seed}
- Metric: Success Probability = fraction of episodes where all agents satisfy all DFAs
- Output: "Success rate: X.XX +/- X.XX"

## Key Files
- Config: acc_marl/config/2buttons_2agents.yaml
- Training: acc_marl/train_policy.py + acc_marl/ppo.py
- Env: /autosota_cache/dfa-gym/dfa_gym/dfa_wrapper.py (editable install)
- Network: ActorCritic in train_policy.py (CNN + DFA encoder + MLP heads)

## Key Insight
- Frozen RAD encoder: 0.824
- Trainable encoder (no-RAD): 0.86
- RAD helps scaling (Buttons-4), trainable is better for Buttons-2

## Safe Modifications
- Config YAML: hyperparameters
- train_policy.py: architecture, encoder
- ppo.py: training loop, GAE, losses
- dfa_wrapper.py: rewards, observations (editable install)

## Do Not Modify
- test_policy.py: evaluation protocol
- DFA samplers: test data distribution
- TokenEnv: environment dynamics
