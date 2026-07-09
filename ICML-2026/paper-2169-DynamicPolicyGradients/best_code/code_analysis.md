# Code Analysis — Paper 2169 (GReinSS)

## Evaluation
- Eval: python3 eval_set.py
- Loads pre-trained model, runs inference, computes F1
- Metric: grep "GReinSS F1.*Median:" for median F1

## Training  
- train_model_off_policy() at sharedGen.py:1074
- RMSprop (alpha=0.99), lr=1e-3
- Early stopping: 500-epoch patience window
- Loss: REINFORCE with SNIS at line 1966
- Gradient clipping commented out at line 1374

## Safe Targets
- sharedGen.py: optimizer, LR schedule, grad clip, patience
- eval_set.py: N_PREDICTION_SAMPLES, endingBias, off-policy multiplier
