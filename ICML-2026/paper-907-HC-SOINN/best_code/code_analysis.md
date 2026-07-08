# Code Analysis for Paper 907 (HC-SOINN)

## Evaluation Path
- Entry: /repo/main.py -> trainer.train(args)
- Config: /repo/exps/dualprompt/dualprompt_hc_soinn.json
- Model: models/dualprompt.py -> Learner class
- Classifier: utils/hc_soinn_classifier.py -> HCSOINNClassifier
- Alignment: utils/STAR.py -> STARAligner

## Metric Parsing
- A_Avg: "Average Accuracy (HC-SOINN):" in stdout/log
- A_Last: Last element of "HC-SOINN top1 curve: [...]"  
- FC_A_Avg: "Average Accuracy (FC):" in stdout/log
- NCM_A_Avg: "Average Accuracy (NCM):" in stdout/log
- Training_Time: "Training time: X.XX seconds (Y.YY minutes)" in log

## Key Levers (from config)
1. hcsoinn_alpha (0.5) - NCM vs sub-cluster blending
2. hcsoinn_max_proto_per_class (60) - prototypes per class
3. hcsoinn_soinn_ad (20) - edge age threshold
4. hcsoinn_soinn_max_iter (1) - SOINN refinement passes
5. star_lambda (0.999) - EMA momentum for STAR
6. tuned_epoch (5) - training epochs per task
7. init_lr (0.001) - learning rate
8. batch_size (24)
9. pull_constraint_coeff (0.1)

## Safe Modification Targets
1. utils/hc_soinn_classifier.py - Distance function, prototype allocation, alpha blending
2. utils/STAR.py - STAR alignment method, anchor selection
3. models/dualprompt.py - Loss function, training loop
4. exps/dualprompt/dualprompt_hc_soinn.json - Config parameters

## Risky Files (do not modify)
- main.py - Entry point (must stay same)
- trainer.py - Metric parsing and logging
- utils/data_manager.py - Data loading
- utils/toolkit.py - Utility functions

## Reusable Resources
- CIFAR-100 data: /data/Datasets/cifar-100-python/
- ViT-B/16-IN1K weights: cached via timm
- No /paper_data mount

## Red Line Confirmation
- Eval command unchanged
- No changes to metric computation, test data, or labels
