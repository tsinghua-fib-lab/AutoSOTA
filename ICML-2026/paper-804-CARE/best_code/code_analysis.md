# Code Analysis — CARE Paper 804

## Evaluation Path
- Entry: main.py -> Trainer.test(mode='test')
- Data: datasets/cifar100.py -> CIFAR100_IR100_NR50
- Forward: self.model(image) -> classifier head + backbone
- TTE: When tte=True, ncrops handled in test() - averages across crops
- Output: evaluator.evaluate() -> prints => result block with accuracy

## Train Path
- Entry: main.py -> Trainer.train()
- Model: PeftModelFromCLIP with AdaptFormer (ViT-B/16 backbone)
- Optimizer: SGD (lr=0.01, wd=5e-4, momentum=0.9)
- Scheduler: CosineAnnealingLR (no warmup)
- Loss: LA (LogitAdjustedLoss) by default
- CARE consensus: compute_candidate_correction() builds candidate_count
- Save: best model by mean_acc saved to output_dir/checkpoint.pth.tar

## Config Path
- Data: configs/data/cifar100_ir100_nr50.yaml
- Model: configs/model/clip_vit_b16.yaml
- CLI overrides: main.py argparse REMAINDER -> cfg.merge_from_list(args.opts)
- Output dir: auto-generated from data+model+opts

## Metric Parser
- Evaluator in utils/evaluator.py prints => result block
- Primary: Top-1 Accuracy = accuracy in result block (percentage)

## Key Files
- main.py — entry point
- trainer.py — Trainer class (build, train, test)
- utils/config.py — YACS config
- utils/losses.py — LDAM, LA, BS, CB, GRW, Focal, LADE, GCL
- utils/evaluator.py — Evaluator
- models/peft_vit.py — AdaptFormer for ViT
- models/classifiers.py — CosineClassifier
- models/models.py — PeftModelFromCLIP, ZeroShotCLIP
- utils/templates.py — ZEROSHOT_TEMPLATES

## Known Issues
1. trainer.py:405 — self.criterion == BalancedSoftmaxLoss(...) (FIXED: =)
2. No LR warmup
3. DataLoader worker_seed not explicitly set

## Safe Modification Targets
- trainer.py — criterion, optimizer, scheduler, training/test loop
- utils/losses.py — new/modified loss functions
- models/ — new PEFT modules
- Config YAML files and CLI opts

## Dangerous (Do Not Modify)
- utils/evaluator.py
- datasets/
- clip/
- Test data at /datasets/cifar-100-python/

## Manifest Recovery
- eval_command uses in-container format
- root /datasets points to container path
- GPU 0 is correct inside container
