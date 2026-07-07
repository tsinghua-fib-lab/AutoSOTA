# Code Analysis: MDGMIX (Paper 1284)

## Evaluation Path
- Entry: runexp.py -> MDGMIX.py
- Eval: python runexp.py --dataset Cora --lr 0.0001 --downstreamlr 0.001 --epochs 500 --shot_num 1 --gpu 0 --skip_pretrain 1
- skip_pretrain=1: loads /repo/model_Cora.pkl, runs downstream only
- Downstream: 100 few-shot splits x 400 inner epochs, early stopping patience=50
- Output: Micro:XX.XX (mean Micro F1 = Accuracy)

## Train/Inference Path
- Pre-training (skip_pretrain=0): MDGMIX.forward() -> GCN encode -> domain classifier -> composition predictor -> KL + CE loss
- Downstream: downprompt -> compose domain tokens with node features (mul) -> GCN encode -> cosine similarity with class prototypes -> softmax -> CE loss
- Prototypes: averageemb() scatter-mean per class (1 node/class in 1-shot)

## Config Path
- hid_units=256, unify_dim=50, l2_coef=1e-4, patience=50, dropout=0.5
- boundary_ratio_list=[0.1], similarity_threshold_list=[0.3] (was changed to 0.7 in repro)

## Metric Parser
- Regex: Micro:(\d+\.\d+)\+/-(\d+\.\d+)
- Micro F1 = Accuracy (multi-class single-label)

## Reusable Resources
- /repo/model_Cora.pkl: pre-trained GCN + domain tokens
- /fewshot_dataset/fewshot_cora_node/: 100 few-shot splits
- /datasets/: 7 graph datasets

## Safe Modification Targets
1. models/gcn.py:35-40: GCN forward (add residuals, CODE-1)
2. MDGMIX.py:302-320: downstream training loop (pseudo-labels, entropy, LR schedule)
3. downprompt.py:48-51,100-102: prototype computation (EMA, PPR)
4. downprompt.py:138-159: token composition (gated fusion)

## Key Observation
- GCN is FROZEN during downstream (only log.parameters() in optimizer)
- CODE-1 needs GCN added to downstream optimizer
- skip_pretrain=1 means pre-training params do not affect results
