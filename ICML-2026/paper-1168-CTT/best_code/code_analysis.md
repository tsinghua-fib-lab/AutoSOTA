# Code Analysis for Paper 1168 - CrossTaskTransfer

## Evaluation Path
- main.py::main() -> run_protocol_once() -> trains F (NC base), H (LP base), weight transfers, embedding transfers, joint baseline
- Key metric extraction from stdout: lines matching "NC_joint_test_acc : <mean> +/- <std>"
- All metrics printed as percentages (0-100 range)

## Train/Inference Path
- train.py::train_node_model() - single-task NC training (GCN/SAGE encoder + linear classifier)
- train.py::train_link_model() - single-task LP training (GCN/SAGE encoder + LinkPredictor)
- train.py::train_joint_model() - joint NC+LP training with shared encoder
- train.py::train_joint_lambda_sweep() - grid search over lambda values

## Config Path
- CLI args in main.py::parse_args()
- ProtocolConfig dataclass in util.py
- Hardcoded: lr=0.01, NC weight_decay=5e-4, LP weight_decay=1e-4 in train.py

## Metric Parser
- Metrics printed at end as "key : mean +/- std" with values in percentage (0-100)
- Parse lines matching "NC_joint_test_acc" and "LP_joint_test_auc"

## Reusable Resources
- Planetoid Citeseer data at /datasets/Planetoid/Citeseer/raw/
- No pre-downloaded paper data mount

## Safe Modification Targets
- train.py: training loops (add gradient clipping, LR schedule, PCGrad, label smoothing, dropout, logging)
- models.py: LinkPredictor (add attention, dropout), JointNC_LP (add features)
- main.py: CLI args for new options

## Risky Files
- datasets.py, util.py: dataset loading and split logic - DO NOT MODIFY
- Scoring scripts, metric definitions - DO NOT MODIFY
