#!/bin/bash
# Reproduction eval script for paper 1168: Cross-Task Transfer in GNNs
# Reproduces: NC Joint Accuracy and LP Joint AUC on CITESEER with GCN
cd /repo
python3 main.py   --root /datasets   --family Planetoid   --name Citeseer   --setting transductive   --seeds 1 2 3 4 5 6 7 8 9 10   --enable_joint_baseline   --hidden 64   --layers 2   --dropout 0.5   --node_epochs 200   --link_epochs 200   --patience 50
