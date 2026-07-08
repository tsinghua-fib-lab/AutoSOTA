#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="llama3.1-8B-Instruct"
FEATURE_DIR="./features_generic"
PROTO_DIR="./prototypes_generic"
KAPPA=20

# BoolQ | layer 12 | alpha 0.5 | beta -1.0
python get_activations_generic.py --model_name "$MODEL" --dataset boolq --split train --layer 12 --save_dir "$FEATURE_DIR"
python get_prototypes_generic.py --feature_file "$FEATURE_DIR/${MODEL}_boolq_train_l12.npz" --save_dir "$PROTO_DIR"
python evaluate_generic.py --model_name "$MODEL" --dataset boolq --layer 12 --prototype_path "$PROTO_DIR/${MODEL}_boolq_train_l12_proto.npz" --kappa "$KAPPA" --alpha 0.5 --beta -1.0

# COPA | layer 16 | alpha 0.9 | beta -0.4
python get_activations_generic.py --model_name "$MODEL" --dataset copa --split train --layer 16 --save_dir "$FEATURE_DIR"
python get_prototypes_generic.py --feature_file "$FEATURE_DIR/${MODEL}_copa_train_l16.npz" --save_dir "$PROTO_DIR"
python evaluate_generic.py --model_name "$MODEL" --dataset copa --layer 16 --prototype_path "$PROTO_DIR/${MODEL}_copa_train_l16_proto.npz" --kappa "$KAPPA" --alpha 0.9 --beta -0.4

# StoryCloze | layer 27 | alpha 0.9 | beta -0.7
python get_activations_generic.py --model_name "$MODEL" --dataset storycloze --split train --layer 27 --save_dir "$FEATURE_DIR"
python get_prototypes_generic.py --feature_file "$FEATURE_DIR/${MODEL}_storycloze_train_l27.npz" --save_dir "$PROTO_DIR"
python evaluate_generic.py --model_name "$MODEL" --dataset storycloze --layer 27 --prototype_path "$PROTO_DIR/${MODEL}_storycloze_train_l27_proto.npz" --kappa "$KAPPA" --alpha 0.9 --beta -0.7

# MMLU Global | layer 28 | alpha 0.8 | beta -0.1
python get_activations_generic.py --model_name "$MODEL" --dataset mmlu_global --split train --layer 28 --save_dir "$FEATURE_DIR"
python get_prototypes_generic.py --feature_file "$FEATURE_DIR/${MODEL}_mmlu_global_train_l28.npz" --save_dir "$PROTO_DIR"
python evaluate_generic.py --model_name "$MODEL" --dataset mmlu_global --layer 28 --prototype_path "$PROTO_DIR/${MODEL}_mmlu_global_train_l28_proto.npz" --kappa "$KAPPA" --alpha 0.8 --beta -0.1

# Winogrande | layer 16 | alpha 0.9 | beta -0.8
python get_activations_generic.py --model_name "$MODEL" --dataset winogrande --split train --layer 16 --save_dir "$FEATURE_DIR"
python get_prototypes_generic.py --feature_file "$FEATURE_DIR/${MODEL}_winogrande_train_l16.npz" --save_dir "$PROTO_DIR"
python evaluate_generic.py --model_name "$MODEL" --dataset winogrande --layer 16 --prototype_path "$PROTO_DIR/${MODEL}_winogrande_train_l16_proto.npz" --kappa "$KAPPA" --alpha 0.9 --beta -0.8
