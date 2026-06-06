#!/bin/bash

devices=[2]
task=patchtst

for input_size in 96
do
    for output_size in 96 192 336 720
    do
        python src/train.py \
        data=ETTh1 \
        if_sample=False \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=16 \
        model.net.d_ff=128 \
        model.net.num_heads=4 \
        model.net.dropout=0.3 \
        model.net.fc_dropout=0.3 \
        model.net.head_dropout=0.0

        python src/train.py \
        data=ETTh2 \
        if_sample=False \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=16 \
        model.net.d_ff=128 \
        model.net.num_heads=4 \
        model.net.dropout=0.3 \
        model.net.fc_dropout=0.3 \
        model.net.head_dropout=0.0

        python src/train.py \
        data=ETTm1 \
        if_sample=False \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.d_ff=256 \
        model.net.num_heads=16 \
        model.net.dropout=0.2 \
        model.net.fc_dropout=0.2 \
        model.net.head_dropout=0.0

        python src/train.py \
        data=ETTm2 \
        if_sample=False \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.d_ff=256 \
        model.net.num_heads=16 \
        model.net.dropout=0.2 \
        model.net.fc_dropout=0.2 \
        model.net.head_dropout=0.0

        python src/train.py \
        data=weather \
        if_sample=False \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.d_ff=256 \
        model.net.num_heads=16 \
        model.net.dropout=0.2 \
        model.net.fc_dropout=0.2 \
        model.net.head_dropout=0.0
    done
done
