#!/bin/bash

devices=[2]
task=patchtst
if_sample=False
for if_snr in True False
do
    for input_size in 96
    do
        for output_size in 96 192 336 720
        do
            python src/train.py \
            data=ETTh1 \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            trainer.devices=$devices \
            model.net.dim=128 \
            model.net.d_ff=128 \
            model.net.num_heads=4 \
            model.net.dropout=0.3 \
            model.net.fc_dropout=0.3 \
            model.net.head_dropout=0.0 \
            model.net.if_snr=$if_snr \
            if_sample=$if_sample

            python src/train.py \
            data=ETTh2 \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            trainer.devices=$devices \
            model.net.dim=128 \
            model.net.d_ff=128 \
            model.net.num_heads=4 \
            model.net.dropout=0.3 \
            model.net.fc_dropout=0.3 \
            model.net.head_dropout=0.0 \
            model.net.if_snr=$if_snr \
            if_sample=$if_sample

            python src/train.py \
            data=ETTm1 \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            trainer.devices=$devices \
            model.net.dim=128 \
            model.net.d_ff=256 \
            model.net.num_heads=8 \
            model.net.dropout=0.2 \
            model.net.fc_dropout=0.2 \
            model.net.head_dropout=0.0 \
            model.net.if_snr=$if_snr \
            if_sample=$if_sample

            python src/train.py \
            data=ETTm2 \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            trainer.devices=$devices \
            model.net.dim=128 \
            model.net.d_ff=256 \
            model.net.num_heads=8 \
            model.net.dropout=0.2 \
            model.net.fc_dropout=0.2 \
            model.net.head_dropout=0.0 \
            model.net.if_snr=$if_snr \
            if_sample=$if_sample

            python src/train.py \
            data=weather \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            trainer.devices=$devices \
            model.net.dim=128 \
            model.net.d_ff=256 \
            model.net.num_heads=8 \
            model.net.dropout=0.2 \
            model.net.fc_dropout=0.2 \
            model.net.head_dropout=0.0 \
            model.net.if_snr=$if_snr \
            if_sample=$if_sample

        done
    done
done