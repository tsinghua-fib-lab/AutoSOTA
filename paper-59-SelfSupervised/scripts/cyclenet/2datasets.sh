#!/bin/bash

devices=[1]
task=cyclenet
if_snr=False
if_sample=True
sample_size=64
batch_size=32
lr=0.001

for input_size in 96
do
    for output_size in 96 192 336 720
    do
        python src/train.py \
        data=electricity \
        if_sample=$if_sample \
        sample_size=$sample_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=512 \
        model.net.if_snr=$if_snr

        python src/train.py \
        data=traffic \
        if_sample=$if_sample \
        sample_size=$sample_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=512 \
        model.net.if_snr=$if_snr

    done
done