#!/bin/bash

devices=[1]
task=itransformer
if_snr=False

for input_size in 96 336
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
        model.net.num_heads=8 \


        python src/train.py \
        data=ETTh2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.num_heads=8 \


        python src/train.py \
        data=ETTm1 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.num_heads=8 \


        python src/train.py \
        data=ETTm2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=128 \
        model.net.num_heads=8 \

        python src/train.py \
        data=electricity \
        if_sample=True \
        sample_size=64 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=512 \
        model.net.num_heads=8 \


        python src/train.py \
        data=traffic \
        if_sample=True \
        sample_size=64 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=512 \
        model.net.num_heads=8 \


        python src/train.py \
        data=weather \
        data.input_size=$input_size \
        data.output_size=$output_size \
        task=$task \
        trainer.devices=$devices \
        model.net.dim=512 \
        model.net.num_heads=8 \

    done
done
