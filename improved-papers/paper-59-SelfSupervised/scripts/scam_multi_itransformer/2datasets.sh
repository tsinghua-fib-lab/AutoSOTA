#!/bin/bash

devices=[2]
task=scam_multi_itransformer
if_snr=True
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
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        task=$task \
        trainer.devices=$devices \
        model.optimizer.lr=$lr \
        model.net.predictor.pred_model.dim=512 \
        model.net.predictor.pred_model.d_ff=512 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=8 \
        model.net.predictor.pred_model.if_snr=$if_snr \
        if_sample=$if_sample \
        sample_size=$sample_size

        python src/train.py \
        data=traffic \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        task=$task \
        trainer.devices=$devices \
        model.optimizer.lr=$lr \
        model.net.predictor.pred_model.dim=512 \
        model.net.predictor.pred_model.d_ff=512 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=8 \
        model.net.predictor.pred_model.if_snr=$if_snr \
        if_sample=$if_sample \
        sample_size=$sample_size
    done
done