#!/bin/bash

devices=[2]
task=scam_multi_itransformer
if_snr=True
batch_size=32
lr=0.0005

for input_size in 96
do
    for output_size in 96 192 336 720
    do
        python src/train.py \
        data=ETTh1 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=$task \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=1 \
        model.net.predictor.pred_model.if_snr=$if_snr


        python src/train.py \
        data=ETTh2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=$task \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=1 \
        model.net.predictor.pred_model.if_snr=$if_snr

        python src/train.py \
        data=ETTm1 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=$task \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=1 \
        model.net.predictor.pred_model.if_snr=$if_snr

        python src/train.py \
        data=ETTm2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=$task \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=1 \
        model.net.predictor.pred_model.if_snr=$if_snr

        # python src/train.py \
        # data=electricity \
        # if_sample=True \
        # sample_size=64 \
        # data.input_size=$input_size \
        # data.output_size=$output_size \
        # task=$task \
        # trainer.devices=$devices \
        # model.net.predictor.pred_model.dim=512 \
        # model.net.predictor.pred_model.d_ff=512 \
        # model.net.predictor.pred_model.num_layers=2 \
        # model.net.predictor.pred_model.num_heads=8 \


        # python src/train.py \
        # data=traffic \
        # if_sample=True \
        # sample_size=64 \
        # data.input_size=$input_size \
        # data.output_size=$output_size \
        # task=$task \
        # trainer.devices=$devices \
        # model.net.predictor.pred_model.dim=512 \
        # model.net.predictor.pred_model.d_ff=512 \
        # model.net.predictor.pred_model.num_layers=2 \
        # model.net.predictor.pred_model.num_heads=8 \


        python src/train.py \
        data=weather \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=$task \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=512 \
        model.net.predictor.pred_model.d_ff=512 \
        model.net.predictor.pred_model.num_layers=2 \
        model.net.predictor.pred_model.num_heads=1 \
        model.net.predictor.pred_model.if_snr=$if_snr
    done
done
