#!/bin/bash

devices=[2]
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
        task=scam_multi_patchtst \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=16 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_heads=4 \
        model.net.predictor.pred_model.dropout=0.2 \
        model.net.predictor.pred_model.fc_dropout=0.2 \
        model.net.predictor.pred_model.head_dropout=0.0 \
        model.net.predictor.pred_model.if_snr=$if_snr

        python src/train.py \
        data=ETTh2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=scam_multi_patchtst \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=16 \
        model.net.predictor.pred_model.d_ff=128 \
        model.net.predictor.pred_model.num_heads=4 \
        model.net.predictor.pred_model.dropout=0.3 \
        model.net.predictor.pred_model.fc_dropout=0.3 \
        model.net.predictor.pred_model.head_dropout=0.0 \
        model.net.predictor.pred_model.if_snr=$if_snr

        python src/train.py \
        data=ETTm1 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=scam_multi_patchtst \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=256 \
        model.net.predictor.pred_model.num_heads=16 \
        model.net.predictor.pred_model.dropout=0.2 \
        model.net.predictor.pred_model.fc_dropout=0.2 \
        model.net.predictor.pred_model.head_dropout=0.0 \
        model.net.predictor.pred_model.if_snr=$if_snr

        python src/train.py \
        data=ETTm2 \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=scam_multi_patchtst \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=256 \
        model.net.predictor.pred_model.num_heads=16 \
        model.net.predictor.pred_model.dropout=0.2 \
        model.net.predictor.pred_model.fc_dropout=0.2 \
        model.net.predictor.pred_model.head_dropout=0.0 \
        model.net.predictor.pred_model.if_snr=$if_snr

        # python src/train.py \
        # data=electricity \
        # data.input_size=$input_size \
        # data.output_size=$output_size \
        # task=scam_multi_patchtst \
        # trainer.devices=$devices \
        # model.net.predictor.pred_model.dim=128 \
        # model.net.predictor.pred_model.d_ff=256 \
        # model.net.predictor.pred_model.num_heads=16 \
        # model.net.predictor.pred_model.dropout=0.2 \
        # model.net.predictor.pred_model.fc_dropout=0.2 \
        # model.net.predictor.pred_model.head_dropout=0.0

        # python src/train.py \
        # data=traffic \
        # data.input_size=$input_size \
        # data.output_size=$output_size \
        # task=scam_multi_patchtst \
        # trainer.devices=$devices \
        # model.net.predictor.pred_model.dim=128 \
        # model.net.predictor.pred_model.d_ff=256 \
        # model.net.predictor.pred_model.num_heads=16 \
        # model.net.predictor.pred_model.dropout=0.2 \
        # model.net.predictor.pred_model.fc_dropout=0.2 \
        # model.net.predictor.pred_model.head_dropout=0.0

        python src/train.py \
        data=weather \
        data.input_size=$input_size \
        data.output_size=$output_size \
        data.batch_size=$batch_size \
        model.optimizer.lr=$lr \
        task=scam_multi_patchtst \
        trainer.devices=$devices \
        model.net.predictor.pred_model.dim=128 \
        model.net.predictor.pred_model.d_ff=256 \
        model.net.predictor.pred_model.num_heads=16 \
        model.net.predictor.pred_model.dropout=0.2 \
        model.net.predictor.pred_model.fc_dropout=0.2 \
        model.net.predictor.pred_model.head_dropout=0.0 \
        model.net.predictor.pred_model.if_snr=$if_snr
    done
done