
devices=[3]
with_revin=False
if_snr=True
lr=0.001
task=scam_multi_patchtst
batch_size=32
if_sample=True
sample_size=64

for data in PEMS03 PEMS04 PEMS07 PEMS08
do
    for input_size in 96
    do
        for output_size in 12 24 36 48
        do
            python src/train.py \
            data=$data \
            trainer.devices=$devices \
            if_sample=$if_sample \
            sample_size=$sample_size \
            data.batch_size=$batch_size \
            model.optimizer.lr=$lr \
            data.input_size=$input_size \
            data.output_size=$output_size \
            task=$task \
            model.net.predictor.pred_model.dim=128 \
            model.net.predictor.pred_model.d_ff=256 \
            model.net.predictor.pred_model.num_heads=8 \
            model.net.predictor.pred_model.dropout=0.2 \
            model.net.predictor.pred_model.fc_dropout=0.2 \
            model.net.predictor.pred_model.head_dropout=0.0 \
            model.net.predictor.pred_model.if_snr=$if_snr \
            model.net.predictor.pred_model.with_revin=$with_revin \

        done
    done
done