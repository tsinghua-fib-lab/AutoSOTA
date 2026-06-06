
devices=[0]
with_revin=True
if_snr=True
lr=0.001
task=itransformer
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
            model.net.dim=512 \
            model.net.d_ff=512 \
            model.net.num_layers=2 \
            model.net.num_heads=8 \
            model.net.if_snr=$if_snr \
            model.net.with_revin=$with_revin \

        done
    done
done