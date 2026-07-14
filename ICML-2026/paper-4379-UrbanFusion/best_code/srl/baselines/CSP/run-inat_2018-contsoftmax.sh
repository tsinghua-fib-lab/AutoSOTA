LOSS=contsoftmax
ACT=leakyrelu

RATIO=0.5
LR=0.0002


DIR=../model_dir/models_inat_2018/

NEG=1
W=1
SIMCSEW=1

TMPBAT=20
TMPNEG=1
TMPSIM=1

DROPOUT=0.5


EPOCH=200
# BATCH=2048
BATCH=1024

NUMFIL=256

MINR=0.1
RATIO=0.05
SPLR=0.001
LR=0.0005
NEG=1
W=1
SIMCSEW=1
TMPBAT=20
TMPNEG=1
TMPSIM=1
DROPOUT=0.5


python3 train_unsuper.py \
                --dataset inat_2018 \
                --meta_type ebird_meta \
                --eval_split val \
                --load_val_op T \
                --load_cnn_features_train  T \
                --load_cnn_features  T \
                --load_cnn_predictions F \
                --device cuda:0 \
                --spa_enc_type gridcell \
                --frequency_num 32 \
                --max_radius 360 \
                --min_radius $MINR \
                --num_hidden_layer 1 \
                --hidden_dim 512 \
                --spa_f_act $ACT  \
                --unsuper_lr  $LR \
                --batch_size $BATCH \
                --num_filts $NUMFIL \
                --lr $SPLR \
                --model_dir $DIR \
                --neg_rand_type sphericalold \
                --num_neg_rand_loc $NEG \
                --rand_sample_weight $W \
                --simcse_weight  $SIMCSEW  \
                --num_epochs $EPOCH \
                --num_epochs_unsuper 30 \
                --eval_frequency 5 \
                --dropout $DROPOUT  \
                --train_sample_ratio $RATIO \
                --train_sample_method stratified-fix  \
                --unsuper_loss $LOSS \
                --unsuper_temp_inbatch $TMPBAT \
                --unsuper_temp_negloc  $TMPNEG \
                --unsuper_temp_simcse  $TMPSIM \
                --inat2018_resolution pretrain \
                --cnn_pred_type full \
                --do_unsuper_train  T \
                --load_unsuper_model F \
                --do_super_train T  \
                --load_super_model F 
                                        

















