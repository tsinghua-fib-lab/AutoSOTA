LOSS=contsoftmax

LR=0.001
ACT=gelu

DIR=../model_dir/models_fmow/



NEG=1
RATIO=0.05
EPOCH=300
W=1
SIMCSEW=1
UNLR=0.00005
LR=0.001
MINR=0.1
TMPBAT=20
TMPNEG=1
TMPSIM=1
        
                

       
python3 train_unsuper.py \
                    --dataset fmow \
                    --meta_type ebird_meta \
                    --eval_split val \
                    --load_val_op T \
                    --load_cnn_features_train  T \
                    --load_cnn_features  T \
                    --load_cnn_predictions T \
                    --device cuda:1 \
                    --spa_enc_type  gridcell \
                    --frequency_num  32  \
                    --max_radius 360 \
                    --min_radius $MINR  \
                    --num_hidden_layer 1 \
                    --hidden_dim 512 \
                    --spa_f_act  $ACT \
                    --unsuper_lr  $UNLR \
                    --lr $LR \
                    --model_dir $DIR \
                    --neg_rand_type sphericalold \
                    --num_epochs $EPOCH \
                    --num_epochs_unsuper 300 \
                    --eval_frequency 5 \
                    --train_sample_ratio $RATIO \
                    --train_sample_method stratified-fix \
                    --unsuper_loss $LOSS \
                    --num_neg_rand_loc $NEG \
                    --rand_sample_weight $W \
                    --simcse_weight  $SIMCSEW  \
                    --unsuper_temp_inbatch $TMPBAT \
                    --unsuper_temp_negloc  $TMPNEG \
                    --unsuper_temp_simcse  $TMPSIM \
                    --inat2018_resolution pretrain \
                    --cnn_pred_type fewshot  \
                    --do_unsuper_train T \
                    --load_unsuper_model T \
                    --do_super_train T \
                    --load_super_model T \
                    --do_epoch_save F
                            





        

