export DATADIR=.
export SAVEDIR=logs_train
export PRECISION=no
export PROCESSES=8
export LOSS=reinforce

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MIS/RB_small_train --valid_path $DATADIR/MIS/RB_small_val --problem mis --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 100 --num_k 20 --num_tp 50 --num_kp 10 --num_h 128 --num_l 5 --num_d 5 --lambd 0.5 --method RLNN --epochs 100 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_rb_small_mis > $LOGDIR/log_rb_small_mis_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MIS/RB_large_train --valid_path $DATADIR/MIS/RB_large_val --problem mis --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 200 --num_k 20 --num_tp 300 --num_kp 10 --num_h 128 --num_l 5 --num_d 5 --lambd 0.5 --method RLNN --epochs 100 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_rb_large_mis > $LOGDIR/log_rb_large_mis_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MIS/ER_small_train --valid_path $DATADIR/MIS/ER_small_val --problem mis --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 200 --num_k 20 --num_tp 300 --num_kp 10 --num_h 128 --num_l 5 --num_d 5 --beta 1.001 --lambd 0.5 --method RLNN --epochs 100 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_er_mis > $LOGDIR/log_er_mis_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MCl/RB_small_train --valid_path $DATADIR/MCl/RB_small_val --problem mcl --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 100 --num_k 20 --num_tp 100 --num_kp 20 --num_h 128 --num_l 5 --num_d 10 --lambd 2 --method RLNN --epochs 200 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_rb_small_mcl > $LOGDIR/log_rb_small_mcl_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MCl/RB_large_train --valid_path $DATADIR/MCl/RB_large_val --problem mcl --train_sample 1000 --valid_sample 100 --batch_size 256 --num_t 200 --num_k 20 --num_tp 500 --num_kp 20 --num_h 128 --num_l 5 --num_d 10 --lambd 2 --method RLNN --epochs 200 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_rb_large_mcl > $LOGDIR/log_rb_large_mcl_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MCut/BA_small_train --valid_path $DATADIR/MCut/BA_small_val --problem mcut --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 100 --num_k 20 --num_tp 50 --num_kp 10 --num_h 128 --num_l 5 --num_d 20 --lambd 0.5 --method RLNN --epochs 100 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_ba_small_mcut > $LOGDIR/log_ba_small_mcut_$LOSS.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --train_path $DATADIR/MCut/BA_large_train --valid_path $DATADIR/MCut/BA_large_val --problem mcut --train_sample 1000 --valid_sample 100 --batch_size 64 --num_t 200 --num_k 20 --num_tp 300 --num_kp 10 --num_h 128 --num_l 5 --num_d 20 --lambd 0.5 --method RLNN --epochs 100 --lr 0.0001 --mixed_precision $PRECISION --do_train --loss $LOSS --save_dir $DATADIR/$LOSS/rlnn_ba_large_mcut > $LOGDIR/log_ba_large_mcut_$LOSS.txt






