export DATADIR=.
export LOGDIR=logs_test
export PRECISION=fp16
export PROCESSES=1
export LOSS=erdoes

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/RB_small_test --problem mis --num_t 100 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_rb_small_mis --log_file $LOGDIR/rlnn_rb_small_mis_$LOSS.txt 

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/RB_large_test --problem mis --num_t 200 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_rb_large_mis --log_file $LOGDIR/rlnn_rb_large_mis_$LOSS.txt

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/ER_small_test --problem mis --num_t 200 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/rlnn_er_mis --log_file $LOGDIR/$LOSS/rlnn_er_small_mis_$LOSS.txt 

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/data_er/ER_large_test --problem mis --num_t 800 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/rlnn_er_mis --log_file $LOGDIR/$LOSS/rlnn_er_large_mis_$LOSS.txt

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCl/RB_small_test --problem mcl --num_t 100 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_rb_small_mcl --log_file $LOGDIR/rlnn_rb_small_mcl_$LOSS.txt 

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCl/RB_large_test --problem mcl --num_t 200 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_rb_large_mcl --log_file $LOGDIR/rlnn_rb_large_mcl_$LOSS.txt

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCut/BA_small_test --problem mcut --num_t 100 --num_k 20 --num_h 128 --num_l 5 --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_ba_small_mcut --log_file $LOGDIR/rlnn_ba_small_mcut_$LOSS.txt 

accelerate launch  --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCut/BA_large_test --problem mcut --num_t 200 --num_k 20 --num_h 128 --num_l 5  --method RLNN --mixed_precision $PRECISION --save_dir $DATADIR/$LOSS/rlnn_ba_large_mcut --log_file  $LOGDIR/rlnn_ba_large_mcut_$LOSS.txt




