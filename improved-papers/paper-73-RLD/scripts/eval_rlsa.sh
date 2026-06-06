export DATADIR=.
export LOGDIR=logs_test
export PRECISION=fp16
export PROCESSES=1


accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/RB_small_test --problem mis --method RLSA --num_t 300 --num_k 200  --num_d 5 --mixed_precision $PRECISION  --log_file $LOGDIR/rlsa_rb_small_mis.txt 

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/RB_large_test --problem mis --method RLSA --num_t 500 --num_k 200  --num_d 5 --mixed_precision $PRECISION  --log_file $LOGDIR/rlsa_rb_large_mis.txt 

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/ER_small_test --problem mis --method RLSA --num_t 500 --num_k 200 --num_d 20 --beta 1.001 --mixed_precision $PRECISION --log_file $LOGDIR/rlsa_er_small_mis.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MIS/data_er/ER_large_test --problem mis --method RLSA --num_t 5000 --num_k 200 --num_d 20 --beta 1.001 --mixed_precision $PRECISION --log_file $LOGDIR/rlsa_er_large_mis.txt 

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCl/RB_small_test --problem mcl --method RLSA --num_t 100 --tau0 4 --num_k 200  --num_d 2 --mixed_precision $PRECISION  --log_file $LOGDIR/rlsa_rb_small_mcl.txt 

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCl/RB_large_test --problem mcl --method RLSA --num_t 500 --tau0 4 --num_k 200  --num_d 2 --mixed_precision $PRECISION  --log_file $LOGDIR/rlsa_rb_large_mcl.txt 

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCut/BA_small_test --problem mcut --method RLSA --tau0 5 --num_t 200 --num_k 200  --num_d 20 --mixed_precision $PRECISION --log_file $LOGDIR/rlsa_ba_small_mcut.txt

accelerate launch --mixed_precision=$PRECISION --num_processes=$PROCESSES main.py --test_path $DATADIR/MCut/BA_large_test --problem mcut --method RLSA --tau0 5 --num_t 500 --num_k 200  --num_d 20 --mixed_precision $PRECISION --log_file $LOGDIR/rlsa_ba_large_mcut.txt

