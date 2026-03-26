echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++              SYNC: ONP               +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name ONP \
        --data_path '../data/processed_ONP.csv'\
        --num_classes 2 \
        --data_size '[1, 58]' \
        --source-domains 12 \
        --intermediate-domains 4 \
        --target-domains 8 \
        --mode train \
        --model-func Toy_Linear_FE \
        --feature-dim 512 \
        --epochs 30 \
        --iterations 200 \
        --train_batch_size 64 \
        --eval_batch_size 50 \
        --test_epoch -1 \
        --algorithm SYNC \
        --zc-dim 32 \
        --lambda_evolve 0.001 \
        --lambda_mi 0.001 \
        --lambda_causal 0.001 \
        --seed $seed \
        --save_path './logs/ONP' \
        --record
  echo "=================="
done