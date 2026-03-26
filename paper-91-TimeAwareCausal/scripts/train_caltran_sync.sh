echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++           SYNC: Caltran              +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name Caltran \
        --data_path '../data/Caltran'\
        --num_classes 2 \
        --data_size '[3, 84, 84]' \
        --source-domains 19 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --model-func Resnet18 \
        --feature-dim 512 \
        --epochs 100 \
        --iterations 200 \
        --train_batch_size 24 \
        --eval_batch_size 24 \
        --test_epoch -1 \
        --algorithm SYNC \
        --zc-dim 32 \
        --lambda_evolve 0.005 \
        --lambda_mi 0.005 \
        --lambda_causal 0.002 \
        --seed $seed \
        --save_path './logs/Caltran' \
        --record
  echo "=================="
done