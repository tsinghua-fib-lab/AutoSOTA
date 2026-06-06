echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++            SYNC: Toy Circle          +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name ToyCircle \
        --data_path '../data/half-circle.pkl' \
        --num_classes 2 \
        --data_size '[1, 2]' \
        --source-domains 15 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --model-func Toy_Linear_FE \
        --feature-dim 128 \
        --epochs 30 \
        --iterations 200 \
        --train_batch_size 64 \
        --eval_batch_size 50 \
        --test_epoch -1 \
        --algorithm SYNC \
        --zc-dim 20 \
        --lambda_evolve 0.01 \
        --lambda_mi 1. \
        --lambda_causal 0.02 \
        --seed $seed \
        --save_path './logs/ToyCircle' \
        --record
  echo "=================="
done