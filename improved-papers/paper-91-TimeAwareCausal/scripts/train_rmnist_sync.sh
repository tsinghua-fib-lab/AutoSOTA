echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++          SYNC: Rotated MNIST         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 0 \
        --data_name RMNIST \
        --data_path '../data/rmnist'\
        --num_classes 10 \
        --data_size '[1, 28, 28]' \
        --source-domains 10 \
        --intermediate-domains 3 \
        --target-domains 6 \
        --mode train \
        --model-func MNIST_CNN \
        --feature-dim 128 \
        --epochs 120 \
        --iterations 200 \
        --train_batch_size 48 \
        --eval_batch_size 48 \
        --test_epoch -1 \
        --algorithm SYNC \
        --zc-dim 32 \
        --lambda_evolve 0.005 \
        --lambda_mi 0.005 \
        --lambda_causal 0.001 \
        --seed $seed \
        --save_path './logs/RMNIST' \
        --record
  echo "=================="
done