echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++         SYNC: PowerSupply            +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

for seed in 0
do
  echo "Running exp with $seed"
  python main.py --gpu_ids 7 \
        --data_name PowerSupply \
        --data_path '../data/powersupply.arff'\
        --num_classes 2 \
        --data_size '[1, 2]' \
        --source-domains 15 \
        --intermediate-domains 5 \
        --target-domains 10 \
        --mode train \
        --model-func Toy_Linear_FE \
        --feature-dim 512 \
        --epochs 50 \
        --iterations 200 \
        --train_batch_size 64 \
        --eval_batch_size 50 \
        --test_epoch -1 \
        --algorithm SYNC \
        --zc-dim 32 \
        --lambda_evolve 0.002 \
        --lambda_mi 0.001 \
        --lambda_causal 0.01 \
        --seed $seed \
        --save_path './logs/PowerSupply' \
        --record
  echo "=================="
done