export CUDA_VISIBLE_DEVICES=2

python benchmark_kpi.py \
  --model KPI \
  --dataset_name ionosphere \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.1 \
  --stop 10 \
  --k 3\
  --metric nan_manhattan \
  --weights uniform
  

  python benchmark_kpi.py \
  --model KPI \
  --dataset_name ionosphere \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.2 \
  --stop 10 \
  --k 2\
  --metric nan_manhattan \
  --weights uniform

  python benchmark_kpi.py \
  --model KPI \
  --dataset_name ionosphere \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.3 \
  --stop 10 \
  --k 4\
  --metric nan_manhattan \
  --weights uniform

  python benchmark_kpi.py \
  --model KPI \
  --dataset_name ionosphere \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.4 \
  --stop 10 \
  --k 5\
  --weights distance
