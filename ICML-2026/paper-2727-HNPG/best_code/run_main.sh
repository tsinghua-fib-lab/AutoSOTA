source .venv/bin/activate
uv sync

python3 main.py --pca-dim 10 --n-runs 10 --dataset mnist --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset mnist --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset mnist --mem-R 3 --beta 1.0 --noise_sigma 0.3


python3 main.py --pca-dim 10 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 1.0 --noise_sigma 0.3




python3 main.py --pca-dim 10 --n-runs 10 --dataset mnist --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset mnist --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset mnist --mem-R 3 --beta 10 --noise_sigma 0.3


python3 main.py --pca-dim 10 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset cifar10 --mem-R 3 --beta 10 --noise_sigma 0.3



python3 main.py --pca-dim 10 --n-runs 10 --dataset synthetic --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset synthetic --mem-R 3 --beta 10 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset synthetic --mem-R 3 --beta 10 --noise_sigma 0.3






python3 main.py --pca-dim 10 --n-runs 10 --dataset synthetic --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 20 --n-runs 10 --dataset synthetic --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 100 --n-runs 10 --dataset synthetic --mem-R 3 --beta 1.0 --noise_sigma 0.3




python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 1 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 2 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 3 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 4 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 5 --beta 1.0 --noise_sigma 0.3

python3 main.py --pca-dim 3 --n-runs 10 --dataset synthetic --mem-R 6 --beta 1.0 --noise_sigma 0.3
