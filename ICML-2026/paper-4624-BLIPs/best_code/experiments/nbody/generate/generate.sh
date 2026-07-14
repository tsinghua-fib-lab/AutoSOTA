python experiments/nbody/generate/generate_dataset.py --num-train 10000 --seed 43 --sufix small
mkdir -p data
mkdir -p data/nbody
mv *.npy data/nbody