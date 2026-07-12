set -e

EXPLAINER_NAME=agi

for dataset in imagenet oxfordpet oxfordflower; do
    for model_name in vgg16 resnet18 inception; do
    python scripts/diffid.py \
        --config-name=$EXPLAINER_NAME \
        dataset=$dataset \
        model_name=$model_name \
        max_eval_samples=500 \
        save_dir=results/benchmark_diffid/$dataset/$EXPLAINER_NAME/$model_name
    done
done