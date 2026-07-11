#!/bin/bash
#SBATCH --job-name=maltree_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

# MalTree Embedding Extraction Script
# Extracts all three embedding types: pseudo-static, dynamic, and image

module purge
module load cuda/11.8
module load python/3.10

source activate malware_analysis

mkdir -p logs
mkdir -p embeddings

echo "========================================"
echo "MalTree Embedding Extraction"
echo "Started at: $(date)"
echo "========================================"

# Step 1: Extract image embeddings (2,048-d)
echo ""
echo "Step 1: Extracting image embeddings..."
cd $SLURM_SUBMIT_DIR/malware_image_analysis

python embeddings.py \
    --model-path trained_models/resnet50_best.pth \
    --data-dir $DATA_DIR \
    --output ../embeddings/image_embeddings.json

# Step 2: Extract pseudo-static embeddings (3,512-d)
echo ""
echo "Step 2: Extracting pseudo-static embeddings..."
cd $SLURM_SUBMIT_DIR/malware_embedding_generator

python pseudo_static_extractor.py embed \
    --input $SAMPLES_DIR \
    --output ../embeddings/pseudo_static_embeddings.json

# Step 3: Extract dynamic/behavioral embeddings (1,000-d)
echo ""
echo "Step 3: Extracting dynamic embeddings..."

# First extract behavioral features from ANY.RUN/VirusTotal
python anyrun_extractor.py batch \
    --sha-list $SHA_LIST \
    --output ../embeddings/behavioral_features.json

# Then generate embeddings from behavioral features
python dynamic_embedding_generator.py from-features \
    --features ../embeddings/behavioral_features.json \
    --output ../embeddings/dynamic_embeddings.json

echo ""
echo "========================================"
echo "Embedding extraction completed at $(date)"
echo "========================================"
