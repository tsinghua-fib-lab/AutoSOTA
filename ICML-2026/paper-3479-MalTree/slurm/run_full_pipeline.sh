#!/bin/bash
#SBATCH --job-name=maltree_full
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

# MalTree Full Pipeline Script
# Runs the complete analysis from raw data to phylogenetic tree

set -e  # Exit on error

module purge
module load cuda/11.8
module load python/3.10

source activate malware_analysis

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

mkdir -p logs embeddings results trained_models

echo "========================================"
echo "MalTree Full Pipeline"
echo "Started at: $(date)"
echo "========================================"

# Phase 1: Train image embedding model
echo ""
echo "Phase 1: Training ResNet-50 for image embeddings..."
cd $SLURM_SUBMIT_DIR/malware_image_analysis
srun --ntasks=4 python train_distributed.py

# Phase 2: Extract embeddings
echo ""
echo "Phase 2: Extracting embeddings..."

# Image embeddings (2,048-d)
echo "  Extracting image embeddings..."
python embeddings.py \
    --model-path trained_models/resnet50_best.pth \
    --output ../embeddings/image_embeddings.json

# Pseudo-static embeddings (3,512-d)
echo "  Extracting pseudo-static embeddings..."
cd $SLURM_SUBMIT_DIR/malware_embedding_generator
python pseudo_static_extractor.py embed \
    --input $SAMPLES_DIR \
    --output ../embeddings/pseudo_static_embeddings.json

# Dynamic embeddings (1,000-d)
echo "  Extracting dynamic embeddings..."
python anyrun_extractor.py batch \
    --sha-list $SHA_LIST \
    --output ../embeddings/behavioral_features.json

python dynamic_embedding_generator.py from-features \
    --features ../embeddings/behavioral_features.json \
    --output ../embeddings/dynamic_embeddings.json

# Phase 3: Phylogenetic analysis
echo ""
echo "Phase 3: Building phylogenetic tree..."
cd $SLURM_SUBMIT_DIR/phylogenetic_tree

# Fuse embeddings
python embedding_fusion.py \
    --pseudo-static ../embeddings/pseudo_static_embeddings.json \
    --dynamic ../embeddings/dynamic_embeddings.json \
    --image ../embeddings/image_embeddings.json \
    --output fused_embeddings.json

# Outlier detection
python outlier_detection.py \
    --embeddings fused_embeddings.json \
    --families family_labels.json \
    --output-report ../results/outlier_report.json

# Distance matrix and tree
python generate_distance_matrix.py
python convert_to_pyphil_format.py

# Build tree
if command -v rapidnj &> /dev/null; then
    rapidnj final_distance_matrix.phy -i pd -o t -c 20 > ../results/tree_nj.nwk
else
    python tree_io.py build final_distance_matrix.phy --method nj --output ../results/tree_nj.nwk
fi

# Validation
python temporal_validation.py \
    --tree ../results/tree_nj.nwk \
    --timestamps timestamps.json \
    --families family_labels.json \
    --output ../results/validation_results.json

# Inter-family analysis
python inter_family_analysis.py \
    --tree ../results/tree_nj.nwk \
    --output ../results/inter_family_analysis.json

# Drift analysis
python embedding_drift.py \
    --embeddings fused_embeddings.json \
    --timestamps timestamps.json \
    --families family_labels.json \
    --output ../results/drift_analysis.json

echo ""
echo "========================================"
echo "Pipeline completed at: $(date)"
echo "Results saved in: results/"
echo "========================================"
