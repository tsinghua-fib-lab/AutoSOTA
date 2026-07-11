#!/bin/bash
#SBATCH --job-name=maltree_phylo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/phylo_%j.out
#SBATCH --error=logs/phylo_%j.err

# MalTree Phylogenetic Tree Construction
# Uses RapidNJ for optimized Neighbor-Joining (O(n^2) complexity)

module purge
module load python/3.10

source activate malware_analysis

mkdir -p logs
mkdir -p results

cd $SLURM_SUBMIT_DIR/phylogenetic_tree

echo "Starting phylogenetic analysis at $(date)"

# Step 1: Fuse multi-modal embeddings
echo "Fusing embeddings..."
python embedding_fusion.py \
    --pseudo-static ../embeddings/pseudo_static_embeddings.json \
    --dynamic ../embeddings/dynamic_embeddings.json \
    --image ../embeddings/image_embeddings.json \
    --output fused_embeddings.json

# Step 2: Run outlier detection
echo "Running outlier detection..."
python outlier_detection.py \
    --embeddings fused_embeddings.json \
    --families family_labels.json \
    --iqr-multiplier 1.5 \
    --output-matrix clean_distance_matrix.npy \
    --output-report results/outlier_report.json

# Step 3: Generate distance matrix
echo "Generating distance matrix..."
python generate_distance_matrix.py \
    --input fused_embeddings.json \
    --output distance_matrix.npy

# Step 4: Convert to PHYLIP format
echo "Converting to PHYLIP format..."
python convert_to_pyphil_format.py

# Step 5: Build tree using RapidNJ (must be installed)
echo "Building Neighbor-Joining tree..."
if command -v rapidnj &> /dev/null; then
    rapidnj final_distance_matrix.phy -i pd -o t -c $SLURM_CPUS_PER_TASK > results/tree_nj.nwk
    echo "NJ tree built successfully"
else
    echo "RapidNJ not found. Building tree with Python (slower)..."
    python tree_io.py build final_distance_matrix.phy --method nj --output results/tree_nj.nwk
fi

# Step 6: Validate tree
echo "Validating tree temporal consistency..."
python temporal_validation.py \
    --tree results/tree_nj.nwk \
    --timestamps timestamps.json \
    --families family_labels.json \
    --granularity year \
    --output results/validation_results.json

# Step 7: Inter-family analysis
echo "Running inter-family analysis..."
python inter_family_analysis.py \
    --tree results/tree_nj.nwk \
    --output results/inter_family_analysis.json

echo "Phylogenetic analysis completed at $(date)"
