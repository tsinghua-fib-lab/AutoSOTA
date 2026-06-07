accelerate launch src/inference_HiDRA.py \
    --sd_path="./stabilityai/sd-turbo" \
    --dem_path="checkpoints/DEM.pkl" \
    --pretrained_path="checkpoints/model.pkl" \
    --scale=1 \
    --lq_path="./datasets/HM-TIR/FPNC/deg" \
    --enable_xformers_memory_efficient_attention \
    --output_dir="./datasets/HM-TIR/FPNC/ours"
