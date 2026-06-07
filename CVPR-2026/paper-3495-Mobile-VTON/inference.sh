# inference for VITON-HD dataset
# Paired
accelerate launch --machine_rank 0 \
    --main_process_ip 0.0.0.0 --main_process_port 20056 --num_machines 1 --num_processes 4 inference.py \
    --data_dir ../VITON-HD \
    --output_dir output/VITON/paired \
    --order paired \
    --height 1024 --width 768 \
    --test_batch_size 16 \
    --num_inference_steps 28 \
    --guidance_scale 2.5 \
    --mixed_precision bf16 \
    --checkpoint_path ./checkpoint

# Unpaired
accelerate launch --machine_rank 0 \
    --main_process_ip 0.0.0.0 --main_process_port 20056 --num_machines 1 --num_processes 4 inference.py \
    --data_dir ../VITON-HD \
    --output_dir output/VITON/unpaired \
    --order unpaired \
    --height 1024 --width 768 \
    --test_batch_size 16 \
    --num_inference_steps 28 \
    --guidance_scale 2.5 \
    --mixed_precision bf16 \
    --checkpoint_path ./checkpoint

# inference for DressCode dataset
# Paired
accelerate launch --machine_rank 0 \
    --main_process_ip 0.0.0.0 --main_process_port 20056 --num_machines 1 --num_processes 4 inference_dc.py \
    --data_dir ../Dress_code \
    --output_dir output/DC/paired \
    --order paired \
    --height 1024 --width 768 \
    --test_batch_size 16 \
    --num_inference_steps 28 \
    --guidance_scale 2.5 \
    --mixed_precision bf16 \
    --file=test_pairs_paired \
    --checkpoint_path ./checkpoint

# Unpaired
accelerate launch --machine_rank 0 \
    --main_process_ip 0.0.0.0 --main_process_port 20056 --num_machines 1 --num_processes 4 inference_dc.py \
    --data_dir ../Dress_code \
    --output_dir output/DC/unpaired \
    --order unpaired \
    --height 1024 --width 768 \
    --test_batch_size 16 \
    --num_inference_steps 28 \
    --guidance_scale 2.5 \
    --mixed_precision bf16 \
    --file=test_pairs_unpaired \
    --checkpoint_path ./checkpoint