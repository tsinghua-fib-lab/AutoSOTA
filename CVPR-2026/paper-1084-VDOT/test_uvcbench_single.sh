torchrun --nproc_per_node=4 inference/vace_alltask_uvcbench_single.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \
    --ring_size 1 \
    --size 480p \
    --sample_guide_scale 1 \
    --sample_steps 4 \
    --ckpt_dir VDOT \
    --save_dir results/uvcbench_single