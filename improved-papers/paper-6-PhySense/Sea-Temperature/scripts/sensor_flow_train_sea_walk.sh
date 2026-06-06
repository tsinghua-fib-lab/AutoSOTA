export CUDA_VISIBLE_DEVICES='0,1,3,4'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 4 --master_port 10086 main.py \
    --config sensor_rect_sea_walk.yml\
    --exp . \
    --doc physense_for_sea_walk_lr_1_cos1epoch_15sensor \
    --ni \
    --runner exp_senseiver_rect_flow_sea_walk