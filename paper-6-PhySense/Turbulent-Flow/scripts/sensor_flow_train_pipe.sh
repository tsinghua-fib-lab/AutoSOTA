export CUDA_VISIBLE_DEVICES='0,1,2'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1 --master_port 10088 main.py \
    --config sensor_rect_pipe.yml\
    --exp . \
    --doc physense_for_pipe\
    --ni \
    --runner exp_senseiver_rect_flow_pipe