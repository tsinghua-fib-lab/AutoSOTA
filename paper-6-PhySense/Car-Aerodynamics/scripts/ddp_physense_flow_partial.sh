export CUDA_VISIBLE_DEVICES='2'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1 --master_port 10081 ddp_main_partial_obs.py \
--cfd_model=physense_transolver_car \
--data_dir /data/physense_car_data/ \
--nb_epochs 301 \
--save_freq 30 
