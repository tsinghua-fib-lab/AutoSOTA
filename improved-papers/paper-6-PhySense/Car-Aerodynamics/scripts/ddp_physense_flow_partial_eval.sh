export CUDA_VISIBLE_DEVICES='4'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1 --master_port 10074 ddp_main_partial_obs_eval.py \
--cfd_model=physense_transolver_car \
--data_dir /data/physense_car_data/ \
--model_path /workspace/mayuezhou/physense_cr_ckpts/car/checkpoints/physense_transolver_car_base.pth \
--sensor_num 15

