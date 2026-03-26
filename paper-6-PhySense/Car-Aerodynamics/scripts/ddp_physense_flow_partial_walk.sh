export CUDA_VISIBLE_DEVICES='3'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1 --master_port 10083 ddp_main_partial_obs_walk.py \
--cfd_model=physense_transolver_car_walk \
--base_model_path ./checkpoints/physense_transolver_car_best_base.pth \
--data_dir /data/physense_car_data/ \
--nb_epochs 5 \
--lr 0.0025 \
--sensor_num 30 \
--seed 1