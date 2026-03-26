export CUDA_VISIBLE_DEVICES='1'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1 --master_port 10071 ddp_main_partial_obs_eval_walk.py \
--cfd_model=physense_transolver_car_walk \
--base_model_path ./checkpoints/physense_transolver_car_best_base.pth \
--data_dir /data/physense_car_data/ \
--nb_epochs 5 \
--lr 0.0025 \
--sensor_num 15 \
--seed 1 \
--model_path /workspace/mayuezhou/physense_cr_ckpts/car/checkpoints/physense_walk_0.0025_15_this.pth
