CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedCM 2>&1 | tee -a  result/89.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method SCAFFOLD 2>&1 | tee -a  result/90.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedSAM 2>&1 | tee -a  result/91.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method MoFedSAM 2>&1 | tee -a  result/92.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedGamma 2>&1 | tee -a  result/93.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedSMOO 2>&1 | tee -a  result/94.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedLESAM_S 2>&1 | tee -a  result/95.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --non-iid --seed 10 --method FedWMSAM 2>&1 | tee -a  result/96.txt &
