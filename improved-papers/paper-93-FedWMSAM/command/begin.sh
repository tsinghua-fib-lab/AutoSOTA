CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 10 --method FedLESAM_S 2>&1 | tee -a  result/12.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method FedCM 2>&1 | tee -a  result/13.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method SCAFFOLD 2>&1 | tee -a  result/14.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method FedSAM 2>&1 | tee -a  result/15.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method FedGamma 2>&1 | tee -a  result/16.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method FedSMOO 2>&1 | tee -a  result/17.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --local-epochs 20 --method FedLESAM_S 2>&1 | tee -a  result/18.txt
CUDA_VISIBLE_DEVICES=2 nohup python train.py --non-iid --total-client 20 --method FedCM 2>&1 | tee -a  result/19.txt
