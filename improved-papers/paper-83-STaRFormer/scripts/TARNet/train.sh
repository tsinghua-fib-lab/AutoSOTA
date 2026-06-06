
###
# CLI Input 
# CUDA_VISIBLE_DEVICES=3 python scripts/TARNet/train.py --dataset dkt --epochs 1 --batch 512 --lr 0.001

# STaRFormer config
# CUDA_VISIBLE_DEVICES=3 python scripts/TARNet/train.py --dataset dkt --epochs 50 --batch 512 --lr 0.006104054297174028 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 8 --nhid_task 8 --nhid_tar 8 --instance 2024-10-16_22:03:27

# better 
# CUDA_VISIBLE_DEVICES=3 python scripts/TARNet/train.py --dataset dkt --epochs 200 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 8 --nhid_task 8 --nhid_tar 8 --instance 2024-10-16_22:03:27


# geolife
# CUDA_VISIBLE_DEVICES=3 python scripts/TARNet/train.py --dataset geolife --epochs 50 --batch 64 --lr 0.0011317951703405868 --nlayers 3 --emb_size 32 --nhead 8 --dropout 0.11918441460166762 --nhid 10 --nhid_task 128 --nhid_tar 128 --instance "2024-10-22_15:41:02"





# DKT

# 42
# CUDA_VISIBLE_DEVICES=2 python scripts/TARNet/train.py --dataset dkt --epochs 300 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 128 --nhid_task 128 --nhid_tar 128 --instance 2024-10-16_22:03:27
# Report -- Best Model found at Epoch 22 -- dkt | 42
# Dataset: dkt, Acc: 0.7804974197247706, F05: 0.7840809442032126, F1: 0.7769847779323369, Prec: 0.7922090215889211, Rec: 0.7777025558682206

# 123
# CUDA_VISIBLE_DEVICES=1 python scripts/TARNet/train.py --dataset dkt --epochs 300 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 128 --nhid_task 128 --nhid_tar 128 --instance 2024-10-28_14:34:17 --seed 123
# Report -- Best Model found at Epoch 65 -- dkt | 123
# Dataset: dkt, Acc: 0.7626863532110092, F05: 0.7631147512525431, F1: 0.7611835909155991, Prec: 0.7651798799116525, Rec: 0.7610013490975385

# 0
# CUDA_VISIBLE_DEVICES=1 python scripts/TARNet/train.py --dataset dkt --epochs 200 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 128 --nhid_task 128 --nhid_tar 128 --instance 2024-10-28_17:57:21  --seed 0
# Report -- Best Model found at Epoch 16 -- dkt | 0
# Dataset: dkt, Acc: 0.7848157970183486, F05: 0.7846928510346856, F1: 0.784519343873124, Prec: 0.7848465880019752, Rec: 0.7843820673478974

# 63
# CUDA_VISIBLE_DEVICES=1 python scripts/TARNet/train.py --dataset dkt --epochs 200 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 128 --nhid_task 128 --nhid_tar 128 --instance 2024-11-08_15:27:47  --seed 63
# Report -- Best Model found at Epoch 8 -- dkt | 63
# Dataset: dkt, Acc: 0.7812052035550459, F05: 0.7815242578855581, F1: 0.7801986976630544, Prec: 0.7828949888193677, Rec: 0.7799061836549485

# 2024
# CUDA_VISIBLE_DEVICES=1 python scripts/TARNet/train.py --dataset dkt --epochs 200 --batch 512 --lr 0.001 --nlayers 4 --emb_size 32 --nhead 4 --dropout 0.2772483650222462 --nhid 128 --nhid_task 128 --nhid_tar 128 --instance 2024-11-09_01:09:33  --seed 2024
# Report -- Best Model found at Epoch 20 -- dkt | 2024
# Dataset: dkt, Acc: 0.7935959002293578, F05: 0.7954539779677947, F1: 0.7915655807473639, Prec: 0.799699620112776, Rec: 0.7914139680290023

####### 
# GL
####### 

# 42
# CUDA_VISIBLE_DEVICES=3 python scripts/TARNet/train.py --dataset geolife --epochs 200 --batch 64 --lr 0.0011317951703405868 --nlayers 3 --emb_size 32 --nhead 8 --dropout 0.11918441460166762 --nhid 10 --nhid_task 128 --nhid_tar 128 --instance "2024-10-22_15:41:02" --s3_bucket_path data-phd
# Report -- Best Model found at Epoch 41
# Dataset: geolife, Acc: 0.8795572916666666, F05: 0.8580687420628292, F1: 0.8571128547170341, Prec: 0.8587523011209728, Rec: 0.8557020686633094

# Training time: 1:32:26.988775


######
# TSR
######

########
# AE
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset AE 

#Report -- Best Model found at Epoch 110 -- AE | 42
#Dataset: AE, RMSE: 3.161145332507258, MAE: 2.6923725605010986
#
#Training time: 0:00:38.043554

#######
# BC
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BC 
# 
#Report -- Best Model found at Epoch 19 -- BC | 42
#Dataset: BC, RMSE: 4.073246284799246, MAE: 2.145862579345703

#Training time: 0:31:44.467208
# 

#######
# BPM10
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BPM10

#Report -- Best Model found at Epoch 156 -- BPM10 | 42
#Dataset: BPM10, RMSE: 116.87056726219609, MAE: 77.80028533935547
#
#Training time: 0:20:07.626730

#######
# BPM25
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BPM25

#Report -- Best Model found at Epoch 87 -- BPM25 | 42
#Dataset: BPM25, RMSE: 85.27108083378855, MAE: 57.627262115478516
#
#Training time: 0:15:41.991222


#######
# LFMC
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset LFMC 

#Report -- Best Model found at Epoch 3 -- LFMC | 42
#Dataset: LFMC, RMSE: 41.90506150439944, MAE: 32.499114990234375
#
#Training time: 0:55:17.351319

#######
# IEEEPPG
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset IEEEPPG
# 
# Report -- Best Model found at Epoch 1 -- IEEEPPG | 42
#Dataset: IEEEPPG, RMSE: 31.245048435837198, MAE: 27.055227279663086
#
#Training time: 1:44:42.493427


#######
# C3M
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset C3M --epochs 300 --batch 16 --lr 0.008175595481116134 --nlayers 1 --emb_size 64 --nhead 8 --dropout 0.02918768814904893 --nhid 8 --nhid_task 8 --nhid_tar 8 --task_type regression
#Report -- Best Model found at Epoch 184 -- C3M | 42
#Dataset: C3M, RMSE: 0.060451911857034454, MAE: 0.04462682083249092
#
#Training time: 0:01:01.791133

#######
# FM1
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset FM1 --epochs 300 --batch 16 --lr 0.008217677337031158 --nlayers 3 --emb_size 8 --nhead 2 --dropout 0.022468467095734135 --nhid 32 --nhid_task 32 --nhid_tar 32 --task_type regression
#Report -- Best Model found at Epoch 183 -- FM1 | 42
#Dataset: FM1, RMSE: 0.016893439278290255, MAE: 0.012778697535395622
#
#Training time: 0:06:29.217556

#######
# FM2
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset FM2 --epochs 300 --batch 32 --lr 0.004567739867286611 --nlayers 1 --emb_size 32 --nhead 2 --dropout 0.20981045802268317 --nhid 128 --nhid_task 128 --nhid_tar 128 --task_type regression
#Report -- Best Model found at Epoch 38 -- FM2 | 42
#Dataset: FM2, RMSE: 0.04763704991900016, MAE: 0.03567153215408325
#
#Training time: 0:04:11.454206

#######
# FM3
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset FM3 --epochs 300 --batch 64 --lr 0.008748207247569756 --nlayers 4 --emb_size 8 --nhead 2 --dropout 0.022468467095734135 --nhid 8 --nhid_task 8 --nhid_tar 8 --task_type regression

#Report -- Best Model found at Epoch 5 -- FM3 | 42
#Dataset: FM3, RMSE: 0.047581056483397714, MAE: 0.042988963425159454
#
#Training time: 0:03:55.198349


########
# AR
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset AR --epochs 300 --batch 512 --lr 0.006360155981042922 --nlayers 1 --emb_size 16 --nhead 8 --dropout 0.021298069769953583 --nhid 64 --nhid_task 64 --nhid_tar 64 --task_type regression
#Report -- Best Model found at Epoch 11 -- AR | 42
#Dataset: AR, RMSE: 8.389735680873496, MAE: 2.233999013900757
#
#Training time: 1:17:29.058288


########
# PPGDalia
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset PPG --epochs 300 --batch 256 --lr 0.006400536217192023 --nlayers 1 --emb_size 16 --nhead 4 --dropout 0.028351334413828387 --nhid 8 --nhid_task 8 --nhid_tar 8 --task_type regression
#
#Report -- Best Model found at Epoch 11 -- PPG | 42
#Dataset: PPG, RMSE: 20.7032303947789, MAE: 16.596080780029297
#
#Training time: 4:46:23.336325

########
# NTS
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset NTS --epochs 300 --batch 512 --lr 0.008128926270509111 --nlayers 3 --emb_size 8 --nhead 2 --dropout 0.26851165347243067 --nhid 64 --nhid_task 64 --nhid_tar 64 --task_type regression
#Report -- Best Model found at Epoch 7 -- NTS | 42
#Dataset: NTS, RMSE: 0.13964076121113292, MAE: 0.10172238200902939
#
#Training time: 3:42:32.648706

########
# NHS
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset NHS --epochs 300 --batch 256 --lr 0.004982699550107359 --nlayers 5 --emb_size 64 --nhead 2 --dropout 0.05548395490179367 --nhid 32 --nhid_task 32 --nhid_tar 32 --task_type regression
# Report -- Best Model found at Epoch 1 -- NHS | 42
# Dataset: NHS, RMSE: 0.144022051052107, MAE: 0.11089083552360535
# 
# Training time: 4:34:50.791509

########
# BIDMCSpO2
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BIDMCSpO2 --epochs 300 --batch 16 --lr 0.000780683535957991 --nlayers 6 --emb_size 64 --nhead 4 --dropout 0.2866744223363014  --nhid 64 --nhid_task 64 --nhid_tar 64 --task_type regression

#Report -- Best Model found at Epoch 8 -- BIDMCSpO2 | 42
#Dataset: BIDMCSpO2, RMSE: 6.408703143590442, MAE: 4.244049072265625
#
# #default settings
# Training time: 3:08:51.511215
#Report -- Best Model found at Epoch 1 -- BIDMCSpO2 | 42
#Dataset: BIDMCSpO2, RMSE: 5.23056046146958, MAE: 4.5303120613098145
#
#Training time: 1:46:52.964796

########
# BIDMCHR
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BIDMCHR --epochs 100 --batch 16 --task_type regression
#Report -- Best Model found at Epoch 91 -- BIDMCHR | 42
#Dataset: BIDMCHR, RMSE: 14.829508999228636, MAE: 11.7609224319458
#
#Training time: 1:46:56.005213

# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BIDMCHR --epochs 100 --batch 16 --lr 0.002290211662060926 --nlayers 5 --emb_size 32 --nhead 4 --dropout 0.04327625135250979  --nhid 16 --nhid_task 16 --nhid_tar 16 --task_type regression

#Report -- Best Model found at Epoch 5 -- BIDMCHR | 42
#Dataset: BIDMCHR, RMSE: 14.071632234401038, MAE: 10.296542167663574
#
#Training time: 2:05:38.841053



########
# BIDMCRR
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BIDMCRR --epochs 100 --batch 16 --task_type regression

# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset BIDMCRR --epochs 100 --batch 16 --lr 0.009672163335508115 --nlayers 2 --emb_size 32 --nhead 4 --dropout 0.49171443292546185  --nhid 32 --nhid_task 32 --nhid_tar 32 --task_type regression
#Report -- Best Model found at Epoch 96 -- BIDMCRR | 42
#Dataset: BIDMCRR, RMSE: 3.4872331742790004, MAE: 2.6344645023345947
#
#Training time: 1:15:51.840291


########
# HPC1
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset HPC1 --epochs 100 --batch 16 --task_type regression
#Report -- Best Model found at Epoch 7 -- HPC1 | 42
#Dataset: HPC1, RMSE: 519.4541004747196, MAE: 401.9941711425781
#
#Training time: 0:27:42.693781

# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset HPC1 --epochs 100 --batch 32 --lr 0.008347218156692909 --nlayers 6 --emb_size 16 --nhead 2 --dropout 0.018231824697429976  --nhid 32 --nhid_task 32 --nhid_tar 32 --task_type regression
#Report -- Best Model found at Epoch 37 -- HPC1 | 42
#Dataset: HPC1, RMSE: 521.6485586580299, MAE: 403.70465087890625
#
#Training time: 0:24:38.397744

########
# HPC2
# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset HPC2 --epochs 100 --batch 16 --task_type regression
#Report -- Best Model found at Epoch 3 -- HPC2 | 42
#Dataset: HPC2, RMSE: 53.789661057853394, MAE: 40.21854019165039
#
#Training time: 0:27:54.857603

# CUDA_VISIBLE_DEVICES=0 python scripts/TARNet/train.py --dataset HPC2 --epochs 100 --batch 16 --lr 0.009689160058244353 --nlayers 1 --emb_size 16 --nhead 8 --dropout 0.27273028824537426  --nhid 32 --nhid_task 32 --nhid_tar 32 --task_type regression
#Report -- Best Model found at Epoch 10 -- HPC2 | 42
#Dataset: HPC2, RMSE: 50.91679506342922, MAE: 38.940460205078125
#
#Training time: 0:17:47.943255




