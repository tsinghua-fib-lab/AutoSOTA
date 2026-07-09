train_lo#!/bin/sh

#使用 Conda 的 hook 在当前 shell 初始化环境
eval "$(/root/miniconda3/bin/conda shell.posix hook 2> /dev/null)"

# 激活指定环境
conda activate iter

#环境变量设置
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
export HF_ENDPOINT=https://hf-mirror.com

#运行 Python 程序并保存日志
# 多保真 3_fi
# python ./train/train.py --mode 3_fi --ccsd_num 100 >./log/3_fi_ccsd100.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 200 >./log/3_fi_ccsd200.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 500 >./log/3_fi_ccsd500.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1000 >./log/3_fi_ccsd1000.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 1.0 >./log/3_fi_ccsd1500alpha_rank1.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 5.0 >./log/3_fi_ccsd1500alpha_rank5.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 10.0 >./log/3_fi_ccsd1500alpha_rank10.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 >./log/3_fi_ccsd1500alpha_rank20.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 30.0 >./log/3_fi_ccsd1500alpha_rank30.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 40.0 >./log/3_fi_ccsd1500alpha_rank40.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 60.0 >./log/3_fi_ccsd1500alpha_rank60.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 80.0 >./log/3_fi_ccsd1500alpha_rank80.log 2>&1
python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 10 >./log/3_fi_ccsd1500alpha_rank20alpha_tau10.log 2>&1
python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 20 >./log/3_fi_ccsd1500alpha_rank20alpha_tau20.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 5 >./log/3_fi_ccsd1500alpha_rank20alpha_tau5.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 2 >./log/3_fi_ccsd1500alpha_rank20alpha_tau2.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 1 >./log/3_fi_ccsd1500alpha_rank20alpha_tau1.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 0.5 >./log/3_fi_ccsd1500alpha_rank20alpha_tau5e-1.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 0.1 >./log/3_fi_ccsd1500alpha_rank20alpha_tau1e-1.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 0.05 >./log/3_fi_ccsd1500alpha_rank20alpha_tau5e-2.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 20.0 --alpha_tau 0.01 >./log/3_fi_ccsd1500alpha_rank20alpha_tau1e-2.log 2>&1


# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 50.0 >./log/3_fi_ccsd1500alpha_rank50.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 100.0 >./log/3_fi_ccsd1500alpha_rank100.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 500.0 >./log/3_fi_ccsd1500alpha_rank500.log 2>&1
# python ./train/train.py --mode 3_fi --ccsd_num 1500 --alpha_rank 1000.0 >./log/3_fi_ccsd1500alpha_rank1000.log 2>&1


# python ./train/train.py --mode 2_fi --ccsd_num 10 >./log/2_fi_ccsd10_ratio100.log 2>&1
# python ./train/train.py --mode 2_fi --ccsd_num 50 >./log/2_fi_ccsd50_ratio100.log 2>&1
# python ./train/train.py --mode 2_fi --ccsd_num 500 >./log/2_fi_ccsd500.log 2>&1
# python ./train/train.py --mode 2_fi --ccsd_num 1000 >./log/2_fi_ccsd1000.log 2>&1
# python ./train/train.py --mode 2_fi --ccsd_num 1500 >./log/2_fi_ccsd1500.log 2>&1

# python ./train/train.py --mode 1_fi --ccsd_num 100 >./log/1_fi_ccsd100.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 200 >./log/1_fi_ccsd200.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 500 >./log/1_fi_ccsd500.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 1000 >./log/1_fi_ccsd1000.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 1500 >./log/1_fi_ccsd1500.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 3000 >./log/1_fi_ccsd3000.log 2>&1
# python ./train/train.py --mode 1_fi --ccsd_num 5000 >./log/1_fi_ccsd5000.log 2>&1
# MP2 迁移学习
# python ./train/train.py --mode mp2_transfer --ccsd_num 10 >./log/mp2_transfer_ccsd10_ratio100.log 2>&1
# python ./train/train.py --mode mp2_transfer --ccsd_num 50 >./log/mp2_transfer_ccsd50_ratio100.log 2>&1
# python ./train/train.py --mode mp2_transfer --ccsd_num 200 >./log/mp2_transfer_ccsd200.log 2>&1
# python ./train/train.py --mode mp2_transfer --ccsd_num 500 >./log/mp2_transfer_ccsd500.log 2>&1
# python ./train/train.py --mode mp2_transfer --ccsd_num 1000 >./log/mp2_transfer_ccsd1000.log 2>&1
# python ./train/train.py --mode mp2_transfer --ccsd_num 1500 >./log/mp2_transfer_ccsd1500.log 2>&1

#  HF 迁移学习
# python ./train/train.py --mode hf_transfer --ccsd_num 10 >./log/hf_transfer_ccsd10_ratio100.log 2>&1
# python ./train/train.py --mode hf_transfer --ccsd_num 50 >./log/hf_transfer_ccsd50_ratio100.log 2>&1
# python ./train/train.py --mode hf_transfer --ccsd_num 200 >./log/hf_transfer_ccsd200.log 2>&1
# python ./train/train.py --mode hf_transfer --ccsd_num 500 >./log/hf_transfer_ccsd500.log 2>&1
# python ./train/train.py --mode hf_transfer --ccsd_num 1000 >./log/hf_transfer_ccsd1000.log 2>&1
# python ./train/train.py --mode hf_transfer --ccsd_num 1500 >./log/hf_transfer_ccsd1500.log 2>&1


