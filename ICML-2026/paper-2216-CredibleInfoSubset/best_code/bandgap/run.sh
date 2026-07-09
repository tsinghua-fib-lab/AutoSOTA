#!/bin/sh

#使用 Conda 的 hook 在当前 shell 初始化环境
eval "$(/root/miniconda3/bin/conda shell.posix hook 2> /dev/null)"

# 激活指定环境
conda activate subset

#环境变量设置
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
export HF_ENDPOINT=https://hf-mirror.com

#运行 Python 程序并保存日志
python ./train/methodmine_draw.py  >./log/trainminedraw.log 2>&1
# python ./train/methodfi.py  >./log/trainfi_300.log 2>&1
# python ./train/methodmine.py  >./log/trainmine.log 2>&1
# python ./train/methodSE.py  >./log/trainSE.log 2>&1
# python ./train/methodtransfer.py  >./log/traintransfer.log 2>&1

