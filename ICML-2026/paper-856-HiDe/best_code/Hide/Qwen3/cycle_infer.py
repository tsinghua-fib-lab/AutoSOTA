import os
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
import json
import torch.multiprocessing as mp
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from PIL import Image
import io
import numpy as np
import base64
import gc
import base64
import multiprocessing
from multiprocessing import Pool
from accelerate import infer_auto_device_map, dispatch_model
import shutil
from inference import cycle_epoch_infer
from utiles import *
import traceback
import subprocess

Image.MAX_IMAGE_PIXELS = 28000000000

def log_error(e):
    print(f"❌ 异常发生: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")

def get_available_gpus(max_memory_mb=1000, max_gpus=None):
    """
    获取显存占用低于 max_memory_mb 的 GPU 设备 ID 列表，并按占用从小到大排序返回

    Args:
        max_memory_mb: 最大允许显存占用（MB），低于此值才认为是“可用”
        max_gpus: 最多返回几个 GPU，None 表示返回所有符合条件的

    Returns:
        按显存占用升序排列的可用 GPU ID 列表，例如 [2, 0, 3]
    """
    try:
        # 使用 nvidia-smi 获取每张 GPU 的显存使用情况
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        # 解析显存使用量（MB）
        used_memory = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        
        # 创建 (gpu_id, memory_used) 的列表并按显存使用量升序排序
        gpu_memory_pairs = [(i, mem) for i, mem in enumerate(used_memory)]
        gpu_memory_pairs.sort(key=lambda x: x[1])  # 按显存使用量从小到大排序
        
        # 筛选低于阈值的 GPU，并保留排序顺序
        available_gpus = [gpu_id for gpu_id, mem in gpu_memory_pairs if mem < max_memory_mb]
        
        # 限制返回数量
        if max_gpus is not None:
            available_gpus = available_gpus[:max_gpus]
        
        return available_gpus

    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
        return []

def main(datasetdir,savedir,max_pixels,Parallels,sig,thre,para_nums=6):
    if not Parallels: para_nums = 1
    dataset = load_dataset_Vstar_json(datasetdir)
    # dataset = load_dataset_hrbench(datasetdir)
    random.shuffle(dataset)
    available_gpus = get_available_gpus(max_memory_mb=1000, max_gpus=para_nums)
    if len(available_gpus) == 0:
        print("❌ 没有找到符合条件的空闲 GPU（占用显存 < 1000MB")
        return
    print(f"✅ 找到 {len(available_gpus)} 个可用 GPU（占用显存 < 1000MB）: {available_gpus}")
    # 分割数据集到不同 GPU 上
    # 将 dataset 划分为 num_gpus 份，每份尽量均衡
    splits = np.array_split(dataset, len(available_gpus))
    print("文件加载完成")
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(gpu_id,rank,dataset_part,savedir,max_pixels,sig,thre)
    else:
        pool = Pool(processes=len(available_gpus))
        results = []
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            res = pool.apply_async(
                cycle_epoch_infer,
                args=(gpu_id,rank, dataset_part, savedir, max_pixels,sig,thre),
                error_callback=log_error
            )
            results.append(res)
        pool.close()
        # 等待并获取结果（可选：获取返回值）
        for res in tqdm(results, desc="等待所有进程完成"):
            res.wait()  # 触发 error_callback
        pool.join()

if __name__ == "__main__":
    # 👇 必须放在这里！
    mp.set_start_method('spawn', force=True)
    maxp = 16384
    #并行多开线程计算，自动寻找满足条件的GPU
    Parallels = True
    #超参数
    sigma = [3]
    threshold = [0.5]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    datasetdir = f"Vstar.json"
    savejson = f"direct_Vstar_results.json"
    # datasetdir = "/data/oss_bucket_0/shijiu/HR_Bench/hr_bench_8k.tsv"
    # savejson = f"HR8k_results.json"
    main(datasetdir,savejson,maxp,Parallels,sigma,threshold,4)
