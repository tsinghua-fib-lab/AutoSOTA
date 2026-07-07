import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from utiles_internvl import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import io
import numpy as np
import base64
import gc
import base64
import multiprocessing
from multiprocessing import Pool
from accelerate import infer_auto_device_map, dispatch_model
import shutil
import json
import torch.multiprocessing as mp
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = 28000000000
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
import json
import subprocess

def cycle_epoch_infer(gpu_id,rank,dataset_part,savedir,CoT,cycle_times,sig,thre):
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"
    print(rank,len(dataset_part),device)
    path = 'OpenGVLab/InternVL3-8B-Instruct'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    for ly in range(model.language_model.model.config.num_hidden_layers):
        model.language_model.model.layers[ly].self_attn.forward = types.MethodType(layer_forward, model.language_model.model.layers[ly].self_attn)
    model.language_model.model.forward = types.MethodType(qwen2_forward, model.language_model.model)
    for sample in tqdm(dataset_part):
        results = sample
        # if sample["category"] == "direct_attributes": continue
        # upscale_factor = 2
        pil_img = Image.open(sample["image"]).convert('RGBA')
        # new_size = (round(np.array(pil_img).shape[1] * upscale_factor), round(np.array(pil_img).shape[0] * upscale_factor))
        # resized_img = pil_img.resize(new_size, Image.BILINEAR)
        img_url = [pil_img]
        pixel_values_list = []
        block_indices = []
        for img in img_url:
            pixel_value,block_index = load_image(img, max_num=128)
            pixel_values_list.append(pixel_value)
            block_indices.append(block_index)
        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(device=model.device)
        # print(sample)
        question = '<image>\n' + sample["Text"]
        pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        # output_text,end_ques = messages2out(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        results["answer"] = {}
        # results["answer"]["ori"] = output_text[0]
        results["enetity"] = {}
        if CoT:
            # results["bounding_box"] = {}
            for i in range(cycle_times):
                # results["bounding_box"][f"TAD_{i+1}"] = []
                torch.cuda.empty_cache()
                #先进行att计算，再回答
                # messages.append({"role":"assistant","content":output_text[0]})
                output_text,crop_list,highlight_imgs,pixel_values,block_indices,words_lines,img_merged_boxes,bounding_boxes,prompt_ques = once_cot_infer(model,tokenizer,pixel_values,block_indices,question,generation_config, img_url,sig,thre)
                results["answer"][f"TAD_{i+1}"] = output_text[0]
                results["enetity"][f"TAD_{i+1}"] = prompt_ques
                for highlight_img in highlight_imgs:
                    img_url.append(highlight_img)
                # for boxidx in bounding_boxes:
                    # results["bounding_box"][f"TAD_{i+1}"].append(bounding_boxes[boxidx])
                # for word in img_merged_boxes:
                # #     results["hint"].append(words_lines[word])
                #     for imgidx in img_merged_boxes[word]:
                #         for boxidx in range(len(img_merged_boxes[word][imgidx])):
                #             results["bounding_box"][f"TAD_{i+1}"].append(img_merged_boxes[word][imgidx][boxidx])
                # results["hint"].append(words_lines[-1])
                    # messages.append({"role":"assistant","content":output_text[0]})
                        
        #保存答案
        results.pop("image")
        serialize_dict(results,savedir)
        torch.cuda.empty_cache()
        print(savedir)
    del model
    torch.cuda.empty_cache()

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

def main(datasetdir,savedir,CoT,cycle_times,Parallels,sig,thre,para_nums=6):
    dataset = load_dataset_Vstar_json(datasetdir)
    random.shuffle(dataset)
    # num_gpus = torch.cuda.device_count()
    available_gpus = get_available_gpus(max_memory_mb=96000-40000, max_gpus=para_nums)
    if len(available_gpus) == 0:
        print("❌ 没有找到符合条件的空闲 GPU（剩余显存 > 40000MB）")
        return
    print(f"✅ 找到 {len(available_gpus)} 个可用 GPU（剩余显存 > 40000MB）: {available_gpus}")
    # 分割数据集到不同 GPU 上
    # 将 dataset 划分为 num_gpus 份，每份尽量均衡
    splits = np.array_split(dataset, len(available_gpus))
    print("文件加载完成")
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(gpu_id,rank,dataset_part,savedir,CoT,cycle_times,sig,thre)
    else:
        pool = Pool(processes=len(available_gpus))
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            pool.apply_async(cycle_epoch_infer, args=(gpu_id,rank,dataset_part,savedir,CoT,cycle_times,sig,thre))
        pool.close()
        pool.join()

if __name__ == "__main__":
    # 👇 必须放在这里！
    mp.set_start_method('spawn', force=True)
    maxp = [16384]
    CoT = [True]
    Parallels = True
    cycle_times = 1
    sigma=[1]
    threshold=[0.4]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    save_dir = f'internvl/internvl_results/{formatted_time}'
    create_directory(save_dir)
    for maxpp in maxp:
        for coti in CoT:
            for sig in sigma:
                for thre in threshold:
                    datasetdir = f"Vstar.json"
                    savejson = f'{save_dir}/Vstar-HiDe-internvl3-{cycle_times}-15layer-sig{sig}-thre{thre}-norm.json'
                    main(datasetdir,savejson,coti,cycle_times,Parallels,sig,thre)
