import os
from transformers import AutoTokenizer, AutoProcessor
from modeling_qwen3_vl_re_infer import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
from utiles import *
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
from Get_box import messages2out,messages2att,from_img_and_att_get_cropbox,get_inputs
import shutil
import cv2

def process_layer_att(start_k, end_k, attention, inputs, dicts, img_url, img_start, img_end):
    # maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    accept_att = {}
    noise_token_num = 8
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]
    for k in range(start_k,end_k):
        max_att_sum = 0
        for img_idx in range(len(inputs["image_grid_thw"])):
            image_grid_thw = inputs["image_grid_thw"][img_idx]
            start = img_start[img_idx]
            end = img_end[img_idx]
            if start_k < end:
                start_k = end+1
            layer_sum = []
            layer_mean = []
            for i in range(len(attention)):
                k_att_map = np.array([row[k] for row in attention[i][0]])
                att_map = k_att_map[:,start:end].reshape(-1, image_grid_thw[1]//2,image_grid_thw[2]//2).mean(axis=0)
                layer_mean.append(att_map)
            if img_idx not in accept_att: accept_att[img_idx] = {}
            accept_att[img_idx][k] = np.array(layer_mean)
    return accept_att

def get_attention_boundingbox_and_allattention(messages,processor,attention, dicts, img_url, img_start, img_end, boundingboxs):
    print(boundingboxs)
    img = Image.open(img_url[0])
    img_w, img_h = img.size
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    end_ques = len(inputs['input_ids'][0])
    print(len(attention))
    tmp_att = []
    for i in range(len(attention)):
        if attention[i] is None:
            continue
        tmp_att.append(attention[i])
    attention = tmp_att
    # print(len(attention))
    start_k = img_end[-1]+1
    # start_k = end_ques-6
    # for i in range(len(inputs['input_ids'][0])):
    #     print(i,inputs['input_ids'][0][i],end="||")
    end_k = len(inputs['input_ids'][0])
    # process(start_k, end_k, attention, inputs, dicts, img_url, img_start, img_end)
    # print(inputs["image_grid_thw"])
    accept_att = process_layer_att(start_k, end_k, attention, inputs, dicts, img_url, img_start, img_end)
    # print(accept_att)
    norm_box = []
    for box in boundingboxs:
        x0,y0,h,w = box
        x1,y1 = x0+w,y0+h
        norm_box.append([x0/img_w,y0/img_h,x1/img_w,y1/img_h])
        # norm_box.append([x0/img_w,y0/img_h,x1/img_w,y1/img_h])
    #layer H W
    # print(inputs['input_ids'][0][-4:])
    target_att = 0
    for i in range(start_k+8,end_k-2):
        target_att += (accept_att[0][i]-accept_att[0][i].min())/(accept_att[0][i].max()-accept_att[0][i].min())
    target_att = target_att/(end_k-2-(start_k+8))
    # print(target_att.shape,norm_box)
    L,H,W = target_att.shape
    layer_bounding_att_mean = []
    layer_att_mean = []
    layer_not_bounding_att_mean = []
    for box in norm_box:
        norm_x0,norm_y0,norm_x1,norm_y1 = box
        x0,y0,x1,y1 = round(norm_x0*W),round(norm_y0*H),round(norm_x1*W),round(norm_y1*H)
        print(box,x0,y0,x1,y1,target_att.shape)
        for ly in range(L):
            layer_bounding_att_mean.append(target_att[ly][y0:y1,x0:x1].mean())
            layer_att_mean.append(target_att[ly].mean())
            # #除了boundingbox区域的attmean
            layer_not_bounding_att_mean.append((target_att[ly].sum()-target_att[ly][y0:y1,x0:x1].sum())/(target_att[ly].shape[0]*target_att[ly].shape[1]-(y1-y0)*(x1-x0)))

    return np.array(layer_bounding_att_mean),np.array(layer_att_mean),np.array(layer_not_bounding_att_mean)

def once_infer(model,qwen_processor,sample,messages,img_url,ori_img_url,ques,sig,thre):
    prompt_ques = '''
You are a highly precise language analysis engine. Your sole function is to extract entities (e.g., objects, people) from a user's question, and deconstruct them into a canonical, attribute-based format by strictly following a set of rules and a thinking process.

### Thinking Process
Before generating the final output, you must internally follow these steps in order:

1.  **Identify Core Entities**: Read the entire question and identify all key noun phrases. For example, "the green surfboard," "the purple umbrella."
2.  **Deconstruct Attributes for Each Entity Individually**: **This is the critical step. Before considering the relationship between entities, look at each entity in isolation and apply Rules 2, 3, and 4 to fully deconstruct its attributes.**
    *   For instance, first process "the green surfboard" using Rule 2 to get `surfboard with green color`.
    *   Then, process "the purple umbrella" using Rule 2 to get `umbrella with purple color`.
3.  **Handle Relationships Between Entities**: After all entities have been individually deconstructed, check for spatial or logical relationships between them (Rule 5). If a relationship exists, you will list the **already deconstructed** entities as separate items.
4.  **Assemble and Normalize**:
    *   Gather all the canonical entity strings you have transformed.
    *   Convert all text to lowercase.
    *   Join all entities into a single line, separated by a comma and a space (", ").
5.  **Final Formatting**: Enclose the resulting single-line string within the `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.

### Extraction Rules

**Rule 1: Simple Entities**
If a noun is not described by any modifiers, extract the noun itself.
*   *Example*: "the scooter" becomes `scooter`.

**Rule 2: Adjective Attribute Deconstruction**
If an entity is modified by one or more adjectives, the format must be `noun with [property] [type]`. Chain multiple properties consecutively.
*   **Format**: `noun with [property1] [type1] with [property2] [type2]`
*   **Common Type Mappings**:
    *   Colors (red, blue, black, silver) -> `color`
    *   Sizes (large, small, big) -> `size`
    *   Materials (wooden, metal, plastic) -> `material`
*   *Example*: "the large blue truck" becomes `truck with large size with blue color`.

**Rule 3: Possessive Inversion**
Convert all possessive forms (e.g., `X's Y`) uniformly into the `Y of X` format.
*   *Example*: "the woman's handbag" becomes `handbag of woman`.

**Rule 4: Attributive Prepositional Phrases**
If a prepositional phrase describes a component of an entity (e.g., "in a shirt"), preserve the structure and recursively apply the rules to the entity within the phrase.
*   *Example*: "the man in the green shirt" becomes `man in a shirt with green color`.

**Rule 5: Relational Prepositional Phrases**
If a prepositional phrase describes a relationship between two separate entities (e.g., `next to`, `on the left of`), **extract the entities as separate items, but only after each entity has been fully deconstructed according to the other rules.** Do not include the relational words in the output.
*   *Example 1 (Simple)*: "the dog on the left side of the scooter" becomes `dog, scooter`.
*   **Example 2 (With Attributes)**: **"Is the green surfboard on the left side of the purple umbrella?" becomes `surfboard with green color, umbrella with purple color`.** (Note: The surfboard and umbrella are first deconstructed individually, then listed as separate entities).

**Rule 6: Compound Nouns**
Recognized compound nouns should be treated as a single entity.
*   *Example*: "the traffic light" becomes `traffic light`.

### Output Format Rules
1.  **Sole Output**: The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.
2.  **Single-Line Format**: The output must be a single continuous line of text.
3.  **Delimiter**: Multiple entities must be separated by a comma followed by a space (", ").
4.  **Lowercase**: All output characters must be in lowercase.
5.  **Content Exclusion**: The final entity string must not include articles (a, an, the), question words (what, which, is), or purely relational words (side, next to, of).

Now, following all the rules above, extract the entities from the question below:
{input_text}
'''

    prompt_messages = [{"role": "user","content": [{"type": "text", "text": prompt_ques.format(input_text=ques)}],},]
    text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(prompt_messages,qwen_processor,model)
    prompt_output_text,_ = messages2out(model,qwen_processor,inputs)
    answer_out = prompt_output_text[0].split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    messages[-1]["content"] = messages[-1]["content"][:-1]
    outputs = {}
    
    #如果提取出了实体词
    if answer_out: 
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: "+answer_out})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(model,qwen_processor,inputs)  # Retrieve attention from model outputs
        
                
    #没有提取出实体词
    else:
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques +"\nAnswer with the option's letter from the given choices letter directly."})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(model,qwen_processor,inputs)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(inputs,qwen_processor,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes = results[str(s)][str(t)]
                messages = [ {"role": "user","content": [],},]
                #加上原图
                for img in ori_img_url:
                    messages[-1]["content"].append({"type": "image", "image": img})
                #加上这次处理新出的图
                for h_img in highlight_imgs:
                    messages[-1]["content"].append({"type": "image", "image": h_img})
                #加上问题
                messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter directly."})
                text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
                output_text,_ = messages2out(model,qwen_processor,inputs)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
    return outputs

def cycle_epoch_infer(gpu_id,rank,dataset_part,savedir,max_pixels,sig,thre):
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"

    print(rank,len(dataset_part),device)

    model_path = r"/data/oss_bucket_0/Qwen/Qwen3-VL-8B-Instruct/"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # load_in_8bit=True,
        device_map=device
    )

    qwen_processor = AutoProcessor.from_pretrained(model_path,use_fast=True,min_pixels=256*32*32,max_pixels=max_pixels*32*32)

    for sample in tqdm(dataset_part):
        results = sample
        img_url = [sample["image"]]
        ori_img_url = []
        for img in img_url:
            ori_img_url.append(img)
        messages = [
                {
                    "role": "user",
                    "content": [],
                },
            ]
        for img in img_url:
            messages[-1]["content"].append({"type": "image", "image": img})

        ques = sample["Text"]

        #直接回答
        messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter directly."})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        output_text,end_ques = messages2out(model,qwen_processor,inputs)
        results["answer"] = {}
        results["answer"]["ori"] = output_text[0]
        results["bounding_box"] = {}
        results["prompt_text"] = {}
        torch.cuda.empty_cache()
        
        #先进行att计算，再回答
        outputs = once_infer(model,qwen_processor,sample,messages,img_url,ori_img_url,ques,sig,thre)
        for s in sig:
            for t in thre:
                prompt_output_text,output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes = outputs[str(s)][str(t)]
                results["answer"][f"HiDe_s{s}_t{t}"] = output_text[0]
                results["prompt_text"][f"HiDe"] = prompt_output_text[0]
                results["bounding_box"][f"HiDe_s{s}_t{t}"] = bounding_boxes
        
        #保存答案
        serialize_dict(results,savedir)
        torch.cuda.empty_cache()
        print(savedir)
    del model
    torch.cuda.empty_cache()
