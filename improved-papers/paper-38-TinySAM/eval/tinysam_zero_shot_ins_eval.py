import os,sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#from lvis.eval import LVISEval
sys.path.append("..")
import pycocotools.mask as mask_util
import json
import sys
from tinysam import sam_model_registry, SamPredictor
import argparse

def eval_zero_shot(eval_type,val_img_path,val_json_path,vit_det_file_path,sam_checkpoint_path):
    if eval_type=='coco':
        print("============== Evaluating on COCO dataset:",sam_checkpoint_path)
    elif eval_type=='lvis':
        print("============== Evaluating on LVIS dataset:",sam_checkpoint_path)
    else: 
        print("Error! Unsupported evaluation dataset!")
        return
        
    with open(vit_det_file_path) as f:
         res = json.load(f)
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    pre_img_id=0
    total_time=0
    print("Total instances num:",len(res))

    for i,res_ins in enumerate(res):
        res_ins=res[i]
        if i%10000==0:
            print(i)
        img_id=res_ins['image_id']
        img_file_name=f'{img_id:012d}'+'.jpg'

        if pre_img_id!=img_id:
            image = cv2.imread(val_img_path+img_file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            pre_img_id=img_id

        input_box=res_ins['bbox']
        input_box=[input_box[0],input_box[1],input_box[0]+input_box[2],input_box[1]+input_box[3]]
        input_box = np.array(input_box)
        masks, ious, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
        )
        
        new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
        new_seg["counts"] = new_seg["counts"].decode("utf-8")
        res[i]["segmentation"]=new_seg

    for c in res:
         c.pop("bbox", None)
    save_res_json_file=eval_type+'_res_tinysam.json'
    
    with open(save_res_json_file, 'w') as fnew:
        json.dump(res, fnew)
        
    if eval_type=='coco':
        cocoGT= COCO(val_json_path)
        coco_dt = cocoGT.loadRes(res)
        coco_eval = COCOeval(cocoGT, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return
    elif eval_type=='lvis':
        lvis_eval = LVISEval(val_json_path, save_res_json_file, "segm")
        lvis_eval.run()
        lvis_eval.print_results()
        return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='zero shot instance segmentation on COCO or lvis')
    parser.add_argument('--eval_type', type=str, default='coco', help='coco or lvis')
    parser.add_argument('--val_img_path', type=str, default="data/coco/val2017/", help='path to validation imgs')
    parser.add_argument('--val_json_path', type=str, default="json_files/instances_val2017.json", help='path to val2017 annotation json file')
    parser.add_argument('--vit_det_file_path', type=str, default="json_files/coco_instances_results_vitdet.json", help='path to vitdet detection results json file')
    parser.add_argument('--sam_checkpoint_path', type=str, default="../weights/tinysam.pth", help='path to ckpt file')
    
    args = parser.parse_args()
    eval_zero_shot(eval_type=args.eval_type,val_img_path=args.val_img_path,val_json_path=args.val_json_path,vit_det_file_path=args.vit_det_file_path,sam_checkpoint_path=args.sam_checkpoint_path)
    