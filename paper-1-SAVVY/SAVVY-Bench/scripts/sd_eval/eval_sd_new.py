"""
Snapshot Descriptor (SD) evaluation code of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import json
import numpy as np


def time_to_second(time_str):
    time_parts = time_str.split(":")
    time_in_seconds = 0
    for idx, time_p in enumerate(time_parts):
        time_in_seconds += 60**(len(time_parts)-1-idx) * float(time_p)
    return int(time_in_seconds)

def abs_dist_norm(flag, pred, target, iou_flag=True):
    """
    flag overlap or distance
    Calculate normalized absolute distance between prediction and target.
    - For 3D coordinates (lists with 3 numbers): Calculate Euclidean distance
    - For time ranges (lists with 2 numbers): Calculate overlap
    - For single numbers: Calculate normalized absolute difference
    """
    try:
        # Handle 3D coordinates (lists with 3 numbers)
        if flag == "distance":
            if isinstance(target, list):
                pred = pred[:len(target)]
                euclidean_dist = np.sqrt(sum((float(p) - float(t))**2 for p, t in zip(pred, target)))
                # Normalize by the magnitude of the target vector
                target_magnitude = np.sqrt(sum(float(t)**2 for t in target))
                return euclidean_dist / target_magnitude
            else:
                if isinstance(pred, list):
                    pred = pred[0]
                # Return normalized absolute difference
                return abs(float(pred) - float(target)) / float(target)
        else:
            # Convert to float to ensure proper calculation
            p_start, p_end = float(pred[0]), float(pred[1])
            t_start, t_end = float(target[0]), float(target[1])
            intersection = max(0, min(p_end, t_end) - max(p_start, t_start))
            pred_length = p_end - p_start
            target_length = t_end - t_start
            if iou_flag:
                union = pred_length + target_length - intersection
            else:
                union = target_length
            if union > 0:
                iou = (intersection / union)
            else:
                iou = 0.0
            return 1 - iou
    except:
        return 0.0


def mean_relative_accuracy(flag, pred, target, start, end, interval, iou_flag=True):
    """
    flag overlap or distance
    Calculate mean relative accuracy over confidence intervals.
    Works with 3D coordinates, time ranges, and single numbers.
    """
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)

    # Calculate normalized distance based on the type of data
    distance = abs_dist_norm(flag, pred, target, iou_flag)

    # Calculate accuracy for each confidence interval
    accuracy = distance <= 1 - conf_intervs


    return accuracy.mean()


gt_sd_file = "data/spatial_avqa/test/gtavmap.json"
gt_data = json.load(open(gt_sd_file, "r"))


tar_folder = "data/sd_output_v2"
methods = ["format_gemini25_flash", "format_gemini25","format_minicpm", "format_ola", "format_videollama_new", "format_longvale", "format_egogpt"]
valid_ids = set()
for doc in json.load(open("data/spatial_avqa/test/savvy_bench.json", "r")):
    valid_ids.add(doc["id"])
for method in methods:
    tar_file = f"{tar_folder}/{method}.json"
    tar_data = json.load(open(tar_file, "r"))
    time_data = tar_data
        
    time_mra_list = []
    iou_time_mra_list = []
    obj_ref_scores_dict = {}
    time_invalid_count = 0
    ego_acc = []
    referral_acc = []
    for idx, cur_gt_data in enumerate(gt_data):
        if cur_gt_data["id"] not in valid_ids:
            continue

        cur_ref_flag = 1
        gt_start_time = time_to_second(cur_gt_data["gt_context_json"]["time range"]["start_time"])
        gt_end_time = time_to_second(cur_gt_data["gt_context_json"]["time range"]["end_time"])
        # time eval
        try:
            pred_start_time = time_to_second(time_data[cur_gt_data["id"]]["start_time"])
            pred_end_time = time_to_second(time_data[cur_gt_data["id"]]["end_time"])
            # print((pred_start_time, pred_end_time), (gt_start_time, gt_end_time))
            time_invalid = False 
        except:
            time_invalid = True
            time_invalid_count += 1

        if time_invalid is False:

            iou_time_mra = mean_relative_accuracy("overlap", (pred_start_time, pred_end_time), (gt_start_time, gt_end_time), 0.05, 0.5, 0.05, iou_flag=True)   
            time_mra = mean_relative_accuracy("overlap", (pred_start_time, pred_end_time), (gt_start_time, gt_end_time), 0.05, 0.5, 0.05, iou_flag=False)   
        else:
            iou_time_mra = 0
            time_mra = 0
        time_mra_list.append(time_mra)  
        iou_time_mra_list.append(iou_time_mra)  
    

        # object class acc
        obj_ref_scores_dict.setdefault("sounding_object", {"tp": 0, "fp": 0, "fn": 0, "count": 0})
        obj_ref_scores_dict.setdefault("reference_object", {"tp": 0, "fp": 0, "fn": 0, "count": 0})
        obj_ref_scores_dict.setdefault("facing_object", {"tp": 0, "fp": 0, "fn": 0, "count": 0})
        obj_ref_scores_dict.setdefault("all", {"tp": 0, "fp": 0, "fn": 0, "count": 0})


        if cur_gt_data["id"] not in tar_data:
            obj_ref_scores_dict["sounding_object"]["fp"] += 1
            if "exo" in cur_gt_data["id"]:
                obj_ref_scores_dict["reference_object"]["fp"] += 1
            if "exo" in cur_gt_data["id"] and "direction" in cur_gt_data["id"]:
                obj_ref_scores_dict["facing_object"]["fp"] += 1
            ego_acc.append(0)
            cur_ref_flag = 0
            referral_acc.append(0)
            continue
        cur_tar_data = tar_data[cur_gt_data["id"]]
        if isinstance(cur_tar_data, list):
            if len(cur_tar_data) > 0:
                cur_tar_data = cur_tar_data[0]
            else:
                cur_tar_data = {}
        
       
        if "mode" in cur_tar_data and "ego" in cur_gt_data["id"] and cur_tar_data["mode"] == "egocentric":
            ego_acc.append(1)
        elif "mode" in cur_tar_data and  "exo" in cur_gt_data["id"] and cur_tar_data["mode"] == "allocentric":
            ego_acc.append(1)
        else:
            ego_acc.append(0)
            cur_ref_flag = 0
        try:
            is_same = cur_tar_data["judge_res"]["sounding_object"]["status"] == "correct"
            if is_same:
                obj_ref_scores_dict["sounding_object"]["tp"] += 1
                obj_ref_scores_dict["all"]["tp"] += 1
            else:
                cur_ref_flag = 0
                obj_ref_scores_dict["sounding_object"]["fp"] += 1
                obj_ref_scores_dict["all"]["fp"] += 1
        except:
            cur_ref_flag = 0
            obj_ref_scores_dict["sounding_object"]["fp"] += 1
            obj_ref_scores_dict["all"]["fp"] += 1

        obj_ref_scores_dict["sounding_object"]["count"] += 1
        obj_ref_scores_dict["all"]["count"] += 1

        if "exo" in cur_gt_data["id"]:
            try:
                gt_ref_object = cur_gt_data["gt_context_json"]["reference_object"]["description"]
                if "stand_by_object" in cur_tar_data:
                    pred_ref_object = cur_tar_data["stand_by_object"]["object_name"]
                else:
                    pred_ref_object = cur_tar_data["reference_object"]["object_name"]
                is_same = cur_tar_data["judge_res"]["reference_object"]["status"] == "correct"
                if is_same:
                    obj_ref_scores_dict["reference_object"]["tp"] += 1
                    obj_ref_scores_dict["all"]["tp"] += 1
                else:
                    cur_ref_flag = 0
                    obj_ref_scores_dict["reference_object"]["fp"] += 1
                    obj_ref_scores_dict["all"]["fp"] += 1
            except:
                if "reference_object" in cur_gt_data["gt_context_json"]:
                    cur_ref_flag = 0
                    obj_ref_scores_dict["reference_object"]["fp"] += 1
                    obj_ref_scores_dict["all"]["fp"] += 1
            obj_ref_scores_dict["reference_object"]["count"] += 1
            obj_ref_scores_dict["all"]["count"] += 1

        if "exo" in cur_gt_data["id"] and "direction" in cur_gt_data["id"]:
            try:
                
                if "facing_direction" in cur_tar_data:
                    pred_facing_object = cur_tar_data["facing_direction"]["object_name"]
                else:
                    pred_facing_object = cur_tar_data["facing_object"]["object_name"]

                gt_facing_object = cur_gt_data["gt_context_json"]["reference_forward_vector"]["description"].replace("pointing from reference object to ", "")

                is_same = cur_tar_data["judge_res"]["facing_object"]["status"] == "correct"
                if is_same:
                    obj_ref_scores_dict["facing_object"]["tp"] += 1
                    obj_ref_scores_dict["all"]["tp"] += 1
                else:
                    cur_ref_flag = 0
                    obj_ref_scores_dict["facing_object"]["fp"] += 1
                    obj_ref_scores_dict["all"]["fp"] += 1
            except:
                if "reference_forward_vector" in cur_gt_data["gt_context_json"]:
                    cur_ref_flag = 0
                    obj_ref_scores_dict["facing_object"]["fp"] += 1
                    obj_ref_scores_dict["all"]["fp"] += 1
            obj_ref_scores_dict["facing_object"]["count"] += 1
            obj_ref_scores_dict["all"]["count"] += 1

        referral_acc.append(cur_ref_flag)



    print("================")
    print(method)
    print("time iou acc: ", round(100*np.mean(iou_time_mra_list), 3), "; count: ", len(time_mra_list))
    print("referral acc: ", round(100*np.mean(referral_acc), 3),  "; count: ", len(referral_acc))
    