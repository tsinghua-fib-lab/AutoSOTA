"""
Eval utils code for SAVVY pipeline evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""
import numpy as np
import math

from utils.format_utils import find_closest
from evaluation.metric_utils import calculate_azimuth_facevec, _calculate_distance, mean_relative_accuracy
from utils.localization import calculate_centroid
from utils.format_utils import get_fuzzy_match_name

## Audio eval
def eval_srp_audio(fps_audio_track, gt_audio_dict, audio_metrics, valid_angle_abs_range=(5, 175), acc_standard={"angle": 45, "distance": 1.0}):
    if "audio_dir_acc" in audio_metrics:
        audio_dir_acc = audio_metrics["audio_dir_acc"]
    if "audio_angle_err" in audio_metrics:
        audio_angle_err = audio_metrics["audio_angle_err"]
    if "audio_dist_err" in audio_metrics:
        audio_dist_err = audio_metrics["audio_dist_err"]
    if "audio_dir_left_acc" in audio_metrics:
        audio_dir_left_acc = audio_metrics["audio_dir_left_acc"]
    if "audio_dir_right_acc" in audio_metrics:
        audio_dir_right_acc = audio_metrics["audio_dir_right_acc"]
    if "audio_dir_front_acc" in audio_metrics:
        audio_dir_front_acc = audio_metrics["audio_dir_front_acc"]
    if "audio_dir_back_acc" in audio_metrics:
        audio_dir_back_acc = audio_metrics["audio_dir_back_acc"]
    if "audio_loc_acc" in audio_metrics:
        audio_loc_acc = audio_metrics["audio_loc_acc"]

    for audio_key in gt_audio_dict:
        cur_gt_audio = gt_audio_dict[audio_key]
        # test on key gt frames
        if abs(cur_gt_audio["angle"]) <= valid_angle_abs_range[0] or abs(cur_gt_audio["angle"]) >= valid_angle_abs_range[1]:
            continue 
        if "direction" not in cur_gt_audio:
            continue
        if audio_key not in fps_audio_track:
            for key in audio_metrics:
                audio_metrics[key].append(0)
            continue
        pred_gt_audio = fps_audio_track[audio_key]
        if cur_gt_audio["direction"] == pred_gt_audio["direction"]:
            audio_dir_acc.append(1)
        else:
            audio_dir_acc.append(0)
        audio_angle_err.append(abs(cur_gt_audio["angle"]-pred_gt_audio["angle"]))
        audio_dist_err.append(abs(cur_gt_audio["distance"]-pred_gt_audio["distance"]))
        
        if abs(pred_gt_audio["angle"]-cur_gt_audio["angle"]) <= acc_standard["angle"] and abs(pred_gt_audio["distance"]-cur_gt_audio["distance"]) <= acc_standard["distance"]:
            audio_loc_acc.append(1)
        else:
            audio_loc_acc.append(0)

        if ("left" in cur_gt_audio["direction"] and "left" in pred_gt_audio["direction"]):
            audio_dir_left_acc.append(1)
        elif ("left" in cur_gt_audio["direction"] and "right" in pred_gt_audio["direction"]):
            audio_dir_left_acc.append(0)
        if ("right" in cur_gt_audio["direction"] and "right" in pred_gt_audio["direction"]):
            audio_dir_right_acc.append(1)
        elif ("right" in cur_gt_audio["direction"] and "left" in pred_gt_audio["direction"]):
            audio_dir_right_acc.append(0)
        if ("front" in cur_gt_audio["direction"] and "front" in pred_gt_audio["direction"]):
            audio_dir_front_acc.append(1)
        elif ("front" in cur_gt_audio["direction"] and "back" in pred_gt_audio["direction"]):
            audio_dir_front_acc.append(0)
        if ("back" in cur_gt_audio["direction"] and "back" in pred_gt_audio["direction"]):
            audio_dir_back_acc.append(1)
        elif ("back" in cur_gt_audio["direction"] and "front" in pred_gt_audio["direction"]):
            audio_dir_back_acc.append(0)
    return audio_metrics





def eval_vis_track2d_loc(seg_res_for_eval, gt_vis_dict, gt_data, dense_seg_metrics, eval_thr_dict={"sounding_object": 0.5, "reference_object": 0.7, "reference_facing_object": 0.7}, acc_standard={"angle": 45, "distance": 0.5}):
    for key in seg_res_for_eval:
        if key not in gt_data or "description" not in gt_data[key]:
            continue
        tar_name = gt_data[key]["description"]
        if "camera" in tar_name:
            continue
        if key == "facing_object":
            tar_name = tar_name.replace("pointing from the 'reference_object' to ", "")
        if key == "sounding_object" and "person" in tar_name:
            tar_name = "guest"
        for time_key in seg_res_for_eval[key]:
            pred_angle = seg_res_for_eval[key][time_key]["direction"]
            pred_distance = seg_res_for_eval[key][time_key]["distance"]
            det_conf = np.mean(seg_res_for_eval[key][time_key]['det_conf'])
            if det_conf < eval_thr_dict[key]:
                continue
            time_key_int = find_closest([int(key) for key in gt_vis_dict.keys()], int(time_key))
            time_key = str(time_key_int).zfill(5)
            gt_det_item_vis = gt_vis_dict[time_key]
            if tar_name not in gt_det_item_vis:
                new_tar_name = get_fuzzy_match_name(tar_name, [name for name in gt_det_item_vis.keys()], thr=0.15)
                if new_tar_name is None:
                    import pdb; pdb.set_trace()
                else:
                    tar_name = new_tar_name
            gt_loc = gt_det_item_vis[tar_name]["sound_loc"]
            gt_angle = gt_det_item_vis[tar_name]["angle"]
            gt_distance = gt_det_item_vis[tar_name]["distance"]

            if key == "sounding_object":
                cur_dense_seg_metrics = dense_seg_metrics["sounding_object"]
            else:
                cur_dense_seg_metrics = dense_seg_metrics["static_object"]
            if abs(pred_angle-gt_angle) <= acc_standard["angle"] and abs(pred_distance-gt_distance) <= acc_standard["distance"]:    
                cur_dense_seg_metrics["pred_acc"].append(1)
            else:
                cur_dense_seg_metrics["pred_acc"].append(0)

            if pred_distance is None or np.isnan(abs(pred_distance-gt_distance)):
                continue

            cur_dense_seg_metrics["angle_err"].append(abs(pred_angle-gt_angle))
            cur_dense_seg_metrics["distance_err"].append(abs(pred_distance-gt_distance))
             
    return dense_seg_metrics


def eval_ego_qa_acc_avmap(final_traj, cam_traj, ori_step1_res, data, ego_qa_metrics, mra_thr_range=(0.5, 0.95, 0.05)):
    prediction = None
    if "ego" not in data["id"]:
        return ego_qa_metrics, prediction
    if not ori_step1_res["mode"] == "egocentric":
        ego_sd_wrong_flag = True
    else:
        ego_sd_wrong_flag = False
    if ego_sd_wrong_flag:
        print("ego sd wrong reference!")
    if len(final_traj) == 0 or ego_sd_wrong_flag or "loc" not in cam_traj:
        if "direction" in data["id"]:
            ego_qa_metrics["ego_out_mca_acc"].append(0)
            if "simple" in data["id"]:
                ego_qa_metrics["ego_out_mca_simple_acc"].append(0)
            elif "hard" in data["id"]:
                ego_qa_metrics["ego_out_mca_hard_acc"].append(0)
        elif "distance" in data["id"]:
            ego_qa_metrics["ego_out_dist_acc"].append(0)
        return ego_qa_metrics, prediction
    
    pred_sounding_loc = calculate_centroid([value for value in final_traj.values()])
    if "direction" in data["id"]:
        angle_degrees = calculate_azimuth_facevec(cam_traj["loc"], cam_traj["forward_vec"], pred_sounding_loc)
        if len(data["options"]) == 3:
            if abs(angle_degrees) >= 120:
                pred = "C"
            elif angle_degrees < 0:
                pred = "A"
            else:
                pred = "B"
            # simple
            ego_qa_metrics["ego_out_mca_simple_acc"].append(pred==data["gt_context_json"]["prediction"])
        elif len(data["options"]) == 4:
            if abs(angle_degrees) >= 90:
                if angle_degrees < 0:
                    pred = "C"
                else:
                    pred = "D"
            else:
                if angle_degrees < 0:
                    pred = "A"
                else:
                    pred = "B"
            ego_qa_metrics["ego_out_mca_hard_acc"].append(pred==data["gt_context_json"]["prediction"])
        ego_qa_metrics["ego_out_mca_acc"].append(pred==data["gt_context_json"]["prediction"])
        prediction = pred
    elif "distance" in data["id"]:
        pred_distance = _calculate_distance(cam_traj["loc"], pred_sounding_loc)
        final_acc = 0.0
        num_pts = int((mra_thr_range[1] - mra_thr_range[0]) / mra_thr_range[2] + 1)
        thr_list = np.linspace(mra_thr_range[0], mra_thr_range[1], num_pts)
        for thr in thr_list:
            if abs(float(pred_distance)-float(data["gt_context_json"]["prediction"])) < thr:
                final_acc += 1.0
            else:
                final_acc += 0.0
        ego_qa_metrics["ego_out_dist_acc"].append(final_acc / len(thr_list))

        prediction = float(pred_distance)
    return ego_qa_metrics, prediction


def eval_exo_qa_acc_avmap(all_final_traj, cam_traj, ori_step1_res, data, exo_qa_metrics, mra_thr_range=(0.5, 0.95, 0.05)):
    prediction = None
    if "exo" not in data["id"]:
        return exo_qa_metrics, prediction
    if len(all_final_traj["sounding_object"]) == 0 or len(all_final_traj["reference_object"]) == 0:
        if "direction" in data["id"]:
            exo_qa_metrics["exo_out_mca_acc"].append(0)
            if "simple" in data["id"]:
                exo_qa_metrics["exo_out_mca_simple_acc"].append(0)
            elif "hard" in data["id"]:
                exo_qa_metrics["exo_out_mca_hard_acc"].append(0)
        elif "distance" in data["id"]:
            exo_qa_metrics["exo_out_dist_acc"].append(0)
        return exo_qa_metrics, prediction
    
    if len(all_final_traj["facing_object"]) == 0 and "direction" in data["id"]:
        exo_qa_metrics["exo_out_mca_acc"].append(0)
        if "simple" in data["id"]:
            exo_qa_metrics["exo_out_mca_simple_acc"].append(0)
        elif "hard" in data["id"]:
            exo_qa_metrics["exo_out_mca_hard_acc"].append(0)
        return exo_qa_metrics, prediction

    pred_sounding_loc = calculate_centroid([value for value in all_final_traj["sounding_object"].values()])
    pred_reference_loc = np.array(all_final_traj["reference_object"])

    if "direction" in data["id"]:
        pred_facing_loc = np.array(all_final_traj["facing_object"])
        forward_vec = pred_facing_loc - pred_reference_loc
        angle_degrees = calculate_azimuth_facevec(pred_reference_loc, forward_vec, pred_sounding_loc)
        if len(data["options"]) == 3:
            if abs(angle_degrees) >= 120:
                pred = "C"
            elif angle_degrees < 0:
                pred = "A"
            else:
                pred = "B"
            # simple
            exo_qa_metrics["exo_out_mca_simple_acc"].append(pred==data["gt_context_json"]["prediction"])
        elif len(data["options"]) == 4:
            if abs(angle_degrees) >= 90:
                if angle_degrees < 0:
                    pred = "C"
                else:
                    pred = "D"
            else:
                if angle_degrees < 0:
                    pred = "A"
                else:
                    pred = "B"
            exo_qa_metrics["exo_out_mca_hard_acc"].append(pred==data["gt_context_json"]["prediction"])
        exo_qa_metrics["exo_out_mca_acc"].append(pred==data["gt_context_json"]["prediction"])
        prediction = pred
    elif "distance" in data["id"]:
        pred_distance = _calculate_distance(pred_reference_loc, pred_sounding_loc)
        
        final_acc = 0.0
        num_pts = int((mra_thr_range[1] - mra_thr_range[0]) / mra_thr_range[2] + 1)
        thr_list = np.linspace(mra_thr_range[0], mra_thr_range[1], num_pts)
        for thr in thr_list:
            if abs(float(pred_distance)-float(data["gt_context_json"]["prediction"])) < thr:
                final_acc += 1.0
            else:
                final_acc += 0.0
        exo_qa_metrics["exo_out_dist_acc"].append(final_acc / len(thr_list))
        prediction = float(pred_distance)
    return exo_qa_metrics, prediction
