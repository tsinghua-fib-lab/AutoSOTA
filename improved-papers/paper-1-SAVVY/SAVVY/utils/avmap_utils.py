"""
Global map trakcing utils code for SAVVY pipeline stage2 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import math
import os
import json
import numpy as np


from utils.format_utils import second_to_time, time_to_second, na_postprocess
from utils.doa_aria import get_doa_srp_phat_aria
from utils.format_utils import find_closest
from utils.localization import calculate_centroid, get_possible_locs, calculate_weighted_centroid, calculate_centroid_cluster
from utils.track_utils import optimize_trajectory_smoothness, filter_outliers, interpolate_missing_timepoints1d, calculate_distance, predict_trajectory_point
import numpy as np
from sklearn.cluster import DBSCAN


def get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[0, 1, 2, 3, 4, 5, 6], audio_fps=4):
    audio_doa_path = f"{audio_doa_root}/{video_id}/audio_track_results.json"
    if not os.path.exists(audio_doa_path):
        os.makedirs(f"{audio_doa_root}/{video_id}/", exist_ok=True)
        audio_path = f"{audio_root}/{video_id}.wav"
        get_doa_srp_phat_aria(audio_path, selected_indices=selected_indices, fps=audio_fps, output_json_path=audio_doa_path)
    audio_track = json.load(open(audio_doa_path, "r"))
    return audio_track
    
def get_sounding_traj(event_label, anchor_event, object_pool, now_guest_traj_dict, time_range):
    if event_label == "speech":
        shift_time = anchor_event["guest_zero_base_start_time"] - anchor_event['zero_base_start_time']
    else:
        shift_time = 0.0

    gt_start_time, gt_end_time = time_range
    gt_start_time = gt_start_time + shift_time
    gt_end_time =(gt_end_time + shift_time)
    list_len = gt_end_time-gt_start_time+1

    if event_label == "speech":
        # Filter trajectory positions within the given time range
        filtered_traj = {}
        if now_guest_traj_dict is None:
            return None
        for time_key, cur_traj in now_guest_traj_dict.items():
            time_key_int = int(time_key)
            if gt_start_time <= time_key_int/20. <= gt_end_time:
                filtered_traj[time_key_int] = cur_traj["loc"]
        if len(filtered_traj) == 0:
            # import pdb; pdb.set_trace()
            return None
        return filtered_traj
    elif event_label == "sound":
        gt_start_time = int(gt_start_time)
        gt_end_time = math.ceil(gt_end_time)
        anchor_event["object"] = anchor_event["object"].lower()
        if "object" in anchor_event.keys():
            for object_item in object_pool:
                if "object_class" in anchor_event.keys() and anchor_event["object_class"] == object_item["object_class"]:
                    filtered_traj = {}
                    for time_key_int in range(gt_start_time, gt_end_time+1):
                        filtered_traj[second_to_time(time_key_int)] = object_item["loc"]
                    return filtered_traj
                if anchor_event["object"] in object_item["object_class"]:
                    filtered_traj = {}
                    for time_key_int in range(gt_start_time, gt_end_time+1):
                        filtered_traj[second_to_time(time_key_int)] = object_item["loc"]
                    return filtered_traj
            
            if "fridge" in anchor_event["object"] or "refrigerator" in anchor_event["object"]:
                anchor_event["object"] = "refrigerator"
            elif "tap" in anchor_event["object"]:
                anchor_event["object"] = "tap"
            elif "oven" in anchor_event["object"] or "hob" in anchor_event["object"]:
                anchor_event["object"] = "oven"
            for object_item in object_pool:
                if anchor_event["object"] in object_item["object_class"]:
                    filtered_traj = {}
                    for time_key_int in range(gt_start_time, gt_end_time+1):
                        filtered_traj[second_to_time(time_key_int)] = object_item["loc"]
                    return filtered_traj
    return None



### get audio
def get_audio_raw_traj_fromsrp(audio_track, cur_Kdrr, angle_scale_range=(5, 175), audio_fps=4):
    srp_angle_dict = {}
    srp_dir_dict = {}
    srp_dist_dict = {}
    fps_audio_track = {}
    cdr_dict = {}
    for audio_key in audio_track:
        for idx in range(len(audio_track[audio_key])):
            if "cdr" in audio_track[audio_key][idx]:
                if cur_Kdrr is not None and isinstance(cur_Kdrr, float):
                    if audio_track[audio_key][idx]["cdr"] != 0:
                        audio_track[audio_key][idx]["distance"] = np.sqrt(cur_Kdrr / audio_track[audio_key][idx]["cdr"])
                else:
                    audio_track[audio_key][idx]["distance"] = 1.0
            else:
                audio_track[audio_key][idx]["distance"] = 1.0

            fps_audio_track[str(int(audio_fps*float(audio_key)+idx))] = audio_track[audio_key][idx]

            if abs(audio_track[audio_key][idx]["angle"]) > angle_scale_range[0] and abs(audio_track[audio_key][idx]["angle"]) < angle_scale_range[1]:
                new_audio_key = int(audio_fps*float(audio_key)+idx)
                srp_angle_dict.setdefault(int(new_audio_key), [])
                srp_dist_dict.setdefault(int(new_audio_key), [])
                srp_dir_dict.setdefault(int(new_audio_key), [])
                srp_angle_dict[int(new_audio_key)].append(audio_track[audio_key][idx]["angle"])
                srp_dir_dict[int(new_audio_key)].append(audio_track[audio_key][idx]["direction"])
                if "cdr" in audio_track[audio_key][idx]:
                    cdr_dict.setdefault(int(new_audio_key), [])
                    cdr_dict[int(new_audio_key)].append(audio_track[audio_key][idx]["cdr"])
                    if cur_Kdrr is not None and isinstance(cur_Kdrr, float) and audio_track[audio_key][idx]["cdr"] !=0:
                        srp_dist_dict[int(new_audio_key)].append(np.sqrt(cur_Kdrr / audio_track[audio_key][idx]["cdr"]))
                    else:
                        srp_dist_dict[int(new_audio_key)].append(1.0)
                else:
                    srp_dist_dict[int(new_audio_key)].append(1.0)
    return fps_audio_track, srp_angle_dict, srp_dir_dict, srp_dist_dict, cdr_dict




## get dense vis seg track
def get_dense_vis_track(all_det_res, my_traj, ori_step1_res, thresh_dict={"sounding_object": 0.5, "reference_object": 0.7, "reference_facing_object": 0.7},
                        half_frustum=0, distance_shift=0, num_angles=1, num_distances=1):
    seg_loc_dict = {}
    seg_res_for_eval = {}
    sound_object_dist = {}
    if len(all_det_res) > 0:
        for time_key in all_det_res:
            if time_key not in my_traj.keys():
                time_traj_key_int = find_closest([int(key) for key in my_traj.keys()], int(time_key))
                if abs(int(time_key)-time_traj_key_int) > 20:
                    # print("no match cam traj for the seg frame!", time_traj_key_int, int(time_key))
                    continue
                else:
                    time_traj_key = str(time_traj_key_int).zfill(5)
            else:
                time_traj_key = time_key
            ori_center = my_traj[time_traj_key]["loc"]  # (x, y)
            forward_vec = my_traj[time_traj_key]["forward_vec"]
            det_item = all_det_res[time_key]
            for det_key in det_item:
                conf = np.mean(det_item[det_key]["det_conf"])
                
                pred_angle = float(det_item[det_key]["direction"])
                pred_distance = float(det_item[det_key]["distance"])

                cur_locs = get_possible_locs(pred_angle, pred_distance, ori_center, forward_vec, half_frustum=half_frustum, distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
                pred_loc = calculate_centroid(cur_locs)
                if np.isnan(np.sum(pred_loc)):
                    continue
                
                if "sounding_object" in ori_step1_res and "description" in ori_step1_res["sounding_object"]:
                    if det_key == ori_step1_res["sounding_object"]["description"]:
                        seg_res_for_eval.setdefault("sounding_object", {})
                        seg_res_for_eval["sounding_object"][time_key] = det_item[det_key]
                        if ori_step1_res["sounding_object"]["is_static"]:
                            thresh_dict["sounding_object"] = thresh_dict["reference_object"]
                        if conf < thresh_dict["sounding_object"] or pred_distance is None or pred_angle is None:
                            continue
                        sound_object_dist.setdefault(round(int(time_key)/20.), []) 
                        sound_object_dist[round(int(time_key)/20.)].append(pred_distance)
                        if not ori_step1_res["sounding_object"]["is_static"]:
                            seg_loc_dict.setdefault("sounding_object", {})
                            seg_loc_dict["sounding_object"].setdefault(round(int(time_key)/20.), [])
                            cur_locs = get_possible_locs(pred_angle, pred_distance, ori_center, forward_vec, distance_shift=1.0)
                            seg_loc_dict["sounding_object"][round(int(time_key)/20.)] = cur_locs
                        else:
                            seg_loc_dict.setdefault("sounding_object", [])
                            # seg_loc_dict["sounding_object"].append([pred_loc, conf])
                            seg_loc_dict["sounding_object"].extend(cur_locs)

                if "reference_object" in ori_step1_res and "description" in ori_step1_res["reference_object"]:
                    if det_key == ori_step1_res["reference_object"]["description"]:
                        seg_res_for_eval.setdefault("reference_object", {})
                        seg_res_for_eval["reference_object"][time_key] = det_item[det_key]
                        if conf < thresh_dict["reference_object"]:
                            continue
                        seg_loc_dict.setdefault("reference_object", [])
                        seg_loc_dict["reference_object"].extend(cur_locs)

                
                if "facing_object" in ori_step1_res and ori_step1_res["facing_object"] is not None and "description" in ori_step1_res["facing_object"]:
                    if det_key == ori_step1_res["facing_object"]["description"]:
                        seg_res_for_eval.setdefault("facing_object", {})
                        seg_res_for_eval["facing_object"][time_key] = det_item[det_key]
                        if conf < thresh_dict["facing_object"]:
                            continue
                        seg_loc_dict.setdefault("facing_object", [])
                        seg_loc_dict["facing_object"].extend(cur_locs)

    return seg_loc_dict, sound_object_dist, seg_res_for_eval


def get_single_sd_loc(ori_time_key, pred_angle, pred_distance, my_traj, half_frustum=0, distance_shift=0, num_angles=1, num_distances=1):
    time_key_int = time_to_second(ori_time_key)
    time_key = str(round(time_key_int*20.)).zfill(5)
    if time_key not in my_traj.keys():
        time_traj_key_int = find_closest([int(key) for key in my_traj.keys()], int(time_key))
        if abs(int(time_key)-time_traj_key_int) > 20:
            # print("no match cam traj for the seg frame!", time_traj_key_int, int(time_key))
            return None, None, time_key_int, time_key
        else:
            time_traj_key = str(time_traj_key_int).zfill(5)
    else:
        time_traj_key = time_key
    ori_center = my_traj[time_traj_key]["loc"]  # (x, y)
    forward_vec = my_traj[time_traj_key]["forward_vec"]
    cur_locs = get_possible_locs(pred_angle, pred_distance, ori_center, forward_vec, half_frustum=half_frustum, distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
    if cur_locs is not None:
        pred_loc = calculate_centroid(cur_locs)
    else:
        pred_loc = None
    return pred_loc, cur_locs, time_key_int, time_key

def get_single_sd_loc_int(time_key_int, pred_angle, pred_distance, my_traj, half_frustum=0, distance_shift=0, num_angles=1, num_distances=1):
    time_key = str(round(time_key_int*20.)).zfill(5)
    if time_key not in my_traj.keys():
        time_traj_key_int = find_closest([int(key) for key in my_traj.keys()], int(time_key))
        if abs(int(time_key)-time_traj_key_int) > 20:
            # print("no match cam traj for the seg frame!", time_traj_key_int, int(time_key))
            return None, None, time_key_int, time_key
        else:
            time_traj_key = str(time_traj_key_int).zfill(5)
    else:
        time_traj_key = time_key
    ori_center = my_traj[time_traj_key]["loc"]  # (x, y)
    forward_vec = my_traj[time_traj_key]["forward_vec"]
    cur_locs = get_possible_locs(pred_angle, pred_distance, ori_center, forward_vec, half_frustum=half_frustum, distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
    if cur_locs is not None:
        pred_loc = calculate_centroid(cur_locs)
    else:
        pred_loc = None
    return pred_loc, cur_locs, round(time_key_int), time_key


def get_sd_track(ori_step1_res, my_traj, half_frustum=0, distance_shift=0, num_angles=1, num_distances=1):
    sd_track = {}
    sound_object_dist = {}
    sd_res_for_eval = {}
    if "sounding_object" in ori_step1_res and "key_frames" in ori_step1_res["sounding_object"] and len(ori_step1_res["sounding_object"]["key_frames"]) > 0:
        sd_res_for_eval.setdefault("sounding_object", {})
        for ori_time_key in ori_step1_res["sounding_object"]["key_frames"]:
            det_item = ori_step1_res["sounding_object"]["key_frames"][ori_time_key]
            pred_angle = na_postprocess(det_item["direction"])
            pred_distance = na_postprocess(det_item["distance"])
            raw_time_key_int = time_to_second(ori_time_key)
            for time_key_int in np.linspace(max(0, raw_time_key_int-1), raw_time_key_int+1, 8):
                pred_loc, cur_locs, time_key_int, time_key = get_single_sd_loc_int(time_key_int, pred_angle, pred_distance, my_traj, half_frustum=half_frustum, 
                                distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
                if pred_loc is None or np.isnan(np.sum(pred_loc)):
                    continue
                sd_res_for_eval["sounding_object"][time_key] = {}
                sd_res_for_eval["sounding_object"][time_key]["det_conf"] = [1.0]
                sd_res_for_eval["sounding_object"][time_key]["direction"] = pred_angle
                sd_res_for_eval["sounding_object"][time_key]["distance"] = pred_distance

                sound_object_dist.setdefault(time_key_int, [])
                sound_object_dist[time_key_int].append(pred_distance)
                if not ori_step1_res["sounding_object"]["is_static"]:
                    sd_track.setdefault("sounding_object", {})
                    sd_track["sounding_object"].setdefault(time_key_int, [])
                    # sd_track["sounding_object"][time_key_int].append(pred_loc)
                    pred_loc, cur_locs, time_key_int, time_key = get_single_sd_loc(ori_time_key, pred_angle, pred_distance, my_traj, half_frustum=45, distance_shift=1.0, num_angles=10, num_distances=5)
                    sd_track["sounding_object"][time_key_int] = cur_locs
                else:
                    sd_track.setdefault("sounding_object", [])
                    sd_track["sounding_object"].extend(cur_locs)

    if "reference_object" in ori_step1_res and "key_frames" in ori_step1_res["reference_object"] and len(ori_step1_res["reference_object"]["key_frames"]) > 0:
        sd_res_for_eval.setdefault("reference_object", {})
        for ori_time_key in ori_step1_res["reference_object"]["key_frames"]:
            det_item = ori_step1_res["reference_object"]["key_frames"][ori_time_key]
            if not isinstance(det_item, dict) or "direction" not in det_item or "distance" not in det_item:
                continue
            pred_angle = na_postprocess(det_item["direction"])
            pred_distance = na_postprocess(det_item["distance"])
            pred_loc, cur_locs, time_key_int, time_key = get_single_sd_loc(ori_time_key, pred_angle, pred_distance, my_traj, half_frustum=half_frustum, 
                              distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
            if pred_loc is None or np.isnan(np.sum(pred_loc)):
                continue
            sd_res_for_eval["reference_object"][time_key] = {}
            sd_res_for_eval["reference_object"][time_key]["det_conf"] = [1.0]
            sd_res_for_eval["reference_object"][time_key]["direction"] = pred_angle
            sd_res_for_eval["reference_object"][time_key]["distance"] = pred_distance
            sd_track.setdefault("reference_object", [])
            sd_track["reference_object"].extend(cur_locs)

    
    if "facing_object" in ori_step1_res and ori_step1_res["facing_object"]is not None and "key_frames" in ori_step1_res["facing_object"] and len(ori_step1_res["facing_object"]["key_frames"]) > 0:
        sd_res_for_eval.setdefault("facing_object", {})
        for ori_time_key in ori_step1_res["facing_object"]["key_frames"]:
            det_item = ori_step1_res["facing_object"]["key_frames"][ori_time_key]
            if not isinstance(det_item, dict) or "direction" not in det_item or "distance" not in det_item:
                continue
            pred_angle = na_postprocess(det_item["direction"])
            pred_distance = na_postprocess(det_item["distance"])
            pred_loc, cur_locs, time_key_int, time_key = get_single_sd_loc(ori_time_key, pred_angle, pred_distance, my_traj, half_frustum=half_frustum, 
                              distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
            if pred_loc is None or np.isnan(np.sum(pred_loc)):
                continue
            sd_res_for_eval["facing_object"][time_key] = {}
            sd_res_for_eval["facing_object"][time_key]["det_conf"] = [1.0]
            sd_res_for_eval["facing_object"][time_key]["direction"] = pred_angle
            sd_res_for_eval["facing_object"][time_key]["distance"] = pred_distance
            sd_track.setdefault("facing_object", [])
            sd_track["facing_object"].extend(cur_locs)

    return sd_track, sound_object_dist, sd_res_for_eval


def get_audio_track(srp_angle_dict, srp_dist_dict, my_traj, ori_step1_res, is_static, start_time, end_time, cdr_dict = None, step1_audio_dist_dict=None, audio_fps=4, half_frustum=0, distance_shift=0, num_angles=1, num_distances=1, gt_audio_dict=None):
    smooth_audio_track = {}
    
    Kdrr = None
    if cdr_dict is not None and len(cdr_dict)>0:
        d_list = []
        for key in step1_audio_dist_dict:
            cur_key = int(key*audio_fps)
            cur_key_dist = find_closest([key for key in cdr_dict.keys()], cur_key)
            try:
                if abs(cur_key_dist-cur_key) < audio_fps:
                    # print(cdr_dict[cur_key_dist], step1_audio_dist_dict[key], srp_angle_dict[cur_key_dist])
                    d_list.append(np.mean(cdr_dict[cur_key_dist]) * step1_audio_dist_dict[key]**2)
            except:
                continue

        if len(d_list) > 0:
            d_list = np.array(d_list)
            db = DBSCAN(eps=0.5, min_samples=1).fit(d_list)
            labels = db.labels_

            # Separate clusters and outliers
            clusters = {}
            for label in set(labels):
                clusters[label] = d_list[labels == label].flatten().tolist()

            Kdrr = np.mean(clusters[0])
            

    for cur_key_int in srp_angle_dict:
        if cur_key_int >= int(audio_fps*start_time) and cur_key_int <= math.ceil(audio_fps*end_time):
            time_key = str(round(cur_key_int*20./audio_fps)).zfill(5)
            srp_pred_angle = np.mean(srp_angle_dict[cur_key_int])
            if Kdrr is None:
                srp_pred_distance = np.mean(srp_dist_dict[cur_key_int])
            else:
                srp_cdr = np.mean(cdr_dict[cur_key_int])
                srp_pred_distance = np.sqrt(Kdrr/ srp_cdr)
            

            is_back = abs(srp_pred_angle) > 90
            if time_key not in my_traj.keys():
                time_traj_key_int = find_closest([int(key) for key in my_traj.keys()], int(time_key))
                if abs(int(time_key)-time_traj_key_int) > 20:
                    # print("no match cam traj for the seg frame!", time_traj_key_int, int(time_key))
                    continue
                else:
                    time_traj_key = str(time_traj_key_int).zfill(5)
            else:
                time_traj_key = time_key
            cam_loc = my_traj[time_traj_key]["loc"]  # (x, y)
            cam_forward_vec = my_traj[time_traj_key]["forward_vec"]
            joint_srp_step1_pred_loc = None
            cur_second = round(cur_key_int*1.0/audio_fps)
            if cur_second in step1_audio_dist_dict:
                cur_second_key = find_closest([key for key in step1_audio_dist_dict.keys()], cur_second)
                if abs(cur_second_key-cur_second) < 2:
                    step1_pred_dist = step1_audio_dist_dict[cur_second_key]
                    try:
                        joint_srp_step1_pred_loc =  get_possible_locs(srp_pred_angle, step1_pred_dist, cam_loc, cam_forward_vec,  distance_shift=1.0)
                    except:
                        pass
            if joint_srp_step1_pred_loc is not None:
                smooth_audio_track.setdefault(cur_second, [])
                smooth_audio_track[cur_second].extend(joint_srp_step1_pred_loc)
            else:
                smooth_audio_track.setdefault(cur_second, [])
                srp_pred_loc = get_possible_locs(srp_pred_angle, srp_pred_distance, cam_loc, cam_forward_vec, distance_shift=1.0)

                smooth_audio_track[cur_second].extend(srp_pred_loc)
            smooth_audio_track.setdefault(-1, {})
            smooth_audio_track[-1][cur_second] = is_back

    return smooth_audio_track, Kdrr




def merge_traj(traj_list):
    # key in int seconds
    final_traj = {}
    for traj in traj_list:
        for key in traj:
            if len(traj[key]) == 0:
                continue
            final_traj.setdefault(key, [])
            if isinstance(traj[key], list):
                final_traj[key].extend(traj[key])
            else:
                final_traj[key].append(traj[key])
    for key in final_traj:
        if final_traj[key][0] is None:
            continue
        final_traj[key] = np.array(final_traj[key]).reshape(len(final_traj[key]), -1).mean(0)
    return final_traj


def merge_traj_from_key(key, sd_track, seg_vis_track, audio_agg_track, start_time, end_time, is_static=False):
    if key in ["reference_object", "facing_object"] or is_static:
        # static obj
        list_to_merge = []
        final_traj = []

       
        if key in sd_track:
            if len(sd_track[key]) > 0:
                list_to_merge.extend(sd_track[key])
                final_traj = calculate_centroid_cluster(list_to_merge, max_distance=1)        
                if key != "sounding_object":
                    return final_traj
        
        if key in seg_vis_track:
            if len(seg_vis_track[key]) > 0:
                if len(final_traj) > 0:
                    filtered_locs = [loc for loc in seg_vis_track[key] if calculate_distance(loc, final_traj) <= 1]
                    list_to_merge.extend(filtered_locs)
                else:
                    list_to_merge.extend(seg_vis_track[key])
                final_traj = calculate_centroid_cluster(seg_vis_track[key], max_distance=1)
    
        
        if key != "sounding_object":
            return final_traj


        if len(list_to_merge) == 0 and key == "sounding_object":
            if len(audio_agg_track) > 0:
                for key_int in audio_agg_track:
                    if key_int == -1:
                        continue
                    if len(final_traj) > 0:
                        filtered_locs = filter_outliers(key_int, audio_agg_track[key_int], {"sounding_object": final_traj}, "sounding_object", max_distance=1)
                        list_to_merge.extend(filtered_locs)
                    else:
                        list_to_merge.extend(audio_agg_track[key_int])
        if len(list_to_merge) > 0:
            final_traj = calculate_centroid_cluster(list_to_merge, max_distance=1)        
        else:
            return []

        if key == "sounding_object":
            valid_final_traj = {}
            for time_key in range(int(start_time), math.ceil(end_time)+1):
                valid_final_traj[time_key] = final_traj
            final_traj = valid_final_traj
        return final_traj


    else:
        final_traj = {}
        # seg
        if key in seg_vis_track and len(seg_vis_track[key]) > 0:
            for time_key in seg_vis_track[key]:
                final_traj[time_key] = calculate_centroid(seg_vis_track[key][time_key])

        if key in sd_track and len(sd_track[key]) > 0:
            for time_key in sd_track[key]:
                # final_traj[time_key] = calculate_centroid_cluster(sd_track[key][time_key], max_distance=2)
                if time_key in final_traj:
                    continue
                try:
                    filtered_locs = filter_outliers(time_key, sd_track[key][time_key], {"sounding_object": final_traj}, "sounding_object", max_distance=None)
                    if filtered_locs is not None and len(filtered_locs) > 0:
                        new_loc = calculate_centroid(filtered_locs)
                        final_traj[time_key] = new_loc
                except:
                    pass
        
        vis_final_traj = {}
        if len(final_traj) > 0:
            interp_vis_final_traj = optimize_trajectory_smoothness(final_traj, filter=True, interpolate_gaps=True)
            for time_key in range(int(start_time), math.ceil(end_time)+1):
                if time_key in interp_vis_final_traj:
                    vis_final_traj[time_key] = interp_vis_final_traj[time_key]
        

        if len(audio_agg_track) > 0:
            for time_key in audio_agg_track:
                if time_key == -1:
                    continue
                if audio_agg_track[-1][time_key] is False and len(vis_final_traj) > 0:
                    continue
                # if sound source is at back or missing visual track
                filtered_locs = filter_outliers(time_key, audio_agg_track[time_key], {"sounding_object": final_traj}, "sounding_object", max_distance=1)
                if filtered_locs is not None and len(filtered_locs) > 0:
                    new_loc = calculate_centroid(filtered_locs)
                    final_traj[time_key] = new_loc

        all_final_traj = {}
        if len(final_traj) > 0:
            for time_key in range(int(start_time), math.ceil(end_time)+1):
                if time_key in final_traj:
                    all_final_traj[time_key] = final_traj[time_key]
        if len(all_final_traj) > 0:
            all_final_traj = optimize_trajectory_smoothness(all_final_traj, filter=True, interpolate_gaps=True)
            return all_final_traj    
        
        if len(final_traj) > 0:
            final_traj = optimize_trajectory_smoothness(final_traj, filter=True, interpolate_gaps=True)
            valid_final_traj = {}
            for time_key in range(int(start_time), math.ceil(end_time)+1):
                if time_key in final_traj:
                    valid_final_traj[time_key] = final_traj[time_key]
                else:
                    if len(final_traj) > 1:
                        valid_final_traj[time_key] = predict_trajectory_point(final_traj, time_key)
                    else:
                        valid_final_traj[time_key] = list([value for value in final_traj.values()])[0]
            if len(valid_final_traj) > 0:
                return valid_final_traj

        return {}


        

