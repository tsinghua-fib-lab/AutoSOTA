"""
Main code for SAVVY pipeline - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import os
import json
from tqdm import tqdm
import numpy as np
import cv2
import time

from utils.avmap_utils import get_audio_tracks, get_audio_raw_traj_fromsrp, get_dense_vis_track, get_sd_track, merge_traj, get_audio_track, merge_traj_from_key
from evaluation.avmap_eval_utils import eval_srp_audio, eval_vis_track2d_loc, eval_ego_qa_acc_avmap, eval_exo_qa_acc_avmap
from utils.format_utils import time_to_second, second_to_time, find_closest
from utils.track_utils import interpolate_missing_timepoints1d
from evaluation.metric_utils import mean_relative_accuracy, _calculate_distance
from utils.localization import calculate_centroid, calculate_centroid_cluster
                    

traj_root = "data/dense_traj/aea"
guest_traj_root = "data/correct_traj/aea"
audio_gt_dir = "data/speech_gt_keyframes_all"
vis_gt_dir = "data/vis_gt_keyframes_all"
gt_sd_file = "data/test_json/gtavmap.json"
gt_output_file = "data/output/predavmap.json"
audio_set_id = "4_3456"
gt_data = json.load(open(gt_sd_file, "r"))
tar_folder = "data/sd_output_v2"
method = "gemini25"
audio_fps = 4
os.makedirs(os.path.dirname(gt_output_file), exist_ok=True)

invalid_count = 0
time_invalid_count = 0
time_mra_list = []

audio_metrics = dict(
    audio_dir_acc = [],
    audio_dir_left_acc = [],
    audio_dir_right_acc = [],
    audio_dir_front_acc = [],
    audio_dir_back_acc = [],
    audio_angle_err = [],
    audio_dist_err = [],
    audio_loc_acc = []
)

dense_seg_track_metrics = {
    "sounding_object": {
        "pred_acc": [],
        "angle_err": [],
        "distance_err": []
    },
    "static_object": {
        "pred_acc": [],
        "angle_err": [],
        "distance_err": []
    }
}


sd_track_metrics = {
    "sounding_object": {
        "pred_acc": [],
        "angle_err": [],
        "distance_err": []
    },
    "static_object": {
        "pred_acc": [],
        "angle_err": [],
        "distance_err": []
    }
}
all_time_list = []
sounding_loc_dist = []
reference_loc_dist = []
facing_loc_dist = []

sounding_loc_acc = []
reference_loc_acc = []
facing_loc_acc = []
missing_sounding_loc_num = 0
missing_reference_loc_num = 0
missing_facing_loc_num = 0
invalid_det_count = 0

ego_qa_metrics = dict(
    ego_out_mca_acc = [],
    ego_out_mca_simple_acc = [],
    ego_out_mca_hard_acc = [],
    ego_out_dist_acc = [],
)

exo_qa_metrics = dict(
    exo_out_mca_acc = [],
    exo_out_mca_simple_acc = [],
    exo_out_mca_hard_acc = [],
    exo_out_dist_acc = [],
)
unseen_list = []
K_drr_dict = {}
K_drr_path = f"data/audio_tracks_drr_mid_{audio_set_id}/kdrr.json"
if os.path.exists(K_drr_path):
    K_drr_dict = json.load(open(K_drr_path, "r"))

tar_file = f"{tar_folder}/{method}.json"
tar_data = json.load(open(tar_file, "r"))
time_mra_list = []
iou_time_mra_list = []
obj_ref_scores_dict = {}
time_invalid_count = 0

new_all_data = []
all_ids = []
invalid_objects = []
idx = -1

valid_ids = set()
for doc in json.load(open("data/test_json/savvy_bench.json", "r")):
    valid_ids.add(doc["id"])
total_frames_dict = {}
for data in tqdm(gt_data):
    idx += 1
    if data["id"] not in valid_ids:
        continue
    loc_id = data["video_id"].split("/")[-1].split("_")[0]
    
    if data["id"] not in tar_data:
        invalid_count += 1
        print(data["id"])
        continue
    ori_step1_res = tar_data[data["id"]]

    start_time_now = time.time()
    video_res = {}
    video_id = data["video_id"].split("/")[-1]
    json_traj_path = f"{traj_root}/{video_id}.json"
    with open(json_traj_path, "r") as f:
        my_traj = json.load(f)
    
    if os.path.exists(f"{audio_gt_dir}/{video_id}.json"):
        gt_audio_dict = json.load(open(f"{audio_gt_dir}/{video_id}.json", "r"))
    else:
        print(f"{audio_gt_dir}/{video_id}.json")
        gt_audio_dict = {}
    if os.path.exists(f"{vis_gt_dir}/{video_id}.json"):
        gt_vis_dict = json.load(open(f"{vis_gt_dir}/{video_id}.json", "r"))
    else:
        print(f"{audio_gt_dir}/{video_id}.json")
        gt_vis_dict = {}
    
    # guest json path
    rec_id = video_id.split("_")[-1]
    if rec_id == "rec1":
        guest_video_name = video_id[:-5] + "_rec2"
    elif rec_id == "rec2":
        guest_video_name = video_id[:-5] + "_rec1"
    guest_traj_file = f"{guest_traj_root}/{guest_video_name}.json"
    if not os.path.exists(guest_traj_file):
        guest_traj = {}
    else:
        with open(guest_traj_file, "r") as f:
            guest_traj = json.load(f)
    
    
    if "is_static" not in ori_step1_res["sounding_object"]:
        ori_step1_res["sounding_object"]["is_static"] = False

    acc_standard = {"angle": 45, "distance": 1.0}

    cur_output = {}    

    audio_doa_root = f"data/audio_tracks_drr_mid_{audio_set_id}/"
    audio_root = "data/audios_new/"
    if audio_set_id == "2_34":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[3, 4], audio_fps=audio_fps)
    elif audio_set_id == "4_3456":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[3, 4, 5, 6], audio_fps=audio_fps)
    elif audio_set_id == "2_56":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[5, 6], audio_fps=audio_fps)
    elif audio_set_id == "2_02":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[0, 2], audio_fps=audio_fps)
    elif audio_set_id == "4_0234":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[0, 2, 3, 4], audio_fps=audio_fps)
    elif audio_set_id == "4_0256":
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[0, 2, 5, 6], audio_fps=audio_fps)
    else:
        audio_track = get_audio_tracks(audio_doa_root, video_id, audio_root, selected_indices=[0, 1, 2, 3, 4, 5, 6], audio_fps=audio_fps)
 
    valid_angle_scale_range = (5, 175)
    half_frustum, distance_shift, num_angles, num_distances = 0, 0, 1, 1

    if data["id"] in K_drr_dict:
        cur_Kdrr = K_drr_dict[data["id"]]
    else:
        cur_Kdrr = None

    fps_audio_track, srp_angle_dict, srp_dir_dict, srp_dist_dict, cdr_dict = get_audio_raw_traj_fromsrp(audio_track, cur_Kdrr, angle_scale_range=valid_angle_scale_range)
    audio_metrics = eval_srp_audio(fps_audio_track, gt_audio_dict, audio_metrics, valid_angle_abs_range=valid_angle_scale_range, acc_standard=acc_standard)
    ######
    


    eval_vis_track2d_thr_dict={"sounding_object": 0.5, "reference_object": 0.6, "facing_object": 0.6}
    ##### Step 2 ClipSeg
    key_det_file_path = f"""data/det_aea_clipseg_sdv2_gemini25/{data["id"]}/key_detection.json"""
    det_file_path = f"""data/det_aea_clipseg_sdv2_gemini25/{data["id"]}/detection.json"""
    if not os.path.exists(det_file_path) and not os.path.exists(key_det_file_path):
        invalid_det_count += 1
        print(data["id"])
    if os.path.exists(det_file_path):
        all_det_res = json.load(open(det_file_path, "r"))
    else:
        all_det_res = {}

    seg_vis_track = seg_res_for_eval = {"reference_object": {}, "facing_object": {}, "sounding_object": {}}
    seg_sound_object_dist = {}
    video_root = "data/videos"
    if video_id not in total_frames_dict:
        video_path = os.path.join(video_root, video_id+".mp4")
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_dict[video_id] = total_frames
    else:
        total_frames = total_frames_dict[video_id]
    
    seg_vis_track, seg_sound_object_dist, seg_res_for_eval = get_dense_vis_track(all_det_res, my_traj, ori_step1_res, thresh_dict=eval_vis_track2d_thr_dict, half_frustum=half_frustum, distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)

    dense_seg_track_metrics = eval_vis_track2d_loc(seg_res_for_eval, gt_vis_dict, data["gt_context_json"], dense_seg_track_metrics, 
                                     eval_thr_dict=eval_vis_track2d_thr_dict, acc_standard=acc_standard)    
    
    #### Step3 SD 
    sd_track = sd_res_for_eval = {"reference_object": {}, "facing_object": {}, "sounding_object": {}}
    sd_audio_sound_object_dist = {}
    sd_track, sd_audio_sound_object_dist, sd_res_for_eval = get_sd_track(ori_step1_res, my_traj, half_frustum=half_frustum, distance_shift=distance_shift, num_angles=num_angles, num_distances=num_distances)
    sd_track_metrics = eval_vis_track2d_loc(sd_res_for_eval, gt_vis_dict, data["gt_context_json"], sd_track_metrics, 
                                     eval_thr_dict=eval_vis_track2d_thr_dict, acc_standard=acc_standard)
    if len(sd_audio_sound_object_dist) == 0:
        step1_audio_dist_dict = merge_traj([{}, seg_sound_object_dist])
    else:
        step1_audio_dist_dict = merge_traj([{}, sd_audio_sound_object_dist])



    try:
        start_time = max(0, int(time_to_second(ori_step1_res["start_time"])))
        end_time = min(round(data["anno_value"]['video_stop_time'] - data["anno_value"]['video_start_time']) + 1, time_to_second(ori_step1_res["end_time"])+1)
        if end_time < start_time:
            start_time = 0
            end_time = round(data["anno_value"]['video_stop_time'] - data["anno_value"]['video_start_time']) + 1
            time_invalid = True
            audio_agg_track = {}
        else:
            time_invalid = False
    except:
        import pdb; pdb.set_trace()
        start_time = 0
        end_time = round(data["anno_value"]['video_stop_time'] - data["anno_value"]['video_start_time']) + 1
        time_invalid = True
        audio_agg_track = {}
    if time_invalid is False:
        audio_agg_track, Kdrr = get_audio_track(srp_angle_dict, srp_dist_dict, my_traj, ori_step1_res, ori_step1_res["sounding_object"]["is_static"], start_time, end_time, cdr_dict=cdr_dict, step1_audio_dist_dict=step1_audio_dist_dict, audio_fps=audio_fps, gt_audio_dict=gt_audio_dict)   
        K_drr_dict[data["id"]] = Kdrr

    final_traj_dict = {}
    for obj_key in ["reference_object", "sounding_object", "facing_object"]:
        cur_final_traj = merge_traj_from_key(obj_key, sd_track, seg_vis_track, audio_agg_track, start_time, end_time, is_static=ori_step1_res["sounding_object"]["is_static"])
        if len(cur_final_traj) == 0:
            cur_final_traj = {}

        final_traj_dict[obj_key] = cur_final_traj

    seen_keys = []
    if "sounding_object" in sd_track and isinstance(sd_track["sounding_object"], dict):
        seen_keys = [key for key in  sd_track["sounding_object"].keys() if len(sd_track["sounding_object"]) > 0]
    if "sounding_object" in seg_vis_track and isinstance(seg_vis_track["sounding_object"], dict):
        seen_keys.extend([key for key in seg_vis_track["sounding_object"].keys() if len(seg_vis_track["sounding_object"][key])>0])
    seen_keys = set(seen_keys)
    seeenflag = False
    for timeint in range(start_time-2, end_time+2):
        if timeint in seen_keys:
            seeenflag = True
            break
    if seeenflag is False:
        unseen_list.append(data["id"])
    
        
    # # get camera traj for cur_output
    cur_output["camera traj"] = {}
    for cur_key_int in range(start_time, end_time):
        traj_key_int = find_closest([int(traj_key) for traj_key in my_traj.keys()], 20*cur_key_int)
        if abs(traj_key_int-20*cur_key_int) < 30:
            traj_time_key = my_traj[str(traj_key_int).zfill(5)]
        cur_output["camera traj"][second_to_time(round(traj_key_int/20.))] = my_traj[str(traj_key_int).zfill(5)]
    
    cam_traj = {}
    for time_key, cur_traj in cur_output["camera traj"].items():
        for key in cur_traj:
            cam_traj.setdefault(key, [])
            cam_traj[key].append(cur_traj[key])
    final_traj_avg = {}
    for key in cam_traj.keys():
        avg_value = [0] * len(cam_traj[key][0])
        for value in cam_traj[key]:
            for i, component in enumerate(value):
                avg_value[i] += component
        final_traj_avg[key] = [round(val / len(cam_traj[key]), 3) for val in avg_value]

    new_data = data.copy()

    ego_qa_metrics, prediction = eval_ego_qa_acc_avmap(final_traj_dict["sounding_object"], final_traj_avg, ori_step1_res, data, ego_qa_metrics, mra_thr_range=(0.1, 1.0, 0.1))
    if prediction is not None:
        new_data["prediction"] = prediction
    exo_qa_metrics, prediction = eval_exo_qa_acc_avmap(final_traj_dict, final_traj_avg, ori_step1_res, data, exo_qa_metrics, mra_thr_range=(0.1, 1.0, 0.1))
    cur_output["time range"] = {
        "start_time": ori_step1_res["start_time"],
        "end_time": ori_step1_res["end_time"]
    }
    if prediction is not None:
        new_data["prediction"] = prediction

    gt_start_time = data["anno_value"]['zero_base_start_time']
    gt_end_time = data["anno_value"]['zero_base_stop_time']
    
    if time_invalid is False:
        time_mra = mean_relative_accuracy("overlap", (start_time, end_time), (gt_start_time, gt_end_time), 0.05, 0.5, 0.05)   
        time_mra_list.append(time_mra)      
    else:
        time_mra = None

    final_avmap = {}
    if "event" in ori_step1_res:
        final_avmap["event"] = ori_step1_res["event"]
    final_avmap["time range"] = {}
    if "start_time" in ori_step1_res:
        final_avmap["time range"]["start_time"] = ori_step1_res["start_time"]
    if "end_time" in ori_step1_res:
        final_avmap["time range"]["end_time"] = ori_step1_res["end_time"]
    
    sound_flag = False
    if "sounding_object" in ori_step1_res:
        final_avmap["sounding_object"] = ori_step1_res["sounding_object"].copy()
        if "key_frames" in final_avmap["sounding_object"]:
            del final_avmap["sounding_object"]["key_frames"]
        if len(final_traj_dict["sounding_object"]) > 0:
            pred_sounding_loc = calculate_centroid([value for value in final_traj_dict["sounding_object"].values()])
            final_avmap["sounding_object"]["loc"] = [pred_sounding_loc[0], pred_sounding_loc[1]]
            sounding_loc_dist.append(_calculate_distance(pred_sounding_loc, data["gt_context_json"]["sounding_object"]["loc"]))
            if sounding_loc_dist[-1] <= acc_standard["distance"]:
                sounding_loc_acc.append(1)
            else:
                sounding_loc_acc.append(0)
            sound_flag = True
        else:
            missing_sounding_loc_num += 1
    else:
        final_avmap["sounding_object"] = {}
    if not sound_flag:
        sounding_loc_acc.append(0)

    pred_reference_loc = None
    is_ego = False
    ref_flag = False
    facing_flag = False
    if "reference_object" in ori_step1_res and isinstance(ori_step1_res["reference_object"], dict):
        final_avmap["reference_object"] = ori_step1_res["reference_object"].copy()
        if "key_frames" in final_avmap["reference_object"]:
            del final_avmap["reference_object"]["key_frames"]
        if "camera" in final_avmap["reference_object"]["object_name"].lower() or "camera" in final_avmap["reference_object"]["description"].lower() or ori_step1_res["mode"] == "egocentric":
            # ego
            is_ego = True
            if "loc" in final_traj_avg:
                final_avmap["reference_object"]["loc"] = [final_traj_avg["loc"][0], final_traj_avg["loc"][1]]
        else:
            if len(final_traj_dict["reference_object"]) > 0:
                pred_reference_loc = np.array(final_traj_dict["reference_object"])
                final_avmap["reference_object"]["loc"] = [pred_reference_loc[0], pred_reference_loc[1]]
                if "exo" in data["id"]:
                    reference_loc_dist.append(_calculate_distance(pred_reference_loc, data["gt_context_json"]["reference_object"]["loc"]))
                    if reference_loc_dist[-1] <= acc_standard["distance"]:
                        reference_loc_acc.append(1)
                    else:
                        reference_loc_acc.append(0)
                    ref_flag = True
            else:
                missing_reference_loc_num += 1
                final_avmap["reference_object"]["loc"] = "unknown, please infer using the video and other context info"
        if "object_name" in final_avmap["reference_object"]:
            del final_avmap["reference_object"]["object_name"]
    else:
        final_avmap["reference_object"] = {}

    if "exo" in data["id"] and ref_flag is False:
        reference_loc_acc.append(0)

    if is_ego:
        final_avmap["reference_forward_vector"] = {
            "description": "camera forward direction"
        }
        if "forward_vec" in final_traj_avg:
            final_avmap["reference_forward_vector"]["vec"] = [final_traj_avg["forward_vec"][0], final_traj_avg["forward_vec"][1]]

    elif "facing_object" in ori_step1_res and "description" in ori_step1_res["facing_object"] and isinstance(ori_step1_res["facing_object"], dict):
        final_avmap["reference_forward_vector"] = {}
        ori_object_name = ori_step1_res["facing_object"]["object_name"]
        ori_description = ori_step1_res["facing_object"]["description"]
        final_avmap["reference_forward_vector"]["description"] = f"""pointing from the 'reference_object' to {ori_object_name}""" + f""" ({ori_description})"""
        if len(final_traj_dict["facing_object"]) > 0 and pred_reference_loc is not None:
            forward_vec = np.array(final_traj_dict["facing_object"]) - pred_reference_loc
            final_avmap["reference_forward_vector"]["vec"] = [forward_vec[0], forward_vec[1]]
            if "exo" in data["id"] and "direction" in data["id"]:
                gt_facing_loc = np.array(data["gt_context_json"]["reference_forward_vector"]["vec"]) + np.array(data["gt_context_json"]["reference_object"]["loc"])
                facing_loc_dist.append(_calculate_distance(np.array(final_traj_dict["facing_object"]), gt_facing_loc))
                if facing_loc_dist[-1] <= acc_standard["distance"]:
                    facing_loc_acc.append(1)
                else:
                    facing_loc_acc.append(0)
                facing_flag = True
        else:
            # miss facing
            missing_facing_loc_num += 1
    else:
        final_avmap["reference_forward_vector"] = {}

    if "exo" in data["id"] and "direction" in data["id"] and facing_flag is False:
        facing_loc_acc.append(0)

    
    new_data["pred_context_json"] = final_avmap.copy()
    del new_data["gt_context_json"]
    new_all_data.append(new_data)

    all_time_list.append(time.time() -  start_time_now)
if not os.path.exists(K_drr_path):
    json.dump(K_drr_dict, open(K_drr_path, "w"))
json.dump(new_all_data, open(gt_output_file, "w"), indent=4)
print("Total data sample count: ", len(new_all_data))
avg_time_mra = round(np.mean(time_mra_list), 3)
print("time mra: ", avg_time_mra, f" ({len(time_mra_list)})")
print(f"Invalid Count: {invalid_count}")
print(f"Invalid Det Count: {invalid_det_count}")
print("sounding_loc_dist: ", round(np.mean(sounding_loc_dist), 3), "   count: ", len(sounding_loc_dist), "   missing: ", missing_sounding_loc_num)
print("reference_loc_dist: ", round(np.mean(reference_loc_dist), 3), "   count: ", len(reference_loc_dist), "   missing: ", missing_reference_loc_num)
print("facing_loc_dist: ", round(np.mean(facing_loc_dist), 3), "   count: ", len(facing_loc_dist), "   missing: ", missing_facing_loc_num)
print("sounding_loc_acc: ", round(np.mean(sounding_loc_acc), 3), "   count: ", len(sounding_loc_acc), "   missing: ", missing_sounding_loc_num)
print("reference_loc_acc: ", round(np.mean(reference_loc_acc), 3), "   count: ", len(reference_loc_acc), "   missing: ", missing_reference_loc_num)
print("facing_loc_acc: ", round(np.mean(facing_loc_acc), 3), "   count: ", len(facing_loc_acc), "   missing: ", missing_facing_loc_num)
for key, value in audio_metrics.items():
    print("audio_metrics - ", key, ": ", round(np.mean(value), 3), f" ({len(value)})")

for org_key in ["sounding_object", "static_object"]:
    for key, value in dense_seg_track_metrics[org_key].items():
        print("dense_seg_metrics - ", org_key, key, ": ", round(np.mean(value), 3), f" ({len(value)})")
    for key, value in sd_track_metrics[org_key].items():
        print("sd_track_metrics - ", org_key, key, ": ", round(np.mean(value), 3), f" ({len(value)})")
for key, value in ego_qa_metrics.items():
    print(key, ": ", round(np.mean(value), 3), f" ({len(value)})")
for key, value in exo_qa_metrics.items():
    print(key, ": ", round(np.mean(value), 3), f" ({len(value)})")
