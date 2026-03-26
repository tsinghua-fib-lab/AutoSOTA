"""
Snapshot Descriptor code using Gemini for SAVVY pipeline stage1 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import google.generativeai as genai
import os
import json
import math
from tqdm import tqdm
import time
import random

NUM_SECONDS_TO_SLEEP = os.getenv("GEMINI_INTV_AFTER_FAILED", 10)
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 40960}
model = genai.GenerativeModel("gemini-2.5-pro", generation_config=generation_config)


###
from utils.file_utils import load_uploaded_files, check_and_reupload
from utils.sd_prompt import get_sd_prompt_v2
from utils.format_utils import parse

RUN_DEMO = True

### paths
video_root = "data/videos/"
json_root = "data/context_json/sdv2/"
os.makedirs(json_root, exist_ok=True)
if RUN_DEMO:
    video_root = "data/demo_videos/"
    json_root = "data/context_json/demo_sdv2/"
    os.makedirs(json_root, exist_ok=True)

qa_data_all = json.load(open("data/test_json/savvy_bench.json", "r"))
if RUN_DEMO:
    qa_data_all = [
        {
            "video_id": "aea/loc2_script3_seq4_rec2",
            "id": "demo-ego-dir",
            "start_frame": 39.00,
            "end_frame": 41.00,
            "task": "spatial_temporal_audio_direction_ego",
            "level": "hard",
            "question": 'Imagine you are the camera wearer, when the speech topic "confirming they have La Croix drinks" come up, relative to where you are facing, where is the other person: front-left, front-right, back-left or back-right?  The directions refer to the quadrants of a Cartesian plane (if you are standing at the origin and facing along the positive y-axis). Consider the center point location of the object as the its location.',
            "groundtruth" :  'B',
            "QA_type":  'MCA',
            "options":  '["A: front-left", "B: front-right", "C: back-left", "D: back-right"]',
            "modality": "audio-visual"
        },
        {
            "video_id": "aea/loc2_script3_seq4_rec2",
            "id": "demo-alo-dist",
            "start_frame": 39.00,
            "end_frame": 41.00,
            "task": "spatial_temporal_audio_direction_ego",
            "level": "hard",
            "question": 'When the speech topic "confirming they have La Croix drinks" is mentioned, what is the distance between the two-seater table and the speech sound source in meters? Consider the center point location of the object as the its location. Answer in numeric format.',
            "groundtruth" :  '3.82',
            "QA_type":  'NA',
            "options":  'None',
            "modality": "audio-visual"
        }
    ]

random.shuffle(qa_data_all)
for data in tqdm(qa_data_all):
    ori_video_id = data["video_id"]
    video_id = ori_video_id.split("/")[-1]
    qid = data["id"]
    res0_json_path = f"{json_root}/{qid}.json"

    if os.path.exists(res0_json_path):
        continue
    
    video_path = f"{video_root}/{video_id}.mp4"
    uploaded_files = load_uploaded_files()
    uploaded_obj = check_and_reupload(video_path, uploaded_files, genai)
    
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [uploaded_obj]
            },
        ]
        )
    question = data["question"]

    contexts1 = get_sd_prompt_v2(uploaded_obj, question)
    content = chat_session.send_message(contexts1)
    
    try:
        res_file = parse(content, res_path=res0_json_path)
    except:
        print("format error: ", res0_json_path)
        continue
    try:
        dynamic_key_frame_data = json.load(open(res0_json_path, "r"))
        object_details = dynamic_key_frame_data["sounding_object"]["description"]
    except:
        chat_session_json = model.start_chat(
            history=[])
        content = chat_session_json.send_message(f"""format the str to a json str that can be parsed {res_file}, contain all original keys""")
        
    try:
        res_file = parse(content, res_path=res0_json_path)
        dynamic_key_frame_data = json.load(open(res0_json_path, "r"))
        object_details = dynamic_key_frame_data["sounding_object"]
        dynamic_key_frame_data["question"] = question
        dynamic_key_frame_data["qid"] = qid
        dynamic_key_frame_data["video_id"] = video_id
        json.dump(dynamic_key_frame_data, open(res0_json_path, "w"), indent=4)
    except:
        print("format error: ", res0_json_path)
        import pdb; pdb.set_trace()
        os.remove(res0_json_path)
