"""
SD format temporal grounding + object referral code of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import json
import google.generativeai as genai
import os
from tqdm import tqdm
import time


def get_prompt(gt, res):

    return f"""
    You are an AI assistant tasked with evaluating object detection predictions against ground truth data.
    Your goal is to compare the predicted objects with the ground truth objects and then generate a JSON output that includes the original prediction data, the evaluation results, and adheres to specific formatting rules.

    Given the Ground Truth (GT):
    {gt}

    And the Prediction (RES):
    {res}

    Follow these steps:

    1.  **Identify and Compare Objects:**
        *   Examine the `sounding_object` in both GT and RES.
        *   Examine the `reference_object` (also referred to as `stand_by_object`) in both GT and RES.
        *   Examine the `facing_object` (also referred to as `facing_direction_object`) in both GT and RES.

    2.  **Judge Each Object:**
        For each of the three objects (`sounding_object`, `reference_object`, `facing_object`):
        *   Determine if the prediction for that object is `correct` or `incorrect` when compared to the GT. Only correct if match with the corresponding ONE object category and you feel they are possible the same identity.
        *   Provide a concise `reasoning` for your judgment. If incorrect, briefly state why (e.g., "GT type is 'car', prediction is 'truck'" or "GT location differs").

    3.  **Construct the `judge_res` Dictionary:**
        Create a dictionary named `judge_res` with the evaluation results, structured as follows:
        ```json
        {{
        "sounding_object": {{
            "status": "correct_or_incorrect",
            "reasoning": "Your concise reason here"
        }},
        "reference_object": {{
            "status": "correct_or_incorrect",
            "reasoning": "Your concise reason here"
        }},
        "facing_object": {{
            "status": "correct_or_incorrect",
            "reasoning": "Your concise reason here"
        }}
        }}
        ```
        Replace `"correct_or_incorrect"` with either `"correct"` or `"incorrect"`.

    4.  **Format the Final JSON Output:**
        *   The final output MUST be a single, valid, parsable JSON string.
        *   This JSON string must **contain all original keys and values from the input `prediction` ({res})**.
        *   **Add the `judge_res` dictionary (created in step 3) as a new key-value pair to this JSON structure.**

    5.  **Apply Specific Data Formatting Rules to the ENTIRE final JSON output (wherever these data types appear, whether in original prediction or generated parts):**
        *   **Distance:** Ensure any distance value contains only numeric characters (e.g., if a value is "10.5m", it should become `10.5` in the JSON).
        *   **Direction:**
            *   Represent direction as a numeric degree.
            *   Use the following mapping for common terms:
                *   `front`: 0
                *   `right`: 90
                *   `left`: -90
                *   `back`: 180 (or -180, be consistent)
                *   `front-right`: 45
                *   `front-left`: -45
                *   `back-right`: 135
                *   `back-left`: -135
            *   If other textual directions are present, convert them to their appropriate numeric degree if a clear mapping exists. Otherwise, retain the numeric degree if already provided.
        *   **Time:** All time values must be formatted as `minutes:seconds` (e.g., `02:35`).

    Please generate the complete JSON string based on these instructions.
    """

def parse(content):
    content = content.text
    json_start = content.find('```json')
    flag = False
    if json_start != -1:
        json_start = content.find('{', json_start)
        json_end = content.rfind('}')
        content = content[json_start:json_end+1]
        try:
            response_json = json.loads(content)
            flag = True
        except:
            try:
                content = '[' + content + ']'
                response_json = json.loads(content)
                flag = True
            except:
                pass
                         
    if flag is False:
        response_json = content
    return flag, response_json

NUM_SECONDS_TO_SLEEP = os.getenv("GEMINI_INTV_AFTER_FAILED", 10)
GOOGLE_API_KEY = ""

genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {'temperature': 0.5, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 10240}
model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)


# method = "minicpm"
# method = "ola"
# method = "egogpt"
# method = "longvale"
method = "videollama_new"
all_data = json.load(open(f"data/sd_output_v2/{method}.json", "r"))
output_file = f"data/sd_output_v2/format_{method}.json"
gt_data = json.load(open("data/spatial_avqa/test/gtavmap.json", "r"))
gt_data_dict = {}
for cur_get_data in gt_data:
    gt_data_dict[cur_get_data["question"]] = cur_get_data
if os.path.exists(output_file):
    format_data_dict = json.load(open(output_file, "r"))
else:
    format_data_dict = {}
for data in tqdm(all_data):
    for key in gt_data_dict:
        if key in data["prompt"]:
            gt = gt_data_dict[key]["gt_context_json"]
            break
    if str(gt_data_dict[key]['id']) in format_data_dict:
        continue
    chat_session_json = model.start_chat(
                    history=[])
    res = data["output"]
    
   
    content = chat_session_json.send_message(get_prompt(gt, res))
    try:
        flag, new_res = parse(content)
    except:
        try:
            flag, new_res = parse(content.parts[0])
        except:
            continue
    if flag:
        try:
            new_res["ori_output"] = data["output"]
        except:
            pass
        format_data_dict[gt_data_dict[key]['id']] = new_res.copy()
        json.dump(format_data_dict, open(output_file, "w"), indent=4)
    else:
        continue
