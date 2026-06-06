import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import json
import re

import datasets



METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}
WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA": 0.
}

with open(Path(__file__).parent / "spatial_avqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
base_cache_dir = yaml.safe_load("".join(safe_data))["dataset_path"]
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def spatial_avqa_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_id"].split("/")[-1] + '.mp4'
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def spatial_avqa_doc_to_sd_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    try:
        duration = round(float(doc["video_end_time"].timestamp()) - float(doc["video_start_time"].timestamp()), 2)
    except:
        import pdb; pdb.set_trace()
    return f"""
            Analyze the given video based on the question: "{question}". The total video length is {duration} seconds.
            Identify the **Sounding Object** (source of sound)
            Identify the **start_time** and **end_time** of the event mentioned in the question.
            Determine the mode:
                - If I'm in the **camera wearer's view** (egocentric), set `mode` to `egocentric`. 
                - If I'm in a **different perspective** rather than the camera's view (allocentric), set `mode` to `allocentric`. 
            
            **Output:**
            Return a single JSON object with the following structure:

            ```json
            {{
                "start_time": //start time of the event asked in the question
                "end_time": //end time
                "mode": egocentric/allocentric,
                "sounding_object": {{
                    "description": "A detailed description of the sounding object (source of sound). Include physical characteristics like type, color, material, and approximate size/shape.",
                    "is_static": true/false // True if the object is generally non-moving, false if it typically moves location
                }},
                "stand_by_object": {{
                    "object_name": "Name", //set to camera if requires_allocentric is false
                    "description": "Description",
                }},
                "facing_direction": {{ //from question
                    "object_name": "Name", 
                    "description": "Description"
                }}
            }}
        ```
        """
        

def spatial_avqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") or "Answer in single letter or numeric format."
    if doc["options"] == "None":
        return pre_prompt + "\n" + question + "\n" + post_prompt
    else:
        options = "Options:\n" + "\n".join(json.loads(doc["options"]))
        return "\n".join([pre_prompt, question, options, post_prompt])


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    if "<|endoftext|>" in pred:
        return pred.split("<|endoftext|>")[1].strip()
    pred = pred.split(' ')[0].rstrip('.').strip()
    if len(pred) > 0:
        pred = pred[0]
    return pred

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def na_postprocess(pred):
    """
    Process prediction strings in various formats.

    Possible formats:
    - List notation: [x, y, z]
    - Tuple notation: (x, y, z)
    - Single value: c

    Returns a list of extracted values.
    """
    try:
        # Check if the prediction contains a list or tuple
        numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', str(pred))
        if len(numbers) > 1:
            return [float(num) for num in numbers]
        elif len(numbers) == 1:
            return float(numbers[0])
        else:
            # If no brackets/parentheses, return the first value as a single-item list
            return float(pred)
    except:
        return None
 

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def spatial_avqa_process_results(doc, results):
    doc['ori_prediction'] = results[0]
    doc['prediction'] = results[0]
    doc['question_task'] = doc['task']
    if doc['options'] == "None":
        # NA
        pred = na_postprocess(doc['prediction'])
        doc['prediction'] = pred
        key = "MRA"
        if pred is None:
            doc[key] = WORST_CASE_FOR_METRICS[key]
        else:
            try:
                if isinstance(pred, list):
                    pred = pred[-1]
                final_acc = 0.0
                mra_thr_range = [0.1, 1.0, 0.1]
                num_pts = int((mra_thr_range[1] - mra_thr_range[0]) / mra_thr_range[2] + 1)
                thr_list = np.linspace(mra_thr_range[0], mra_thr_range[1], num_pts)
                for thr in thr_list:
                    if abs(pred-json.loads(doc['groundtruth'])) < thr:
                        final_acc += 1.0
                    else:
                        final_acc += 0.0
                doc[key] = final_acc / len(thr_list)
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        # MCA
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['groundtruth'])

    return {"spatial_avqa_score": doc}



def spatial_avqa_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}
    for question_type, question_type_indexes in results.groupby('question_task').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        flag = False
        for metric in WORST_CASE_FOR_METRICS.keys():
            if metric in per_question_type.keys():
                if not np.isnan(per_question_type[metric].mean()):
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                    flag = True
        if flag:
            continue

        raise ValueError(f"Unknown question type: {question_type}")

    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall']*100
