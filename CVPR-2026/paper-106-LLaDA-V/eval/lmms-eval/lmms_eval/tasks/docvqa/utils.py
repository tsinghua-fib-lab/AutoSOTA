import json
import os

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

def parse_VWImodel_response(response):
    if not response:
        return None
    
    try:
        response = response.replace('*', '')
        
        answer_index = response.rfind("Answer:")
        if answer_index == -1:
            return None
            
        answer_part = response[answer_index:].strip()
        
        answer = answer_part.split(":")[-1].strip().strip("., ")
        
        return answer
        #return response
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None
    
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls(
    references,
    predictions,
    thresh_hold=0.5,
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = "None" if predictions is None or predictions[0] is None else (predictions[0] if isinstance(predictions[0], str) else predictions[0][0] if predictions[0][0] is not None else "None")

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return {"anls": question_result}

def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def docvqa_process_results(doc, results):
    gold = doc["answers"]
    result = [res.strip() for res in results]
    result = [parse_VWImodel_response(res) for res in result]

    # 确保 gold 是列表形式（兼容单答案和多答案）
    if not isinstance(gold, list):
        gold = [gold]

    # 计算 ANLS 分数（假设 self._metric_fn_list["anls"] 是 ANLS 计算函数）
    anls_score = anls(
        references=gold,
        predictions=result,
    )

    # 如果返回的是字典（如 HuggingFace Evaluate），提取 ANLS 值
    if isinstance(anls_score, dict):
        anls_score = anls_score["anls"]

    return {"anls": anls_score}

def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
