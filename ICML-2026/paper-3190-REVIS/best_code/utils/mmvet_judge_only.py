# utils/mmvet_eval_only.py
import os
import json
import time
import re
from tqdm import tqdm
from collections import Counter
import requests
import argparse


QWEN_API_KEY = "YOUR_API_KEY" 
QWEN_MODEL = "Qwen3-235B-A22B-Instruct-2507"
BASE_URL = "YOUR_URL"

# --- MM-Vet Grading Prompt ---
GRADING_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


def qwen_api_call(system_prompt: str, prompt: str, model: str = QWEN_MODEL, temperature: float = 0.25, max_tokens: int = 512) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "SOFA-TraceId": "trace_" + str(int(time.time())),
        "SOFA-RpcId": "0"
    }
    payload = {
        "model": model,
        "messages": [
            {'role': 'system', 'content': system_prompt}, 
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload), timeout=40)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API call error: {e}")
        return ""

def calculate_metrics(mmvet_path, grade_results):
    mmvet_json = os.path.join(mmvet_path, "mm-vet.json")
    with open(mmvet_json, "r") as f:
        data = json.load(f)
        
    scores = Counter()
    counts = Counter()
    caps = ["rec", "ocr", "know", "gen", "spat", "math"]
    
    for sample_id, grade_info in grade_results.items():
        score = grade_info["score"]
        # Ensure ID exists in original dataset
        if sample_id not in data: continue
        
        capability = data[sample_id]["capability"]
        for c in capability:
            scores[c] += score
            counts[c] += 1
        scores["total"] += score
        counts["total"] += 1

    final_metrics = {}
    for c in caps + ["total"]:
        if counts[c] > 0:
            final_metrics[c] = round(scores[c] / counts[c] * 100, 2)
        else:
            final_metrics[c] = 0.0
            
    final_metrics["score"] = final_metrics["total"]
    
    return final_metrics

def run_grading(mmvet_path, result_file, output_file):

    mmvet_json = os.path.join(mmvet_path, "mm-vet.json")
    if not os.path.exists(mmvet_json):
        raise FileNotFoundError(f"MM-Vet data not found at {mmvet_json}")
    
    with open(mmvet_json, "r") as f:
        data = json.load(f)
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Prediction file not found at {result_file}")
        
    with open(result_file, "r") as f:
        results = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(output_file):
        grade_results = {} 
    else:
        grade_results = {}

    print(f"Starting Evaluation using Qwen API...")
    print(f"Dataset: {len(data)} | Predictions: {len(results)}")

    for sample_id, sample_info in tqdm(data.items(), desc="Judging"):
        if sample_id not in results: 
            continue
            
        model_pred = results[sample_id]
        gt_answer = sample_info["answer"]
        question_text = sample_info["question"]

        user_content = " | ".join([
                question_text,
                gt_answer.replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                model_pred,
                ""
        ])
        
        current_temperature = 0.0 
        score = 0.0
        success = False
        try_count = 0

        while not success and try_count < 5:
            try:
                content = qwen_api_call(
                    system_prompt=GRADING_PROMPT, 
                    prompt=user_content,
                    temperature=current_temperature
                )
                
                if not content: raise ValueError("Empty response")
                
                match = re.search(r"(\d+\.\d+|\d+)", content)
                if match:
                    parsed_score = float(match.group(1))
                    if 0.0 <= parsed_score <= 1.0:
                        score = parsed_score
                        success = True
                    else:
                        raise ValueError(f"Score out of range: {parsed_score}")
                else:
                     raise ValueError(f"No score found in content")

            except Exception as e:
                if "Predict the correctness" not in user_content:
                    user_content += "\nPredict the correctness of the answer (digit only 0.0 to 1.0): "
                current_temperature = min(current_temperature + 0.2, 0.8)
                try_count += 1
                time.sleep(0.5)
                continue
        
        grade_results[sample_id] = {
            "score": score,
            "prediction": model_pred
        }

    metrics = calculate_metrics(mmvet_path, grade_results)
    
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    detail_file = output_file.replace(".json", "_details.json")
    with open(detail_file, "w") as f:
        json.dump(grade_results, f, indent=4)
        
    print("\n>>> Evaluation Results <<<")
    print(json.dumps(metrics, indent=4))
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmvet_path", required=True)
    parser.add_argument("--result_file", required=True, help="Input prediction json")
    parser.add_argument("--output_path", required=True, help="Output score json")
    parser.add_argument("--openai_api_key", default=None) 
    
    args = parser.parse_args()
    
    run_grading(args.mmvet_path, args.result_file, args.output_path)