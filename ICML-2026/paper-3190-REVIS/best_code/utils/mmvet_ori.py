# utils/mmvet.py
import os
import json
import time
import re
from tqdm import tqdm
from collections import Counter
import os
import requests
import json
import time
QWEN_API_KEY = "YOUR_API_KEY"
QWEN_MODEL = "Qwen3-235B-A22B-Instruct-2507"#"Qwen3-235B-A22B-Instruct-2507" 
BASE_URL = "YOUR_URL"


def qwen_api_call(system_prompt: str,prompt: str, model: str = QWEN_MODEL,temperature: float = 0.25, max_tokens: int = 512) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "SOFA-TraceId": "trace_" + str(int(time.time())),
        "SOFA-RpcId": "0"
    }
    payload = {
        "model": model,
        "messages": [{
                    'role': 'system',
                    'content': system_prompt}, 
                    {"role": "user", "content": prompt}],
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

# --- MM-Vet Grading Prompt  ---
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


def run_mmvet_inference(inference_fn, mmvet_path, result_file):

    mmvet_json = os.path.join(mmvet_path, "mm-vet.json")
    images_dir = os.path.join(mmvet_path, "images")
    
    if not os.path.exists(mmvet_json):
        raise FileNotFoundError(f"MM-Vet json not found at {mmvet_json}")

    with open(mmvet_json, "r") as f:
        data = json.load(f)

    results = {}
    
    if os.path.exists(result_file):
        print(f"Result file {result_file} exists. Loading existing results...")
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
        except:
            print("Error loading existing file, starting fresh.")
            results = {}

    print(f"Starting Inference on MM-Vet ({len(data)} samples)...")

    # print(data)
    # data = data[:100] # for debug
    for sample_id, sample_info in tqdm(data.items(), desc="MM-Vet Inference"):
        if sample_id in results and results[sample_id] != "Error":
            continue

        question = sample_info["question"]
        image_name = sample_info["imagename"]
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            # print(f"Warning: Image {image_name} not found.")
            continue
            
        try:

            output_text = inference_fn(image_path, question)
            results[sample_id] = output_text

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            results[sample_id] = "Error"
        
        if len(results) % 10 == 0:
             with open(result_file, "w") as f:
                json.dump(results, f, indent=4)

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Inference done. Results saved to {result_file}")
    return results


def grade_mmvet_results(mmvet_path, result_file, output_dir):

    mmvet_json = os.path.join(mmvet_path, "mm-vet.json")
    with open(mmvet_json, "r") as f:
        data = json.load(f)
    
    with open(result_file, "r") as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    grade_file = os.path.join(output_dir, "mmvet_grades.json")
    
    if os.path.exists(grade_file):
        with open(grade_file, "r") as f:
            grade_results = json.load(f)
    else:
        grade_results = {}

    print(f"Starting Grading using utils.api (Total: {len(results)})...")

    for sample_id, sample_info in tqdm(data.items(), desc="LLM Grading"):
        if sample_id not in results: continue
        if sample_id in grade_results: continue 

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
                     raise ValueError(f"No score found in content: {content}")

            except Exception as e:
                if "Predict the correctness" not in user_content:
                    user_content += "\nPredict the correctness of the answer (digit only 0.0 to 1.0): "
            
                current_temperature = min(current_temperature + 0.2, 0.8)
                try_count += 1
                time.sleep(1) 
                continue

        grade_results[sample_id] = {
            "score": score,
            "prediction": model_pred,
            "content": content if success else "Failed"
        }

        if len(grade_results) % 10 == 0:
            with open(grade_file, "w") as f:
                json.dump(grade_results, f, indent=4)

    with open(grade_file, "w") as f:
        json.dump(grade_results, f, indent=4)

    return grade_results

def calculate_metrics(mmvet_path, grade_results):
    mmvet_json = os.path.join(mmvet_path, "mm-vet.json")
    with open(mmvet_json, "r") as f:
        data = json.load(f)
        
    scores = Counter()
    counts = Counter()
    caps = ["rec", "ocr", "know", "gen", "spat", "math"]
    
    for sample_id, grade_info in grade_results.items():
        score = grade_info["score"]
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
            
    return final_metrics

def calculate_mmvet_metric(inference_fn, mmvet_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    result_json_path = os.path.join(results_dir, "mmvet_predictions.json")
    
    print("\n>>> Step 1: Running MM-Vet Inference...")
    run_mmvet_inference(inference_fn, mmvet_path, result_json_path)

    print("\n>>> Step 2: Running Grading (via utils.api)...")
    grade_results = grade_mmvet_results(mmvet_path, result_json_path, results_dir)
    
    if grade_results:
        metrics = calculate_metrics(mmvet_path, grade_results)
        print("\n>>> MM-Vet Evaluation Results <<<")
        print(json.dumps(metrics, indent=4))
        
        with open(os.path.join(results_dir, "mmvet_summary.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        return metrics
    else:
        print("Grading failed, no results to calculate.")
        return {}