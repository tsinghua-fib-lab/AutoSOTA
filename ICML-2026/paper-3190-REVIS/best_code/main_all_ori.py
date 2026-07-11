#!/usr/bin/env python3
import os
import argparse
import json
import torch
import string
from tqdm import tqdm
from PIL import Image

# HuggingFace Libraries
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq
)

# Qwen-VL Utility
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Metric Utils
from utils.mme import calculate_mme_metric
from utils.mmvet_ori import calculate_mmvet_metric
from utils.chair import calculate_chair_metric
from utils.hallusion_bench import evaluate_hallusion_bench
from utils.mmhal_bench import calculate_mmhal_metric

try:
    from transformers import Qwen3VLForConditionalGeneration
except:
    Qwen3VLForConditionalGeneration = None
# ==============================================================================
# 1. Model Loading
# ==============================================================================
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model_path_lower = model_path.lower()
    
    # --- A. Qwen2.5-VL ---
    if "qwen2.5" in model_path_lower:
        print("Using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    # --- B. LLaVA-NeXT (v1.6) ---
    elif "next" in model_path_lower or "v1.6" in model_path_lower:
        print("Using LlavaNextForConditionalGeneration")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    # --- C. LLaVA-1.5 ---
    elif "llava-1.5" in model_path_lower:
        print("Using LlavaForConditionalGeneration")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    # --- D. InternVL ---
    elif "internvl" in model_path_lower:
        print("Using AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        # if processor.tokenizer.pad_token is None:
        #     processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif "qwen3" in model_path_lower:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    # --- E. Fallback ---

    else:
        print("Warning: Unknown model type, falling back to AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        
    return model, processor

def get_model_inputs(model, processor, image, text_prompt):
    model_type = getattr(model.config, "model_type", "").lower()
    is_qwen = "qwen2" in model_type
    is_qwen3 = "qwen3" in model_type
    
    # Ensure RGB
    if isinstance(image, Image.Image) and image.mode != "RGB":
        image = image.convert("RGB")
    
    if is_qwen:
        # Qwen Logic
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if process_vision_info:
            img_in, vid_in = process_vision_info(messages)
            inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    elif is_qwen3:
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": text_prompt},
                        ],
                    }]
        inputs = processor.apply_chat_template(
                                                    messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt"
                                                )
    else:
        # LLaVA / InternVL Logic
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}]
        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
        except:
            text = f"USER: <image>\n{text_prompt}\nASSISTANT:" 
            
        inputs = processor(text=text, images=image, padding=True, return_tensors="pt").to(model.device)
        
    return inputs, is_qwen

# ==============================================================================
# 2. Utils
# ==============================================================================
def parse_pope_answer(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    if 'yes' in words and 'no' in words: return 'yes' if text.find('yes') < text.find('no') else 'no'
    if 'yes' in words: return 'yes'
    if 'no' in words: return 'no'
    return 'unknown'

def evaluate_pope_metrics(pred_file, gt_file):
    with open(pred_file) as f: preds = [json.loads(l) for l in f if l.strip()]
    with open(gt_file) as f: gts = [json.loads(l) for l in f if l.strip()]
    
    pred_map = {x['question_id']: parse_pope_answer(x['text']) for x in preds}
    gt_map = {x['question_id']: x['label'].lower() for x in gts}
    
    tp, fp, tn, fn, total = 0, 0, 0, 0, 0
    for qid, label in gt_map.items():
        if qid not in pred_map: continue
        pred = pred_map[qid]
        total += 1
        if pred == 'unknown': continue
        if label == 'yes':
            if pred == 'yes': tp += 1
            else: fn += 1
        else:
            if pred == 'no': tn += 1
            else: fp += 1
    if total == 0: return {}
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"Acc": round(acc, 4), "F1": round(f1, 4), "Prec": round(prec, 4), "Rec": round(rec, 4)}

# ==============================================================================
# 3. Main Benchmark Logic
# ==============================================================================
def run_benchmark(args):
    # 1. Load Model
    model, processor = load_model(args.model_path)
    
    # 2. Setup Output Dir
    save_dir = os.path.join(args.results_dir, args.benchmark, "original")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # 3. Define Inference Function
    def inference_adapter(image_path, question):
        try:
            if not os.path.exists(image_path): return "Error: Image Not Found"
            pil_image = Image.open(image_path).convert("RGB")
            
            # Prompt Tuning per Benchmark
            if args.benchmark == "pope":
                # prompt = question + " Please answer yes or no."
                prompt = question + "Please answer this question with one word."

                max_tokens = 1024 # max token = 10
            elif args.benchmark == "mme":
                prompt = question # MME usually has prompt inside
                max_tokens = 10
            elif args.benchmark == "chair":
                prompt = "Describe this image in detail."
                max_tokens = 512
            elif args.benchmark == "mmvet":
                prompt = question
                max_tokens = 1024
            elif args.benchmark == "hallusion":
                prompt = question # HallusionBench uses direct questions
                max_tokens = 128
            elif args.benchmark == "mmhal":  
                prompt = question
                max_tokens = 1024
            else:
                prompt = question
                max_tokens = 128

            inputs, is_qwen = get_model_inputs(model, processor, pil_image, prompt)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            
            if is_qwen:
                # Qwen outputs include input, strip it
                gen_text = processor.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            else:
                gen_text = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            return gen_text
        except Exception as e:
            print(f"Inference Error: {e}")
            return "Error"

    # 4. Dispatch Benchmark
    
    # --- POPE ---
    if args.benchmark == "pope":
        summary = {}
        for subset in ["random", "popular", "adversarial"]:
            print(f"Processing POPE {subset}...")
            q_file = os.path.join(args.question_dir, f"coco_pope_{subset}.json")
            if not os.path.exists(q_file): continue
            
            out_file = os.path.join(save_dir, f"pope_{subset}.jsonl")
            with open(q_file) as f: questions = [json.loads(l) for l in f if l.strip()]
            
            results = []
            for item in tqdm(questions):
                img_path = os.path.join(args.image_folder, item['image'])
                ans = inference_adapter(img_path, item['text'])
                results.append({"question_id": item['question_id'], "text": ans})
            
            with open(out_file, 'w') as f:
                for r in results: f.write(json.dumps(r) + '\n')
            
            metrics = evaluate_pope_metrics(out_file, q_file) # GT is usually the Q file itself in this format
            summary[subset] = metrics
            print(f"Scores: {metrics}")
        
        with open(os.path.join(save_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

    # --- MME ---
    elif args.benchmark == "mme":
        calculate_mme_metric(
            inference_fn=inference_adapter,
            mme_data_dir=args.question_dir,
            results_dir=save_dir
        )

    # --- MM-Vet ---
    elif args.benchmark == "mmvet":
        calculate_mmvet_metric(
            inference_fn=inference_adapter,
            mmvet_path=args.question_dir,
            results_dir=save_dir
        )

    # --- CHAIR ---
    elif args.benchmark == "chair":
        print("Running CHAIR...")
        chair_list = [json.loads(l) for l in open(args.question_dir)]
        results = []
        for item in tqdm(chair_list):
            image_file = item.get("image", item.get("image_file"))
            img_path = os.path.join(args.image_folder, image_file)
            
            # ID Extract Logic
            img_id = item.get("image_id", -1)
            if img_id == -1: # Try filename
                try: img_id = int(os.path.splitext(os.path.basename(image_file))[0].split('_')[-1])
                except: pass
            
            cap = inference_adapter(img_path, "")
            results.append({"image_id": img_id, "caption": cap})
        
        out_file = os.path.join(save_dir, "chair_captions.jsonl")
        with open(out_file, 'w') as f:
            for r in results: f.write(json.dumps(r) + '\n')
        
        calculate_chair_metric(out_file, args.gt_dir)
            print(f"MMHal Results: {json.dumps(metrics, indent=4)}")

    print("Benchmark Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benchmark", required=True, choices=["pope", "mme", "mmvet", "chair"])
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--question_dir", required=True, help="Data Root")
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--gt_dir", default="")
    
    args = parser.parse_args()
    run_benchmark(args)