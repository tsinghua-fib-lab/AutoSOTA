#!/usr/bin/env python3
import os
import argparse
import json
import torch
import string
from tqdm import tqdm
from PIL import Image

# Import HuggingFace general classes
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForImageTextToText,
    Qwen2VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info

# Import local modules (Steering Core)
from steering_utils.vector_calculator import (
    calculate_das_vectors_qwen, 
    calculate_das_vectors_llava,
    calculate_das_vectors_internvl
)
from steering_utils.llm_layers import (
    get_llm_layers_module,
    MADRSteeringHook_visual_only
)

# Import evaluation tools
from utils.mme import calculate_mme_metric
from utils.mmvet_ori import calculate_mmvet_metric
from utils.chair import calculate_chair_metric
from utils.text_halu import calculate_ocr_metric 
from utils.mmhal_bench import calculate_mmhal_metric

# ==============================================================================
# 1. Model Loading & Input Adaptation
# ==============================================================================
def load_model_generic(args):
    print(f"Loading model from {args.model_path}...")
    model_path_lower = args.model_path.lower()
    
    # --- A. Qwen2.5-VL ---
    if "qwen2.5" in model_path_lower:
        print("Using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    elif "qwen2-" in model_path_lower:
        print("Using Qwen2_VLForConditionalGeneration")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    # --- B. LLaVA-NeXT (llava-v1.6) ---
    elif "next" in model_path_lower or "v1.6" in model_path_lower:
        print("Using LlavaNextForConditionalGeneration")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    # --- C. LLaVA-1.5 ---
    elif "llava-1.5" in model_path_lower:
        print("Using LlavaForConditionalGeneration")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    # --- D. InternVL ---
    elif "internvl" in model_path_lower:
        print("Using AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, 
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    else:
        print("Warning: Unknown model type, falling back to AutoModelForVision2Seq")
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
        
    return model, processor

def get_model_inputs(model, processor, image, text_prompt):
    """Unified input construction logic for different VLM architectures."""
    model_type = getattr(model.config, "model_type", "").lower()
    is_qwen = "qwen" in model_type
    
    if isinstance(image, Image.Image) and image.mode != "RGB":
        image = image.convert("RGB")
    
    if is_qwen:
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
    else:
        messages = [{"role": "user", "content": [
            {"type": "image"}, 
            {"type": "text", "text": text_prompt}
        ]}]
        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
        except:
            text = f"USER: <image>\n{text_prompt}\nASSISTANT:" 
            
        inputs = processor(text=text, images=image, padding=True, return_tensors="pt").to(model.device)
        
    return inputs, is_qwen

# ==============================================================================
# 2. POPE Specific Evaluation Logic
# ==============================================================================
def parse_pope_answer(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    has_yes = 'yes' in words
    has_no = 'no' in words
    if has_yes and has_no:
        return 'yes' if text.find('yes') < text.find('no') else 'no'
    if has_yes: return 'yes'
    if has_no: return 'no'
    return 'unknown'

def evaluate_pope_metrics(pred_file, gt_file):
    with open(pred_file) as f: preds = [json.loads(l) for l in f if l.strip()]
    with open(gt_file) as f: gts = [json.loads(l) for l in f if l.strip()]
    
    pred_map = {x['question_id']: parse_pope_answer(x['text']) for x in preds}
    gt_map = {x['question_id']: x['label'].lower() for x in gts}
    
    tp = fp = tn = fn = 0
    unk_count = 0
    total = 0
    
    for qid, label in gt_map.items():
        if qid not in pred_map: continue
        pred = pred_map[qid]
        total += 1
        if pred == 'unknown':
            unk_count += 1; continue
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
    yes_ratio = (tp + fp) / total
    
    return {
        "Acc": round(acc, 4), 
        "Prec": round(prec, 4), 
        "Rec": round(rec, 4), 
        "F1": round(f1, 4), 
        "Yes%": round(yes_ratio, 4), 
        "Unk%": round(unk_count/total, 4)
    }

# ==============================================================================
# 3. Core Running Logic (do_run)
# ==============================================================================
def parse_layer_str(layers_str, num_layers):
    if not layers_str or layers_str.lower() == "none": return set()
    if layers_str.lower() == "all": return set(range(num_layers))
    s = set()
    for p in layers_str.split(','):
        if '-' in p: 
            start, end = map(int, p.split('-'))
            s.update(range(start, end+1))
        else: 
            try: s.add(int(p))
            except: pass
    return s

def do_run(args):
    # 1. Load Model
    model, processor = load_model_generic(args)
    
    # 2. Load and Inject Steering Hook (Visual Only)
    hooks = []
    if args.vector_file and args.vector_file.lower() != "none":
        print(f"Loading vectors from {args.vector_file}")
        vectors = torch.load(args.vector_file, map_location='cpu')
        llm_layers = get_llm_layers_module(model)
        num_layers = len(llm_layers)
        
        vis_set = parse_layer_str(args.vis_layers, num_layers)
        target_layers = sorted(list(vis_set))
        
        print(f"\n[MADR Config] Model: {args.model_path}")
        print(f"[Layers] Visual Tracking: {args.vis_layers}")
        print(f"[Params] AlphaV: {args.alpha_visual} | Tau_Low: {args.tau_low}")
        
        for idx in target_layers:
            vector_idx = idx + 1 
            if vector_idx not in vectors: 
                continue
            
            cur_alpha_vis = args.alpha_visual if idx in vis_set else 0.0
            
            # Use the refactored visual-only class
            hook = MADRSteeringHook_visual_only(
                vectors=vectors[vector_idx], 
                layer_idx=idx, 
                alpha_vis=cur_alpha_vis, 
                tau_low=args.tau_low, 
                risk_gamma=args.risk_gamma
            )
            hook.attach(llm_layers[idx])
            hooks.append(hook)
    else:
        print("Running Baseline (No Steering).")

    # 3. Prepare result directory
    param_str = f"Vis{args.vis_layers}_aV{args.alpha_visual}_Gamma{args.risk_gamma}_TauL{args.tau_low}"
    if not args.vector_file or args.vector_file == "none": 
        param_str = "Baseline"
        
    save_dir = os.path.join(args.results_dir, args.benchmark, param_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save current configuration
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # ----------------------------------------------------------------
    # 4. Inference Adapter (Closure)
    # ----------------------------------------------------------------
    def inference_adapter(image_path, question):
        try:
            if not os.path.exists(image_path): return "Error: Image Not Found"
            pil_image = Image.open(image_path).convert("RGB")
            
            # Adjust Prompt and Max Tokens per Benchmark
            if args.benchmark == "pope":
                prompt = question + " Please answer yes or no."
                max_tokens = 10
            elif args.benchmark == "mme":
                prompt = question
                max_tokens = 10
            elif args.benchmark == "chair":
                prompt = "Describe this image in detail."
                max_tokens = 512
            elif args.benchmark == "mmvet":
                prompt = question
                max_tokens = 1024
            elif args.benchmark == "texthalu":
                prompt = question
                max_tokens = 512
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
                gen_text = processor.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            else:
                gen_text = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            return gen_text
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return "Error"

    # ----------------------------------------------------------------
    # 5. Benchmark Dispatcher
    # ----------------------------------------------------------------
    print(f"\n>>> Running Benchmark: {args.benchmark.upper()}")
    print(f">>> Results will be saved to: {save_dir}")

    try:
        if args.benchmark == "pope":
            summary = {}
            subsets = ["random", "popular", "adversarial"]
            for subset in subsets:
                print(f"\n--- Sub-task: {subset} ---")
                q_file = os.path.join(args.question_dir, f"coco_pope_{subset}.json")
                gt_file = os.path.join(args.gt_dir, f"coco_pope_{subset}.json")
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
                
                metrics = evaluate_pope_metrics(out_file, gt_file)
                print(f"Result {subset}: {metrics}")
                summary[subset] = metrics
            
            print("\nPOPE Summary:", json.dumps(summary, indent=2))
            with open(os.path.join(save_dir, "pope_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        elif args.benchmark == "mme":
            calculate_mme_metric(
                inference_fn=inference_adapter,
                mme_data_dir=args.question_dir, 
                results_dir=save_dir
            )

        elif args.benchmark == "mmvet":
            calculate_mmvet_metric(
                inference_fn=inference_adapter,
                mmvet_path=args.question_dir,
                results_dir=save_dir
            )
        
        elif args.benchmark == "chair":
            print("Generating CHAIR captions...")
            chair_list = [json.loads(l) for l in open(args.question_dir)]
            results = []
            
            for item in tqdm(chair_list):
                image_file = item.get("image", item.get("image_file"))
                img_path = os.path.join(args.image_folder, image_file)
                
                # Image ID extraction logic
                if "image_id" in item:
                    img_id = item["image_id"]
                else:
                    try:
                        base_name = os.path.basename(image_file)
                        name_no_ext = os.path.splitext(base_name)[0]
                        id_str = name_no_ext.split('_')[-1]
                        img_id = int(id_str)
                    except Exception:
                        img_id = item.get("question_id", -1)

                if not isinstance(img_id, int):
                    try: img_id = int(img_id)
                    except: pass 

                cap = inference_adapter(img_path, "") 
                results.append({"image_id": img_id, "caption": cap})
            
            out_file = os.path.join(save_dir, "chair_captions.jsonl")
            with open(out_file, 'w') as f:
                for r in results: f.write(json.dumps(r) + '\n')
            
            print("Calculating CHAIR Metrics...")
            calculate_chair_metric(out_file, args.gt_dir)

    finally:
        # Cleanup hooks and GPU memory
        for h in hooks: h.detach()
        torch.cuda.empty_cache()

# ==============================================================================
# 4. Compute Vectors
# ==============================================================================
def do_compute(args):
    model, processor = load_model_generic(args)
    model_path_lower = args.model_path.lower()
    
    if "qwen" in model_path_lower:
        vecs = calculate_das_vectors_qwen(model, processor, args.steering_data_file, args.steering_image_dir)
    elif "internvl" in model_path_lower:
        vecs = calculate_das_vectors_internvl(model, processor, args.steering_data_file, args.steering_image_dir)
    else:
        print("Using LLaVA vector calculation")
        vecs = calculate_das_vectors_llava(model, processor, args.steering_data_file, args.steering_image_dir)
        
    os.makedirs(os.path.dirname(args.vector_file), exist_ok=True)
    torch.save(vecs, args.vector_file)
    print(f"Vectors saved to {args.vector_file}")

# ==============================================================================
# 5. Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--model_path", default="./Qwen2.5-VL-7B-Instruct")

    # Compute Parser
    comp = subparsers.add_parser("compute", parents=[base])
    comp.add_argument("--steering_data_file", type=str, default="./data/OH_sampled_data_100.json")
    comp.add_argument("--steering_image_dir", type=str, default="./data/OH_sampled_images_100")
    comp.add_argument("--vector_file", type=str, default="./vectors/qwen2.5vl_none_image.pt")
    
    # Run Parser
    run = subparsers.add_parser("run", parents=[base])
    run.add_argument("--vector_file", required=True, help="Vectors file path, or 'none' for baseline")
    run.add_argument("--benchmark", type=str, required=True, choices=["pope", "mme", "mmvet", "chair", "texthalu", "mmhal"])
    run.add_argument("--results_dir", required=True, help="Base output directory")
    run.add_argument("--question_dir", required=True, help="Data Root for benchmarks")
    run.add_argument("--image_folder", type=str, default="", help="Images folder (e.g., COCO val2014)")
    run.add_argument("--gt_dir", type=str, default="", help="Ground truth directory")
    
    # Steering Parameters
    run.add_argument("--vis_layers", default="27")
    run.add_argument("--lang_layers", default="27") # Kept for arg compatibility
    run.add_argument("--alpha_visual", type=float, default=2.5)
    run.add_argument("--alpha_lang", type=float, default=1.0) # Kept for arg compatibility
    run.add_argument("--tau_high", type=float, default=0.6) # Kept for arg compatibility
    run.add_argument("--tau_low", type=float, default=0.2)
    run.add_argument("--risk_gamma", type=float, default=1.0)
    
    args = parser.parse_args()
    
    if args.command == "compute":
        do_compute(args)
    elif args.command == "run":
        do_run(args)