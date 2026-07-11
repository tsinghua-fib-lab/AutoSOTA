#!/usr/bin/env python3
import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from steering_utils.llm_layers import get_llm_layers_module


def load_model_generic(args):
    print(f"Loading model from {args.model_path}...")
    model_path_lower = args.model_path.lower()
    
    if "qwen2.5" in model_path_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    elif "next" in model_path_lower or "v1.6" in model_path_lower:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        processor = LlavaNextProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
        
    return model, processor

def get_model_inputs(model, processor, image_input, text_prompt):
    model_type = getattr(model.config, "model_type", "").lower()
    is_qwen = "qwen2" in model_type
    
    if is_qwen:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": text_prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(model.device)
    else:
        pil_image = Image.open(image_input).convert("RGB") if isinstance(image_input, str) else image_input
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}]
        try: text = processor.apply_chat_template(messages, add_generation_prompt=True)
        except: text = f"USER: <image>\n{text_prompt}\nASSISTANT:" 
        inputs = processor(text=text, images=[pil_image], padding=True, return_tensors="pt").to(model.device)
    return inputs


class DiagnosticsHook:
    def __init__(self, vectors, layer_idx):
        self.v_vis = vectors.get("visual")
        self.captured_risk = None
        self.handle = None

    def _compute_risk(self, h):
        """Equation: R(h) = -CosSim(h, v_vis_perp)"""
        if self.v_vis is None: return 0.0
        v_v = F.normalize(self.v_vis.to(device=h.device, dtype=h.dtype), dim=-1)
        h_norm = F.normalize(h, dim=-1)
        return -torch.matmul(h_norm, v_v).item()

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.captured_risk is None:
            last_token_h = h[:, -1, :] 
            self.captured_risk = self._compute_risk(last_token_h)
        return output

    def attach(self, module): self.handle = module.register_forward_hook(self.hook_fn)
    def detach(self): 
        if self.handle: self.handle.remove()
    def clear(self): self.captured_risk = None

# Calibration Logic 
def process_layer_calibration(records, layer_idx, output_dir, k):
    """
    Calculates L* criteria (Eq 4) and Tau (Eq 5).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # S_fact: Risks from correct responses
    fact_risks = [r['risk'] for r in records if r['category'] in ['TP', 'TN']]
    # H_hall: Risks from hallucinated responses
    hall_risks = [r['risk'] for r in records if r['category'] == 'FP']
    
    if not fact_risks or not hall_risks:
        return None

    avg_fact = np.mean(fact_risks)
    avg_hall = np.mean(hall_risks)
    separability = avg_hall - avg_fact

    # Tau = Percentile(S_fact, k)
    tau = np.percentile(fact_risks, k)

    # Visualization for reference
    plt.figure(figsize=(8, 5))
    sns.kdeplot(fact_risks, label='Factual', fill=True, color='green')
    sns.kdeplot(hall_risks, label='Hallucination', fill=True, color='red')
    plt.axvline(tau, color='purple', linestyle='--', label=f'Tau (k={k})')
    plt.title(f"Layer {layer_idx} | Sep: {separability:.3f}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_calib.png"))
    plt.close()

    return {"separability": separability, "tau": tau}

def analyze_single_layer(args, model, processor, questions, vectors, l_idx):
    llm_layers = get_llm_layers_module(model)
    # Vectors are often saved as layer_idx + 1 in your previous logic
    v_idx = l_idx + 1
    if v_idx not in vectors: return None

    observer = DiagnosticsHook(vectors[v_idx], l_idx)
    observer.attach(llm_layers[l_idx])
    
    records = []
    subset = questions[:args.max_samples]
    
    for item in tqdm(subset, desc=f"Evaluating Layer {l_idx}", leave=False):
        observer.clear()
        img_path = os.path.join(args.image_folder, item['image'])
        inputs = get_model_inputs(model, processor, img_path, item['text'] + " Please answer yes or no.").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
        gen_text = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        pred = 'yes' if 'yes' in gen_text else ('no' if 'no' in gen_text else 'unknown')
        gt = item['label'].lower()

        category = "Other"
        if gt == 'yes' and pred == 'yes': category = "TP"
        elif gt == 'no' and pred == 'yes': category = "FP" 
        elif gt == 'no' and pred == 'no': category = "TN"
        
        if observer.captured_risk is not None:
            records.append({"risk": observer.captured_risk, "category": category})

    observer.detach()
    return process_layer_calibration(records, l_idx, args.output_dir, args.k_percentile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_file", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="./auto_calib_results")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--k_percentile", type=float, default=90.0)
    parser.add_argument("--min_layer", type=int, default=10, help="Stop searching after this layer")
    args = parser.parse_args()
    
    model, processor = load_model_generic(args)
    vectors = torch.load(args.vector_file, map_location='cpu')
    with open(args.question_file) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    
    llm_layers = get_llm_layers_module(model)
    total_layers = len(llm_layers)
    
    print(f"\n[Backward Search] Starting from Layer {total_layers-1} down to {args.min_layer}...")
    
    final_layer = None
    final_tau = None

    # L -> 1
    for l_idx in range(total_layers - 1, args.min_layer - 1, -1):
        res = analyze_single_layer(args, model, processor, questions, vectors, l_idx)
        
        if res and res["separability"] > 0:
            final_layer = l_idx
            final_tau = res["tau"]
            print(f"\nFound Optimal Layer L* = {final_layer}")
            print(f"Calibrated Threshold Tau = {final_tau:.4f}")
            print(f"Separability (R_hall - R_fact) = {res['separability']:.4f}")
            break
        else:
            sep = res['separability'] if res else "N/A"
            print(f"Layer {l_idx}: Separability {sep} <= 0. Moving up...")

    if final_layer is not None:
        print("\n" + "="*40)
        print(f"CALIBRATION COMPLETE")
        print(f"Use L*={final_layer} and Tau={final_tau:.4f} for intervention.")
        print("="*40)
    else:
        print("\nSearch Failed: No layer satisfied the positive separability constraint.")

if __name__ == "__main__":
    main()