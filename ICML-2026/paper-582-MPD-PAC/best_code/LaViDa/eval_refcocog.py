#!/usr/bin/env python3
"""Evaluate LaViDa + MPD-PAC on RefCOCOg val set."""
import os, sys, json, time, argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/models'

# Add repo path
sys.path.insert(0, '/repo/LaViDa')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy, warnings
warnings.filterwarnings('ignore')

from pycocoevalcap.eval import Bleu, Cider, Meteor, COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prior', type=float, default=0.3, help='MPS lambda')
    parser.add_argument('--rope', type=float, default=0.01, help='MRS beta')
    parser.add_argument('--k', type=int, default=3, help='PCA rank')
    parser.add_argument('--slope', type=float, default=12.0, help='MRS eta')
    parser.add_argument('--center', type=float, default=0.6, help='MRS tau_0')
    parser.add_argument('--max-new-tokens', type=int, default=8, help='Generation length')
    parser.add_argument('--step-per-block', type=int, default=4, help='Inference steps')
    parser.add_argument('--output', type=str, default='/repo/outputs/refcocog_results.json')
    args = parser.parse_args()

    print(f"Args: {args}")
    
    # Load model
    pretrained = "/models/lavida-llada-v1.0-instruct"
    model_name = "llada_ours"
    device = "cuda"
    device_map = "cuda:0"
    
    vision_kwargs = dict(
        mm_vision_tower="/models/siglip-so400m-patch14-384",
        mm_resampler_type=None, mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152, use_mm_proj=True
    )
    
    print("Loading model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map,
        vision_kwargs=vision_kwargs, torch_dtype='bfloat16'
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    print(f"Model loaded. Max length: {max_length}")
    
    # Load dataset
    print("Loading RefCOCOg dataset...")
    from datasets import load_dataset
    ds = load_dataset('lmms-lab/RefCOCOg', 'default', split='val', streaming=False)
    print(f"Dataset size: {len(ds)}")
    
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"Using {len(ds)} samples")
    
    # Setup conversation template
    conv_template = "llada"
    
    predictions = []
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(ds, desc="Generating")):
        # Draw bounding box on image
        bbox = sample['bbox']
        image = sample['image'].convert('RGB')
        draw = ImageDraw.Draw(image)
        bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        draw.rectangle(bbox_xy, outline='red', width=3)
        
        # Prepare prompt
        question = DEFAULT_IMAGE_TOKEN + "\nProvide a short description for this region."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Process image
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_img.to(dtype=torch.bfloat16, device=device) for _img in image_tensor]
        
        # Encode prompt
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]
        
        # Generate
        block_length = args.max_new_tokens  # must divide max_new_tokens
        result = model.generate(
            input_ids, images=image_tensor, image_sizes=image_sizes,
            do_sample=False, temperature=0.0,
            max_new_tokens=args.max_new_tokens, block_length=block_length,
            step_per_block=args.step_per_block, tokenizer=tokenizer,
            prefix_lm=False, verbose=False,
            schedule='shift', prior=args.prior, rope=args.rope,
            mode='sigmoid', slope=args.slope, center=args.center, k=args.k
        )
        
        # Decode
        if isinstance(result, tuple):
            cont = result[0]
        else:
            cont = result
        
        # Get only generated tokens
        prompt_len = input_ids.shape[1]
        gen_tokens = cont[0, prompt_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        # Clean leading mask tokens
        gen_text = gen_text.lstrip('!').strip()
        
        predictions.append({
            'image_id': idx,
            'caption': gen_text,
            'question_id': sample['question_id'],
            'references': sample['answer']
        })
        
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (len(ds) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(ds)}] {rate:.1f} samples/sec, ETA: {eta/60:.1f}min")
    
    elapsed = time.time() - start_time
    print(f"Generation complete: {len(predictions)} samples in {elapsed:.1f}s ({elapsed/len(predictions):.2f}s/sample)")
    
    # Compute metrics
    print("Computing metrics...")
    dataset = {"annotations": [], "images": []}
    stored_results = []
    ann_id = 0
    
    for pred in predictions:
        img_id = pred['image_id']
        stored_results.append({"image_id": img_id, "caption": pred['caption']})
        for ref_text in pred['references']:
            dataset["annotations"].append({"image_id": img_id, "caption": ref_text, "id": ann_id})
            ann_id += 1
        dataset["images"].append({"id": img_id})
    
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    coco_result = coco.loadRes(stored_results)
    
    metrics = {}
    for scorer_cls, metric_name in [(Bleu(4), 'Bleu_4'), (Bleu(4), 'Bleu_3'), (Bleu(4), 'Bleu_2'), (Bleu(4), 'Bleu_1'), (Meteor(), 'METEOR'), (Cider(), 'CIDEr')]:
        coco_eval = COCOEvalCap(coco, coco_result)
        imgIds = coco_eval.params["image_id"]
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = coco_eval.coco.imgToAnns[imgId]
            res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]
        
        tokenizer_ptb = PTBTokenizer()
        gts = tokenizer_ptb.tokenize(gts)
        res = tokenizer_ptb.tokenize(res)
        
        score, scores = scorer_cls.compute_score(gts, res)
        if isinstance(score, list):
            n = int(metric_name.split('_')[-1])
            score = score[n - 1]
        metrics[metric_name] = float(score) * 100  # Convert to percentage
        print(f"  {metric_name}: {metrics[metric_name]:.2f}")
    
    # Save results
    results = {
        'args': vars(args),
        'metrics': metrics,
        'num_samples': len(predictions),
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n===== RESULTS =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    print("===================")

if __name__ == '__main__':
    main()
