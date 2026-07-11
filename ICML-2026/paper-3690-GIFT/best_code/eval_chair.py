#!/usr/bin/env python3
"""
CHAIR evaluation with GIFT for LLaVA-1.5-7B.
Evaluates hallucination in image captions using CHAIRs and CHAIRi metrics.

Usage:
    python eval_chair.py --config configs/chair_llava_1.5_7b.yaml
"""

import os
import json
import argparse
import logging
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import spacy
import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Constants
CONTENT_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
MODEL2ID = {
    "llava_1.5_7b": "llava-hf/llava-1.5-7b-hf",
    "llava_1.5_13b": "llava-hf/llava-1.5-13b-hf",
    "qwen2_vl_7b": "Qwen/Qwen2-VL-7B-Instruct"
}

# MSCOCO 80 object categories (used by CHAIR)
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Strict synonym mapping for COCO categories (CHAIR standard).
# We match exact category names and minimal well-known synonyms.
# Overly broad synonyms (e.g., "table" for "dining table", "ball" for "sports ball",
# "bag" for "handbag", "plant" for "potted plant") are excluded because they
# produce excessive false positives in general image descriptions.
COCO_SYNONYMS = {
    'person': ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'lady', 'human', 'men', 'women', 'children', 'boys', 'girls'],
    'bicycle': ['bicycle', 'bicycles', 'bike', 'bikes'],
    'car': ['car', 'cars', 'automobile', 'automobiles'],
    'motorcycle': ['motorcycle', 'motorcycles', 'motorbike'],
    'airplane': ['airplane', 'airplanes', 'aeroplane', 'plane', 'planes', 'aircraft', 'jet'],
    'bus': ['bus', 'buses'],
    'train': ['train', 'trains', 'locomotive'],
    'truck': ['truck', 'trucks', 'lorry'],
    'boat': ['boat', 'boats', 'ship', 'ships', 'sailboat', 'canoe', 'kayak'],
    'traffic light': ['traffic light', 'traffic lights', 'stoplight'],
    'fire hydrant': ['fire hydrant', 'fire hydrants', 'hydrant'],
    'stop sign': ['stop sign', 'stop signs'],
    'parking meter': ['parking meter', 'parking meters'],
    'bench': ['bench', 'benches'],
    'bird': ['bird', 'birds'],
    'cat': ['cat', 'cats', 'kitten', 'kittens'],
    'dog': ['dog', 'dogs', 'puppy', 'puppies'],
    'horse': ['horse', 'horses'],
    'sheep': ['sheep'],
    'cow': ['cow', 'cows', 'cattle', 'bull'],
    'elephant': ['elephant', 'elephants'],
    'bear': ['bear', 'bears'],
    'zebra': ['zebra', 'zebras'],
    'giraffe': ['giraffe', 'giraffes'],
    'backpack': ['backpack', 'backpacks', 'rucksack'],
    'umbrella': ['umbrella', 'umbrellas', 'parasol'],
    'handbag': ['handbag', 'handbags', 'purse', 'purses'],
    'tie': ['tie', 'ties', 'necktie'],
    'suitcase': ['suitcase', 'suitcases', 'luggage'],
    'frisbee': ['frisbee', 'frisbees'],
    'skis': ['skis', 'ski'],
    'snowboard': ['snowboard', 'snowboards'],
    'sports ball': ['sports ball', 'sports balls', 'soccer ball', 'basketball', 'tennis ball', 'baseball', 'volleyball', 'football'],
    'kite': ['kite', 'kites'],
    'baseball bat': ['baseball bat', 'baseball bats'],
    'baseball glove': ['baseball glove', 'baseball gloves', 'mitt'],
    'skateboard': ['skateboard', 'skateboards'],
    'surfboard': ['surfboard', 'surfboards'],
    'tennis racket': ['tennis racket', 'tennis rackets', 'tennis racquet', 'racket', 'racquet'],
    'bottle': ['bottle', 'bottles'],
    'wine glass': ['wine glass', 'wine glasses', 'wineglass'],
    'cup': ['cup', 'cups', 'mug', 'mugs', 'teacup', 'coffee cup'],
    'fork': ['fork', 'forks'],
    'knife': ['knife', 'knives'],
    'spoon': ['spoon', 'spoons'],
    'bowl': ['bowl', 'bowls'],
    'banana': ['banana', 'bananas'],
    'apple': ['apple', 'apples'],
    'sandwich': ['sandwich', 'sandwiches', 'burger', 'hamburger'],
    'orange': ['orange', 'oranges'],
    'broccoli': ['broccoli'],
    'carrot': ['carrot', 'carrots'],
    'hot dog': ['hot dog', 'hot dogs', 'hotdog'],
    'pizza': ['pizza', 'pizzas'],
    'donut': ['donut', 'donuts', 'doughnut'],
    'cake': ['cake', 'cakes', 'cupcake', 'pastry'],
    'chair': ['chair', 'chairs', 'stool', 'stools'],
    'couch': ['couch', 'couches', 'sofa', 'sofas', 'loveseat', 'futon'],
    'potted plant': ['potted plant', 'potted plants'],
    'bed': ['bed', 'beds'],
    'dining table': ['dining table', 'dining tables'],
    'toilet': ['toilet', 'toilets'],
    'tv': ['tv', 'television', 'televisions'],
    'laptop': ['laptop', 'laptops', 'notebook', 'notebooks'],
    'mouse': ['mouse', 'computer mouse'],
    'remote': ['remote', 'remotes', 'remote control'],
    'keyboard': ['keyboard', 'keyboards'],
    'cell phone': ['cell phone', 'cell phones', 'cellphone', 'smartphone', 'phone'],
    'microwave': ['microwave', 'microwaves'],
    'oven': ['oven', 'ovens', 'stove', 'stoves'],
    'toaster': ['toaster', 'toasters'],
    'sink': ['sink', 'sinks', 'basin'],
    'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges'],
    'book': ['book', 'books'],
    'clock': ['clock', 'clocks'],
    'vase': ['vase', 'vases'],
    'scissors': ['scissors', 'shears'],
    'teddy bear': ['teddy bear', 'teddy bears', 'teddy'],
    'hair drier': ['hair drier', 'hair driers', 'hair dryer', 'hairdryer', 'blow dryer'],
    'toothbrush': ['toothbrush', 'toothbrushes'],
}

# Build reverse mapping: synonym word -> category
SYNONYM_TO_CATEGORY = {}
for category, synonyms in COCO_SYNONYMS.items():
    for syn in synonyms:
        if syn not in SYNONYM_TO_CATEGORY:
            SYNONYM_TO_CATEGORY[syn] = category
        else:
            # If already mapped, keep existing (first-come-first-served, which handles overlaps)
            pass

# Words to skip (determiners, etc.)
SKIP_WORDS = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'many', 'several',
              'few', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
              'each', 'every', 'all', 'both', 'other', 'another', 'no', 'any'}


def load_yaml_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CHAIR evaluation with GIFT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def setup_environment(seed: int = 42):
    """Setup random seeds and CUDA environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model_and_processor(model_name: str, device: torch.device, model_path: str = None):
    """Load model and processor."""
    if model_path:
        model_id = model_path
    else:
        model_id = MODEL2ID[model_name]
    logger.info(f'Loading model from {model_id}')

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def extract_and_align(sentence: str, nlp, processor, model_name: str) -> List[int]:
    """Extract and align content words with token indices."""
    doc = nlp(sentence)
    words = [(token.text, token.idx, token.idx + len(token.text), token.pos_)
             for token in doc if token.pos_ in CONTENT_TAGS]

    encoding = processor(
        sentence,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding.offset_mapping[0]
    word_to_token_idxs = {}

    for word_text, w_start, w_end, pos in words:
        token_idxs = [
            i for i, offset in enumerate(offsets)
            if not (offset[1] <= w_start or offset[0] >= w_end)
        ]
        word_to_token_idxs[word_text + f" ({pos})"] = token_idxs

    return [token_id for word, ids in word_to_token_idxs.items() for token_id in ids]


def prepare_llava_inputs(query: str, image: Image.Image, processor, device: torch.device):
    """Prepare inputs for LLaVA model."""
    conversation = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = image.convert("RGB")

    return processor(
        images=image,
        text=prompt,
        return_tensors='pt',
        padding=True
    ).to(device, torch.float16)


def get_token_indices(inputs: Dict, model_name: str) -> Dict:
    """Get token indices for image and query."""
    indices = torch.where(inputs['input_ids'][0] == 32000)
    vision_start_token_index = indices[0][0].item()
    vision_end_token_index = indices[0][-1].item()

    query_start_token_index = vision_end_token_index + 3
    query_end_token_index = len(inputs['input_ids'][0]) - 6

    return {
        "vision_start_idx": vision_start_token_index,
        "vision_end_idx": vision_end_token_index,
        "query_start_idx": query_start_token_index,
        "query_end_idx": query_end_token_index
    }


def process_attention_maps(output, token_indices, query, processor, nlp, configs) -> torch.Tensor:
    """Process attention maps from model output for saliency map computation."""
    attn = [output["attentions"][0][l] for l in configs["visual_saliency_computation_layers"]]
    attn = torch.stack(attn).squeeze(1)
    # Compute positive attention shifts (gaze shifts)
    attn = attn[:, :,
                token_indices["query_start_idx"]:token_indices["query_end_idx"] + 1,
                token_indices["vision_start_idx"]:token_indices["vision_end_idx"] + 1] - \
           attn[:, :,
                token_indices["query_start_idx"] - 1:token_indices["query_end_idx"],
                token_indices["vision_start_idx"]:token_indices["vision_end_idx"] + 1]

    attn[attn < 0.0] = 0.0

    # Get token indices for content words
    selected_query_tokens = extract_and_align("\n" + query, nlp, processor, configs["model_name"])
    selected_query_tokens = [token_id - 2 for token_id in selected_query_tokens]

    if len(selected_query_tokens) == 0:
        # Fallback: use all query tokens
        selected_query_tokens = list(range(attn.size(2)))

    assert selected_query_tokens == [] or selected_query_tokens[0] >= 0, f"Negative token index: {selected_query_tokens}"
    attn = attn[:, :, selected_query_tokens]

    # Select top 50% attention heads
    k = max(1, attn.size(1) // 2)
    selected_heads = torch.topk(attn.sum(dim=(2, 3)), k=k, dim=-1).indices
    selected_heads = selected_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, attn.size(-2), attn.size(-1))
    attn = attn.gather(dim=1, index=selected_heads)

    return attn


def compute_attention_heatmap(attn: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute attention heatmap from attention tensor."""
    heatmap = attn.mean((0, 1, 2))

    # Normalize using 3-sigma rule
    mean, std = heatmap.mean(), heatmap.std()
    if std == 0:
        return torch.ones_like(heatmap)
    lower, upper = mean - 3 * std, mean + 3 * std
    heatmap = torch.clamp(heatmap, min=lower.item(), max=upper.item())

    # Min-max normalization and scaling
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin == 0:
        return torch.ones_like(heatmap)
    heatmap = ((heatmap - hmin) / (hmax - hmin)) * scale

    return torch.exp(heatmap)


def compute_visual_saliency_map(model, inputs, token_indices, query, nlp, processor, configs):
    """Compute visual saliency map using gaze shift tracking."""
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
        )

        attn = process_attention_maps(output, token_indices, query, processor, nlp, configs)
        return compute_attention_heatmap(attn, configs["alpha"])


def parse_objects_from_caption(caption: str) -> Set[str]:
    """
    Parse a caption to identify mentioned MSCOCO objects.
    Returns a set of MSCOCO category names found in the caption.
    """
    caption_lower = caption.lower()
    mentioned = set()

    # Remove punctuation for word matching
    caption_clean = re.sub(r'[^\w\s]', ' ', caption_lower)
    words = caption_clean.split()

    # Check multi-word synonyms first (longest matches)
    # Sort synonym phrases by length (longest first)
    sorted_syns = sorted(SYNONYM_TO_CATEGORY.keys(), key=len, reverse=True)

    matched_spans = set()  # Track character spans that have been matched

    for syn in sorted_syns:
        # Find all occurrences
        pattern = re.compile(r'\b' + re.escape(syn) + r'\b')
        for match in pattern.finditer(caption_lower):
            start, end = match.span()
            # Check if this span overlaps with already matched spans
            if any(not (end <= ms or start >= me) for ms, me in matched_spans):
                continue
            category = SYNONYM_TO_CATEGORY[syn]
            mentioned.add(category)
            matched_spans.add((start, end))

    return mentioned


def load_coco_ground_truth(annotations_path: str, image_dir: str) -> Dict[str, Set[str]]:
    """
    Load MSCOCO instance annotations and build a mapping:
    image_filename -> set of ground truth object categories present in the image.
    """
    import json
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Build category id -> name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Build image id -> filename mapping
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # Build image id -> set of category names
    img_id_to_cats = defaultdict(set)
    for ann in coco_data['annotations']:
        img_id_to_cats[ann['image_id']].add(cat_id_to_name[ann['category_id']])

    # Build filename -> set of category names (only for images we have)
    existing_images = set(os.listdir(image_dir)) if os.path.exists(image_dir) else set()
    filename_to_cats = {}
    for img_id, filename in img_id_to_filename.items():
        if filename in existing_images:
            filename_to_cats[filename] = img_id_to_cats.get(img_id, set())

    return filename_to_cats


def compute_chair_metrics(captions: Dict[str, str], gt_objects: Dict[str, Set[str]]) -> Dict:
    """
    Compute CHAIRs and CHAIRi metrics.

    CHAIRs: proportion of captions with at least one hallucinated object.
    CHAIRi: proportion of hallucinated object instances among all mentioned instances.

    Args:
        captions: dict mapping image_filename -> generated_caption
        gt_objects: dict mapping image_filename -> set of ground truth MSCOCO category names

    Returns:
        dict with CHAIRs, CHAIRi, and detailed stats
    """
    total_captions = 0
    captions_with_hallucination = 0
    total_mentioned_objects = 0
    total_hallucinated_objects = 0

    per_image_results = []

    for filename, caption in captions.items():
        gt_cats = gt_objects.get(filename, set())
        mentioned_cats = parse_objects_from_caption(caption)

        hallucinated = mentioned_cats - gt_cats
        matched = mentioned_cats & gt_cats

        total_captions += 1
        total_mentioned_objects += len(mentioned_cats)
        total_hallucinated_objects += len(hallucinated)

        if len(hallucinated) > 0:
            captions_with_hallucination += 1

        per_image_results.append({
            "filename": filename,
            "caption": caption,
            "mentioned": list(mentioned_cats),
            "ground_truth": list(gt_cats),
            "hallucinated": list(hallucinated),
            "matched": list(matched),
        })

    chairs = (captions_with_hallucination / total_captions * 100) if total_captions > 0 else 0.0
    chairi = (total_hallucinated_objects / total_mentioned_objects * 100) if total_mentioned_objects > 0 else 0.0

    return {
        "CHAIRs": round(chairs, 1),
        "CHAIRi": round(chairi, 1),
        "total_captions": total_captions,
        "captions_with_hallucination": captions_with_hallucination,
        "total_mentioned_objects": total_mentioned_objects,
        "total_hallucinated_objects": total_hallucinated_objects,
        "per_image": per_image_results,
    }


def main():
    """Main function for CHAIR evaluation with GIFT."""
    args = parse_args()
    configs = load_yaml_config(args.config)

    model_name = configs["model_name"]
    caption_prompt = configs.get("caption_prompt", "Please describe this image in detail.")
    max_new_tokens = configs.get("max_new_tokens", 1024)

    logger.info(f"Starting CHAIR evaluation with model: {model_name}")
    logger.info(f"GIFT enabled: {configs.get('use_gift', True)}")

    # Setup
    device = setup_environment()

    # Load model
    model_path = configs.get("model_path", None)
    model, processor = load_model_and_processor(model_name, device, model_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Load COCO ground truth
    coco_ann_path = configs["coco_annotations_path"]
    image_dir = configs["image_dir"]
    logger.info(f"Loading COCO annotations from {coco_ann_path}")
    gt_objects = load_coco_ground_truth(coco_ann_path, image_dir)
    logger.info(f"Loaded ground truth for {len(gt_objects)} images")

    # Get list of images
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Optionally limit number of images
    max_images = configs.get("max_images", None)
    if max_images is not None:
        image_files = image_files[:max_images]

    logger.info(f"Evaluating on {len(image_files)} images")

    # Generate captions
    captions = {}
    output_path = configs.get("output_path", "outputs/chair_results.json")

    for img_file in tqdm(image_files, desc="Generating captions"):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            continue

        # Prepare inputs
        inputs = prepare_llava_inputs(caption_prompt, image, processor, device)
        token_indices = get_token_indices(inputs, model_name)

        # Compute visual saliency map if GIFT is enabled
        visual_saliency_map = None
        if configs.get("use_gift", True):
            visual_saliency_map = compute_visual_saliency_map(
                model, inputs, token_indices, caption_prompt, nlp, processor, configs
            )

        gift_configs = {
            "visual_saliency_map": visual_saliency_map,
            "attention_enhancement_layers": configs.get("attention_enhancement_layers", []),
        }
        gift_configs.update(token_indices)

        # Generate caption
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_attentions": False,
            }
            if configs.get("use_gift", True):
                gen_kwargs["gift_configs"] = gift_configs
            output = model.generate(**inputs, **gen_kwargs)

        decoded = processor.decode(
            output["sequences"][0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        captions[img_file] = decoded.strip()

    # Compute CHAIR metrics
    logger.info("Computing CHAIR metrics...")
    results = compute_chair_metrics(captions, gt_objects)

    # Save results
    output_dir = Path(os.path.dirname(output_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "config": configs,
        "metrics": {
            "CHAIRs": results["CHAIRs"],
            "CHAIRi": results["CHAIRi"],
            "total_captions": results["total_captions"],
            "captions_with_hallucination": results["captions_with_hallucination"],
            "total_mentioned_objects": results["total_mentioned_objects"],
            "total_hallucinated_objects": results["total_hallucinated_objects"],
        },
        "per_image": results["per_image"],
    }

    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    # Print final metrics
    logger.info("=" * 60)
    logger.info("CHAIR Evaluation Results:")
    logger.info(f"  CHAIRs: {results['CHAIRs']:.1f}%")
    logger.info(f"  CHAIRi: {results['CHAIRi']:.1f}%")
    logger.info(f"  Total captions: {results['total_captions']}")
    logger.info(f"  Captions with hallucinations: {results['captions_with_hallucination']}")
    logger.info(f"  Total mentioned objects: {results['total_mentioned_objects']}")
    logger.info(f"  Total hallucinated objects: {results['total_hallucinated_objects']}")
    logger.info(f"  Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
