# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
VLM Inference with GIFT for hallucination mitigation.
This script performs inference using Vision Language Models with GIFT technique
to mitigate hallucinations in model outputs.
"""

import os
import json
import argparse
import logging
import pickle
import random
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import spacy
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

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

def load_yaml_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_configs() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLM Inference with GIFT for hallucination mitigation.")

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)    
    args = parser.parse_args()
    return load_yaml_config(args.config)

def setup_environment(seed: int = 42) -> torch.device:
    """Setup random seeds and CUDA environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def load_model_and_processor(model_name: str, device: torch.device) -> Tuple[torch.nn.Module, AutoProcessor]:
    """Load model and processor based on model name."""
    model_id = MODEL2ID[model_name]
    logger.info(f'Initializing {model_id}')
    
    if "qwen" in model_name:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
    elif "llava" in model_name:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
    else:
        raise ValueError(f"model name {model_name} not supported")

    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def extract_and_align(sentence: str, nlp, processor, model_name: str) -> List[int]:
    """Extract and align content words with token indices."""
    doc = nlp(sentence)
    words = [(token.text, token.idx, token.idx + len(token.text), token.pos_)
             for token in doc if token.pos_ in CONTENT_TAGS]
    if "qwen" in model_name:
        encoding = processor(
            text = [sentence],
            return_offsets_mapping=True,
            add_special_tokens=False
        )
    else:
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
    
    return [id for word, ids in word_to_token_idxs.items() for id in ids]

def prepare_inputs(query: str, image: Image.Image, processor, device: torch.device, model_name: str) -> Dict:
    """Prepare model inputs based on model type."""
    if "qwen" in model_name:
        return prepare_qwen_inputs(query, image, processor, device)
    elif "llava" in model_name:
        return prepare_llava_inputs(query, image, processor, device)
    else:
        raise ValueError(f"Model name {model_name} not supported")

def prepare_qwen_inputs(query: str, image: Image.Image, processor, device: torch.device) -> Dict:
    """Prepare inputs for Qwen model."""
    max_pixels = 1280 * 28 * 28
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image.convert("RGB"), "max_pixels": max_pixels},
            {"type": "text", "text": query},
        ],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

def prepare_llava_inputs(query: str, image: Image.Image, processor, device: torch.device) -> Dict:
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
    indices = torch.where(inputs['input_ids'][0] == (151655 if "qwen" in model_name else 32000))
    vision_start_token_index = indices[0][0].item()
    vision_end_token_index = indices[0][-1].item()
    
    if "qwen" in model_name:
        query_start_token_index = vision_end_token_index + 2
        query_end_token_index = len(inputs['input_ids'][0]) - 6
    elif "llava" in model_name:
        query_start_token_index = vision_end_token_index + 3
        query_end_token_index = len(inputs['input_ids'][0]) - 6
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    return {
        "vision_start_idx": vision_start_token_index,
        "vision_end_idx": vision_end_token_index,
        "query_start_idx": query_start_token_index,
        "query_end_idx": query_end_token_index
    }

def process_attention_maps(output: Dict, token_indices: Dict, query: str, processor, nlp, configs) -> torch.Tensor:
    """Process attention maps from model output."""
    # At first decoding step, extract attention from specified layers
    attn = [output["attentions"][0][l] for l in configs["visual_saliency_computation_layers"]]
    # Assume a batch size of 1
    attn = torch.stack(attn).squeeze(1)
    # Extract query-vision attention
    attn = attn[:, :, 
                token_indices["query_start_idx"]:token_indices["query_end_idx"] + 1,
                token_indices["vision_start_idx"]:token_indices["vision_end_idx"] + 1] - \
           attn[:, :,
                token_indices["query_start_idx"] - 1:token_indices["query_end_idx"],
                token_indices["vision_start_idx"]:token_indices["vision_end_idx"] + 1]
    
    attn[attn < 0.0] = 0.0
    
    # Get token indices for content words
    
    if "llava" in configs["model_name"]:
        selected_query_tokens = extract_and_align("\n" + query, nlp, processor, configs["model_name"])
        selected_query_tokens = [token_id - 2 for token_id in selected_query_tokens]
    else:
        selected_query_tokens = extract_and_align(query, nlp, processor, configs["model_name"])
    
    assert selected_query_tokens == [] or selected_query_tokens[0] >= 0
    attn = attn[:, :, selected_query_tokens]

    # Select top attention heads
    k = attn.size(1) // 2
    selected_heads = torch.topk(attn.sum(dim=(2, 3)), k=k, dim=-1).indices
    selected_heads = selected_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, attn.size(-2), attn.size(-1))
    attn = attn.gather(dim=1, index=selected_heads)
    
    return attn

def compute_attention_heatmap(attn: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute attention heatmap from attention tensor."""
    # Average across layers, heads and query tokens
    heatmap = attn.mean((0, 1, 2))
    
    # Normalize using 3-sigma rule
    mean, std = heatmap.mean(), heatmap.std()
    lower, upper = mean - 3 * std, mean + 3 * std
    heatmap = torch.clamp(heatmap, min=lower.item(), max=upper.item())
    
    # Min-max normalization and scaling
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())) * scale
    
    return torch.exp(heatmap)

def compute_visual_saliency_map(model, inputs: Dict, token_indices: Dict, query: str, nlp, processor, configs) -> Optional[torch.Tensor]:
    """Compute visual saliency map using attention allocated from query tokens to visual tokens."""   
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

def save_results(outputs: List[Dict], configs: Dict):
    """Save inference results to file."""
    final_outputs = {"outputs": outputs}
    final_outputs.update(configs)
    
    output_file = configs["output_path"]
    output_dir = Path(os.path.dirname(output_file))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(final_outputs, f)
    logger.info(f"Results saved to {output_file}")

def main():
    """Main execution function."""
    configs = parse_configs()
    
    model_name = configs["model_name"]
    logger.info(f"Starting inference with model: {model_name}")

    # Setup environment
    device = setup_environment()
    
    # Load models and processors
    model, processor = load_model_and_processor(model_name, device)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Load data
    logger.info(f"Loading data from {configs['data_path']}")
    with open(configs["data_path"], "rb") as f_r:
        data = pickle.load(f_r)
    logger.info(f"Loaded {len(data)} examples")
    
    # Process examples
    outputs = []
    logger.info("Starting evaluation...")
    

    for ex in tqdm(data, desc="Processing"):
        query = ex['question']
        image = ex['image']
        label = ex['answer']

        # Prepare inputs
        inputs = prepare_inputs(query, image, processor, device, model_name)
        
        # Get token indices
        token_indices = get_token_indices(inputs, model_name)
        
        # Process global attention if needed
        visual_saliency_map = None
        if configs["use_gift"]:
            visual_saliency_map = compute_visual_saliency_map(
                model, inputs, token_indices, query, nlp, processor, configs,
            )

        gift_configs = {
            "visual_saliency_map": visual_saliency_map,
            "attention_enhancement_layers": configs["attention_enhancement_layers"]
        }
        gift_configs.update(token_indices)
        # Generate output
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=configs["max_new_tokens"],
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=False,
                gift_configs=gift_configs
            )
        
        decoded_output = processor.decode(
            output["sequences"][0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        outputs.append({
            "label": label,
            "output": decoded_output,
            "question": query,
            "dataset": ex['dataset']
        })
    
    # Save results
    save_results(outputs, configs)
    logger.info("Inference completed successfully")

if __name__ == "__main__":
    main()
