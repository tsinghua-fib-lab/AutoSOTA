"""
vLLM accelerated VLM evaluation script
Supports batch inference for Qwen3-VL, InternVL, SailVL
Supports vlm_tagging (image captioning) and image_classification tasks
"""

import os
import json
import argparse

# Ensure vLLM subprocesses correctly inherit CUDA_VISIBLE_DEVICES
# vLLM v0.14+ uses V1 engine, need to ensure subprocesses can access CUDA devices
if "CUDA_VISIBLE_DEVICES" in os.environ:
    _cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"[INFO] CUDA_VISIBLE_DEVICES={_cuda_devices}")
else:
    print("[WARNING] CUDA_VISIBLE_DEVICES not set, vLLM will use all available GPUs")

# Set distributed environment variables to ensure single GPU operation
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")

# Disable vLLM V1 engine (if there are compatibility issues) - use V0 engine
# Uncomment the line below to force V0 engine
# os.environ["VLLM_USE_V1"] = "0"

# Set multiprocessing start method to spawn (recommended by vLLM)
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import pandas as pd

# Import datasets
from data import Flickr30kDataset, COCOCaptionDataset, CIFAR10Dataset, CIFAR100Dataset, ImageNet1kDataset

# Embedding model (lazy loading)
_embedding_model = None
_class_embeddings_cache = {}


def get_embedding_model():
    """Lazy load the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model for semantic matching...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_class_embeddings(class_names: list) -> np.ndarray:
    """Get class name embeddings (with caching)"""
    cache_key = tuple(class_names[:10])  # Use first 10 class names as cache key
    if cache_key not in _class_embeddings_cache:
        model = get_embedding_model()
        # Clean class names for embedding
        cleaned_names = [name.replace('_', ' ').replace('-', ' ') for name in class_names]
        _class_embeddings_cache[cache_key] = model.encode(cleaned_names, show_progress_bar=False)
    return _class_embeddings_cache[cache_key]


def calculate_semantic_match_scores(output: str, class_names: list) -> np.ndarray:
    """Hybrid matching strategy: string matching first, then embedding matching

    1. Prefer exact string matching (most reliable)
    2. If no good string match, use embedding semantic matching
    """
    # Try string matching first
    string_scores = calculate_match_scores(output, class_names)
    max_string_score = string_scores.max()

    # If string matching has high confidence result (>0.3), return directly
    if max_string_score > 0.3:
        return string_scores

    # Otherwise use embedding matching
    model = get_embedding_model()
    class_embeddings = get_class_embeddings(class_names)

    # Encode model output
    output_embedding = model.encode([output], show_progress_bar=False)[0]

    # Calculate cosine similarity
    similarities = np.dot(class_embeddings, output_embedding) / (
        np.linalg.norm(class_embeddings, axis=1) * np.linalg.norm(output_embedding) + 1e-8
    )

    # If string has medium match (0.05-0.3), combine both scores
    if max_string_score > 0.05:
        # Normalize embedding similarities
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max > sim_min:
            norm_similarities = (similarities - sim_min) / (sim_max - sim_min)
        else:
            norm_similarities = np.ones_like(similarities) / len(similarities)

        # Combine both scores: string matching has higher weight
        combined = 0.6 * string_scores + 0.4 * norm_similarities
        combined = combined / combined.sum()
        return combined

    # Pure embedding matching
    temperature = 0.05  # Sharper distribution
    exp_scores = np.exp((similarities - similarities.max()) / temperature)
    probs = exp_scores / exp_scores.sum()

    return probs


def image_to_base64(image: Image.Image, square_crop: bool = False) -> str:
    """Convert PIL image to base64 string

    Args:
        image: PIL image
        square_crop: Whether to crop to square (Ministral recommends aspect ratio close to 1:1)
    """
    if square_crop:
        # Crop image to square (take center region)
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        # Scale to reasonable size (avoid being too large)
        if min_dim > 768:
            image = image.resize((768, 768), Image.Resampling.LANCZOS)
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_vllm_evaluation(
    task_type: str,
    dataset_name: str,
    model_name: str,
    output_dir: str = "./outputs/vlm_tagging",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    num_samples: int = None
):
    """Run batch inference using vLLM"""

    from vllm import LLM, SamplingParams

    # Load dataset
    print(f"\n{'='*50}")
    print(f"Task: {task_type}")
    print(f"Loading dataset: {dataset_name}")
    if num_samples:
        print(f"⚠️ TEST MODE: Only processing {num_samples} samples")
    print(f"{'='*50}")
    
    # Dataset configuration (default sample count limits)
    dataset_limits = {
        "Flickr30k": 10000,
        "COCO": 10000,
        "CIFAR-10": 10000,
        "CIFAR-100": 10000,
        "ImageNet-1k": 10000
    }
    
    # If num_samples is specified, use the smaller value
    actual_samples = num_samples if num_samples else dataset_limits.get(dataset_name, 10000)
    
    # VLM Tagging task datasets
    if dataset_name == "Flickr30k":
        dataset = Flickr30kDataset(split='test', num_samples=actual_samples)
    elif dataset_name == "COCO":
        dataset = COCOCaptionDataset(split='val', num_samples=actual_samples)
    # Image classification task datasets
    elif dataset_name == "CIFAR-10":
        dataset = CIFAR10Dataset(split='test', num_samples=actual_samples)
    elif dataset_name == "CIFAR-100":
        dataset = CIFAR100Dataset(split='test', num_samples=actual_samples)
    elif dataset_name == "ImageNet-1k":
        dataset = ImageNet1kDataset(split='validation', num_samples=actual_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset.load_data()
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Model configuration
    MODEL_MAP = {
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",
        "sailvl-8b": "BytedanceDouyinContent/SAIL-VL2-8B",
        "ministral3-vl-8b": "mistralai/Ministral-3-8B-Instruct-2512",
        "pixtral-12b": "mistral-community/pixtral-12b",  # Community version, uses standard Llava architecture
        "glm-4.6v-flash": "zai-org/GLM-4.6V-Flash",  # Zhipu AI GLM-4.6V series lightweight model
        "gemma3-4b": "google/gemma-3-4b-it",  # Google Gemma 3 4B multimodal model
        "step3-vl-10b": "stepfun-ai/Step3-VL-10B",  # StepFun Step3-VL 10B vision-language model
    }
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}")
    
    model_path = MODEL_MAP[model_name]
    
    # Initialize vLLM
    print(f"\n{'='*50}")
    print(f"Loading vLLM model: {model_path}")
    print(f"{'='*50}")
    
    # vLLM configuration
    # ImageNet-1k class list is about 4000+ tokens, requires larger max_model_len
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": 8192,  # Increased to support ImageNet-1k long prompts
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "limit_mm_per_prompt": {"image": 1},  # Maximum one image per prompt
        "enforce_eager": True,  # Disable CUDA Graph to avoid multi-process CUDA initialization issues
    }
    
    # If encountering CUDA subprocess initialization issues, try disabling V1 engine
    # vLLM v0.14+ defaults to V1 engine, which may have CUDA inheritance issues in some environments
    use_v1 = os.environ.get("VLLM_USE_V1", "1")
    if use_v1 == "0":
        print("[INFO] Using vLLM V0 engine (VLLM_USE_V1=0)")
    
    # Step3-VL requires special handling: force V0 engine (V1 engine has compatibility issues)
    if "step3" in model_name.lower():
        print("[INFO] Step3-VL detected, forcing V0 engine for compatibility")
        os.environ["VLLM_USE_V1"] = "0"

    llm = LLM(**llm_kwargs)
    
    # Set different sampling parameters and prompts based on task type
    if task_type == "vlm_tagging":
        sampling_params = SamplingParams(
            max_tokens=18,
            temperature=0,
            stop=["\n", ".", "!"],  # Stop at end of sentence
        )
        text_prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        class_names = None
        use_semantic_matching = False
    elif task_type == "image_classification":
        class_names = dataset.class_names

        # Choose different strategies based on number of classes
        if len(class_names) > 100:
            # ImageNet-1k: Must list classes, but use concise format
            sampling_params = SamplingParams(
                max_tokens=15,  # Limit output length
                temperature=0,
            )
            # Build class list (same format as Qwen3VL)
            class_list = ", ".join([c.replace('_', ' ') for c in class_names])

            # Use prompt format similar to Qwen3VL
            text_prompt = f"""Classify this image into ONE of these categories:
{class_list}

Answer with ONLY the exact category name, nothing else."""
            use_semantic_matching = True
            # Pre-load embedding model and class embeddings
            print("Pre-loading embeddings for semantic matching...")
            _ = get_class_embeddings(class_names)
        else:
            # CIFAR-10/100: Few classes, can list them
            sampling_params = SamplingParams(
                max_tokens=15,
                temperature=0,
            )
            class_list = ", ".join([c.replace('_', ' ') for c in class_names])
            text_prompt = f"""Classify this image. Choose EXACTLY ONE from: {class_list}

Reply with ONLY the exact category name, nothing else."""
            use_semantic_matching = False
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Get processor to construct the correct prompt format
    # Ministral 3 VL uses MistralCommonBackend, pixtral uses AutoProcessor, other models also use AutoProcessor
    processor = None
    if "ministral" in model_name.lower():
        try:
            from transformers import MistralCommonBackend
            processor = MistralCommonBackend.from_pretrained(model_path)
            print(f"Using MistralCommonBackend for {model_name}")
        except Exception as e:
            print(f"Warning: Could not load MistralCommonBackend: {e}")
            print("Will use manual prompt construction for Ministral")
    elif "pixtral" in model_name.lower():
        # Pixtral uses Llava architecture, uses AutoProcessor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"Using AutoProcessor (Llava) for {model_name}")
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Prepare all inputs
    print(f"\nPreparing inputs...")
    all_inputs = []
    all_metadata = []
    
    for idx in tqdm(range(len(dataset)), desc="Preparing"):
        item = dataset[idx]
        image = item['input']
        
        # Use chat template to construct prompt
        if "qwen" in model_name.lower() or "sailvl" in model_name.lower():
            # Qwen3-VL / SAIL-VL2 format - uses apply_chat_template
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }]
            prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            all_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [image]}
            })
        elif "ministral" in model_name.lower():
            # Ministral 3 VL (Pixtral) format
            # vLLM uses chat format to handle multimodal input
            # Official recommendation: 1) Use system prompt 2) Image aspect ratio close to 1:1
            img_base64 = image_to_base64(image, square_crop=True)
            
            # Build system prompt based on task
            if task_type == "image_classification":
                system_content = "You are an expert biologist and object classifier with deep knowledge of animal breeds, bird species, dog breeds, and fine-grained visual differences. Identify objects with maximum specificity. For animals, identify the exact species or breed, not just the general category."
            else:
                system_content = "You are a helpful assistant that describes images concisely and accurately."
            
            all_inputs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
            })
        elif "pixtral" in model_name.lower():
            # Pixtral-12B community version (mistral-community/pixtral-12b)
            # Uses llm.chat() + OpenAI format

            # Image preprocessing
            processed_image = image.copy()
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')

            # Scale to reasonable size
            width, height = processed_image.size
            max_dim = max(width, height)
            if max_dim > 1024:
                scale = 1024 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed_image = processed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to base64
            img_base64 = image_to_base64(processed_image)

            # OpenAI format messages
            all_inputs.append({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": text_prompt}
                    ]
                }]
            })
        elif "gemma" in model_name.lower():
            # Gemma 3 format - uses apply_chat_template
            # Based on official docs: https://huggingface.co/google/gemma-3-4b-it
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }]
            prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            all_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [image]}
            })
        elif "step3" in model_name.lower():
            # Step3-VL-10B format - uses apply_chat_template
            # Based on official docs: https://huggingface.co/stepfun-ai/Step3-VL-10B
            # Need to convert image to base64 URL format
            img_base64 = image_to_base64(image)
            img_url = f"data:image/png;base64,{img_base64}"
            
            # Prompt to force English output
            english_prompt = f"[Reply in English only] {text_prompt}"
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": english_prompt}
                ]
            }]
            prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode
            )
            all_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [image]}
            })
        else:
            # InternVL format
            all_inputs.append({
                "prompt": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [image]}
            })
        
        # Prepare metadata based on task type
        if task_type == "vlm_tagging":
            true_captions = item['label']
            if not isinstance(true_captions, list):
                true_captions = [true_captions]
            all_metadata.append({
                'index': idx,
                'true_captions': true_captions,
                'metadata': item.get('metadata', {})
            })
        elif task_type == "image_classification":
            all_metadata.append({
                'index': idx,
                'true_label': item['label'],
                'true_label_name': class_names[item['label']] if class_names else str(item['label']),
                'metadata': item.get('metadata', {})
            })
    
    # Batch inference
    import time
    total_batches = (len(all_inputs) + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print(f"Running batch inference with vLLM...")
    print(f"Total samples: {len(all_inputs)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"{'='*60}\n")
    
    results = []
    start_time = time.time()
    
    pbar = tqdm(
        range(0, len(all_inputs), batch_size), 
        desc="Generating",
        total=total_batches,
        unit="batch",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i in pbar:
        batch_start = time.time()
        batch_inputs = all_inputs[i:i+batch_size]
        batch_metadata = all_metadata[i:i+batch_size]
        
        # vLLM batch generation
        if "pixtral" in model_name.lower() or "ministral" in model_name.lower():
            # Pixtral/Ministral uses chat method
            # Force openai content format to correctly parse image_url
            
            # Debug: print the first message structure of the first batch
            if i == 0:
                first_msg = batch_inputs[0]["messages"][0]
                print(f"\n[DEBUG] First message structure:")
                print(f"  role: {first_msg['role']}")
                print(f"  content types: {[c['type'] for c in first_msg['content']]}")
                if first_msg['content'][0]['type'] == 'image_url':
                    url = first_msg['content'][0]['image_url']['url']
                    print(f"  image_url length: {len(url)} chars")
                    print(f"  image_url prefix: {url[:50]}...")
            
            outputs = llm.chat(
                messages=[inp["messages"] for inp in batch_inputs],
                sampling_params=sampling_params,
                chat_template_content_format="openai",
            )
        else:
            # InternVL, Qwen, SailVL etc. use generate method
            outputs = llm.generate(batch_inputs, sampling_params)
        
        for output, meta in zip(outputs, batch_metadata):
            generated_text = output.outputs[0].text.strip()
            
            if task_type == "vlm_tagging":
                # VLM captioning task
                predicted_caption = clean_caption(generated_text)
                result = {
                    'index': meta['index'],
                    'true_captions': meta['true_captions'],
                    'predicted_caption': predicted_caption,
                    'metadata': meta['metadata']
                }
            elif task_type == "image_classification":
                # Image classification task - clean output first
                cleaned_text = clean_classification_output(generated_text)

                # Choose matching strategy based on dataset
                if use_semantic_matching:
                    # ImageNet: Use embedding semantic matching
                    class_scores = calculate_semantic_match_scores(cleaned_text, class_names)
                else:
                    # CIFAR: Use string matching
                    class_scores = calculate_match_scores(cleaned_text, class_names)

                predicted_class = int(np.argmax(class_scores))
                confidence = float(class_scores[predicted_class])
                
                # Calculate Top5
                sorted_indices = np.argsort(class_scores)[-min(5, len(class_names)):][::-1]
                top5_indices = sorted_indices.tolist()
                top5_probs = class_scores[sorted_indices].tolist()
                
                result = {
                    'index': meta['index'],
                    'true_label': meta['true_label'],
                    'true_label_name': meta['true_label_name'],
                    'prediction': predicted_class,
                    'prediction_name': class_names[predicted_class],
                    'confidence': confidence,
                    'top5_predictions': top5_indices,
                    'top5_prediction_names': [class_names[i] for i in top5_indices],
                    'top5_confidences': top5_probs,
                    'raw_output': generated_text,
                    'cleaned_output': cleaned_text  # Cleaned output used for matching
                }
            
            results.append(result)
        
        # Update progress bar display
        batch_time = time.time() - batch_start
        samples_done = len(results)
        samples_per_sec = samples_done / (time.time() - start_time)
        pbar.set_postfix({
            'done': f'{samples_done}/{len(all_inputs)}',
            'speed': f'{samples_per_sec:.1f} img/s'
        })
    
    # Save results
    # Choose output directory based on task type
    if task_type == "image_classification":
        actual_output_dir = output_dir.replace("vlm_tagging", "image_classification")
    else:
        actual_output_dir = output_dir
    
    os.makedirs(actual_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{model_name}_{timestamp}"
    
    json_path = os.path.join(actual_output_dir, f"{filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also save CSV file (for quick viewing and statistics)
    if task_type == "image_classification":
        # CSV format for image classification task
        csv_data = []
        for r in results:
            csv_data.append({
                'index': r['index'],
                'true_label': r['true_label'],
                'predicted_label': r['prediction'],
                'confidence': r['confidence'],
                'correct': r['prediction'] == r['true_label']
            })
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(actual_output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to: {csv_path}")
    elif task_type == "vlm_tagging":
        # CSV format for VLM captioning task
        csv_data = []
        for r in results:
            csv_row = {
                'index': r['index'],
                'predicted_caption': r['predicted_caption'],
                'true_caption_1': r['true_captions'][0] if len(r['true_captions']) > 0 else "",
                'true_caption_2': r['true_captions'][1] if len(r['true_captions']) > 1 else "",
                'true_caption_3': r['true_captions'][2] if len(r['true_captions']) > 2 else "",
                'num_true_captions': len(r['true_captions']),
                'image_id': r['metadata'].get('image_id', '')
            }
            csv_data.append(csv_row)
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(actual_output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nCSV saved to: {csv_path}")
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {json_path}")
    print(f"Total samples: {len(results)}")
    
    # Display statistics based on task type
    if task_type == "image_classification":
        correct = sum(1 for r in results if r['prediction'] == r['true_label'])
        accuracy = correct / len(results) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    
    print(f"{'='*50}")
    
    # Show examples
    print(f"\nSample predictions (first 3):")
    for result in results[:3]:
        print(f"\n[Index {result['index']}]")
        if task_type == "vlm_tagging":
            print(f"  Predicted: {result['predicted_caption']}")
            print(f"  True: {result['true_captions'][0][:70]}...")
        elif task_type == "image_classification":
            print(f"  Predicted: {result['prediction_name']} (conf: {result['confidence']:.3f})")
            print(f"  True: {result['true_label_name']}")
            print(f"  Raw output: {result['raw_output']}")
    
    return results


def clean_caption(caption: str) -> str:
    """Clean the generated caption"""
    # Remove thinking markers
    if "<think>" in caption:
        if "</think>" in caption:
            caption = caption.split("</think>")[-1]
        else:
            caption = caption.split("<think>")[0]
    
    # Remove extra whitespace
    caption = ' '.join(caption.split())
    
    # Ensure ending with punctuation
    if caption and not caption.endswith(('.', '!', '?')):
        caption += '.'
    
    return caption.strip()


def clean_classification_output(text: str) -> str:
    """Clean classification output, remove markdown, scientific names, explanations, etc.

    Handles cases like:
    - "**fingerless glove** (specifically,"
    - "black-capped chickadee (*Poecile"
    - "The objects in the image are **craft supplies for"
    """
    import re
    
    # Remove markdown bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove scientific names (italic Latin names in parentheses)
    text = re.sub(r'\s*\([^)]*\)', '', text)
    
    # Remove "The ... is/are" sentence pattern, keep only keywords
    text = re.sub(r'^(The\s+)?(object|image|animal|item)s?\s+(in\s+the\s+image\s+)?(is|are)\s+', '', text, flags=re.IGNORECASE)
    
    # Remove subsequent explanations like "specifically," "or" etc.
    text = re.sub(r'\s*(specifically|or|also|namely|i\.e\.|e\.g\.).*$', '', text, flags=re.IGNORECASE)
    
    # Remove leading articles
    text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)
    
    # Keep only the first phrase (if there are commas or newlines)
    text = text.split(',')[0].split('\n')[0]
    
    return text.strip()
    
    return caption.strip()


def calculate_match_scores(output: str, class_names: list) -> np.ndarray:
    """Calculate match scores between generated text and classes

    Uses multi-level matching strategy:
    1. Exact match (highest priority)
    2. Substring containment
    3. Word-level intersection
    4. Partial word matching (for handling compound words)
    """
    output_lower = output.lower().strip()
    # Clean punctuation from output
    output_lower = output_lower.rstrip('.,!?;:')
    output_words = set(output_lower.split())
    
    num_classes = len(class_names)
    class_scores = np.zeros(num_classes)
    
    for i, class_name in enumerate(class_names):
        class_lower = class_name.lower().replace('_', ' ').replace('-', ' ').strip()
        class_words = set(class_lower.split())
        
        # Exact match
        if class_lower == output_lower:
            class_scores[i] = 100.0
            continue
            
        # Output fully contains class name (e.g., "golden retriever" in "a golden retriever")
        if class_lower in output_lower:
            class_scores[i] = 50.0 + len(class_lower) * 0.5
            continue
            
        # Class name contains output (e.g., "retriever" in "golden retriever")
        if output_lower in class_lower and len(output_lower) >= 3:
            class_scores[i] = 30.0 + len(output_lower) * 0.3
            continue
        
        # Word-level intersection matching
        common_words = class_words.intersection(output_words)
        if common_words:
            # Weight: longer word matches get higher scores
            score = sum(len(word) * 2.0 for word in common_words if len(word) > 2)
            class_scores[i] = max(class_scores[i], score)
        
        # Partial word matching (handling "greyhound" vs "Italian greyhound")
        for class_word in class_words:
            if len(class_word) >= 4:  # Only consider longer words
                for out_word in output_words:
                    if len(out_word) >= 4:
                        # Stem matching
                        if class_word.startswith(out_word) or out_word.startswith(class_word):
                            class_scores[i] = max(class_scores[i], 15.0)
                        elif class_word in out_word or out_word in class_word:
                            class_scores[i] = max(class_scores[i], 10.0)
    
    # Normalize
    if class_scores.sum() > 0:
        class_scores = class_scores / class_scores.sum()
    else:
        # When no match, use uniform distribution with very low confidence
        class_scores = np.ones(num_classes) * 0.001 / num_classes
    
    return class_scores


def main():
    parser = argparse.ArgumentParser(description="vLLM accelerated VLM evaluation")
    parser.add_argument('--task', type=str, default='vlm_tagging',
                       choices=['vlm_tagging', 'image_classification'],
                       help='Task type (default: vlm_tagging)')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['Flickr30k', 'COCO', 'CIFAR-10', 'CIFAR-100', 'ImageNet-1k'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       choices=['qwen3-vl-8b', 'internvl3.5-8b', 'sailvl-8b', 'ministral3-vl-8b', 'pixtral-12b', 'glm-4.6v-flash', 'gemma3-4b', 'step3-vl-10b'],
                       help='Model name')
    parser.add_argument('--output_dir', type=str, default='./outputs/vlm_tagging',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--tp', type=int, default=1,
                       help='Tensor parallel size (number of GPUs)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (for testing, default: all)')
    
    args = parser.parse_args()
    
    run_vllm_evaluation(
        task_type=args.task,
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tp,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()

