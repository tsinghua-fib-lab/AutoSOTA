"""
knn_image_vlm.py

Image KNN computation script, supporting multiple VLMs for image embedding extraction.

Supported embedding methods:
- metaclip: facebook/metaclip-2-worldwide-huge-quickgelu
- qwen3vl: Qwen/Qwen3-VL-8B-Instruct
- llama32vl: meta-llama/Llama-3.2-11B-Vision-Instruct
- internvl: OpenGVLab/InternVL3_5-8B
- sailvl: BytedanceDouyinContent/SAIL-VL2-8B

Supported data sources:
- HuggingFace datasets (flickr30k, coco)
- Local image directory

Run:
  # Using MetaCLIP (default)
  python knn_image_vlm.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method metaclip \
    --k 9

  # Using Qwen3-VL
  python knn_image_vlm.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method qwen3vl \
    --k 9

  # Using Llama-3.2-Vision
  python knn_image_vlm.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method llama32vl \
    --k 9

  # Using InternVL
  python knn_image_vlm.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method internvl \
    --k 9

  # Using SAIL-VL2
  python knn_image_vlm.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method sailvl \
    --k 9
"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ==================== Embedding Methods ====================

class MetaCLIPEmbedder:
    """MetaCLIP image embedder"""
    
    def __init__(self, model_name: str = "facebook/metaclip-2-worldwide-huge-quickgelu", device: str = "cuda"):
        from transformers import AutoProcessor, AutoModel
        
        self.device = device
        self.embed_dim = None
        print(f"Loading MetaCLIP: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("✓ MetaCLIP loaded")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from a list of PIL Images"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="MetaCLIP embedding"):
            batch_images = images[i:i+batch_size]

            valid_images = []
            valid_positions = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_positions.append(j)

            if valid_images:
                inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
                embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
                del outputs, inputs  # Clean up intermediate variables
                
                if self.embed_dim is None:
                    self.embed_dim = embeddings.shape[1]
                
                emb_idx = 0
                for j, img in enumerate(batch_images):
                    if img is not None:
                        all_embeddings.append(embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
            else:
                dim = self.embed_dim if self.embed_dim else 1280
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)


class Qwen3VLVisionEmbedder:
    """Qwen3-VL Vision Encoder"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading Qwen3-VL: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"  # PyTorch 2.0 built-in Scaled Dot Product Attention
        )
        self.model.eval()
        self.visual = self.model.visual
        
        # Try to accelerate vision encoder with torch.compile
        try:
            self.visual = torch.compile(self.visual, mode="reduce-overhead")
            print("   ✓ torch.compile enabled for visual encoder")
        except Exception as e:
            print(f"   torch.compile failed: {e}")

        self.image_processor = self.processor.image_processor

        # Limit image resolution to avoid generating too many vision tokens which causes slow processing
        # Default may be min=256*28*28=200704, max=1280*28*28=1003520 pixels
        # Here limited to ~384*384 = 147456 pixels, significantly reducing computation
        self.image_processor.min_pixels = 128 * 28 * 28  # ~100K pixels
        self.image_processor.max_pixels = 384 * 28 * 28  # ~300K pixels (~550x550)
        print(f"   Image processor: min_pixels={self.image_processor.min_pixels}, max_pixels={self.image_processor.max_pixels}")
        
        self.device = self.model.device
        self.embed_dim = None
        
        print(f"✓ Qwen3-VL loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from a list of PIL Images - true batch processing"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Qwen3-VL embedding"):
            batch_images = images[i:i+batch_size]

            # Preprocessing: collect valid images
            valid_images = []
            valid_indices = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_indices.append(j)
            
            # If no valid images, fill all with zero vectors
            if not valid_images:
                dim = self.embed_dim if self.embed_dim else 4096
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
                continue

            try:
                # Batch preprocessing
                inputs = self.image_processor(images=valid_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device, non_blocking=True)
                image_grid_thw = inputs.get('image_grid_thw')
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device, non_blocking=True)
                
                # Debug info (only print for first batch)
                if i == 0:
                    total_tokens = sum(g[0].item() * g[1].item() * g[2].item() for g in image_grid_thw) if image_grid_thw is not None else 0
                    print(f"   Batch 0: {len(valid_images)} images, total_tokens={total_tokens}")

                # Batch forward pass
                if image_grid_thw is not None:
                    vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    vision_outputs = self.visual(pixel_values)
                
                # Process output
                if isinstance(vision_outputs, tuple):
                    vision_hidden = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_hidden = vision_outputs.last_hidden_state
                else:
                    vision_hidden = vision_outputs
                
                # vision_hidden: (total_tokens, hidden_dim)
                # Need to split by per-image token count
                # Note: Qwen3-VL uses spatial merge, actual token count = raw / merge_ratio
                if image_grid_thw is not None:
                    token_counts_raw = [g[0].item() * g[1].item() * g[2].item() for g in image_grid_thw]
                    total_raw = sum(token_counts_raw)
                    
                    # Compute merge ratio and adjust token counts
                    if vision_hidden.shape[0] != total_raw:
                        merge_ratio = total_raw / vision_hidden.shape[0]
                        token_counts = [int(t / merge_ratio) for t in token_counts_raw]
                        # Adjust the last one to ensure the total is correct
                        token_counts[-1] = vision_hidden.shape[0] - sum(token_counts[:-1])
                    else:
                        token_counts = token_counts_raw
                    
                    hidden_splits = torch.split(vision_hidden, token_counts, dim=0)
                    
                    # Average and normalize per image
                    batch_embeddings = []
                    for hidden in hidden_splits:
                        emb = hidden.mean(dim=0)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        batch_embeddings.append(emb.detach().cpu().float().numpy())
                else:
                    # fallback: average over all
                    emb = vision_hidden.mean(dim=0)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    batch_embeddings = [emb.detach().cpu().float().numpy()] * len(valid_images)

                if self.embed_dim is None:
                    self.embed_dim = batch_embeddings[0].shape[0]

                # Fill results in original order (including zero vectors for None images)
                emb_idx = 0
                for j in range(len(batch_images)):
                    if j in valid_indices:
                        all_embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
                
            except Exception as e:
                print(f"WARNING: Batch {i//batch_size} failed: {e}, falling back to per-image processing")
                # Fall back to per-image processing
                for j, img in enumerate(batch_images):
                    if img is None:
                        dim = self.embed_dim if self.embed_dim else 4096
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    else:
                        try:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            inputs = self.image_processor(images=img, return_tensors="pt")
                            pv = inputs['pixel_values'].to(self.device)
                            gt = inputs.get('image_grid_thw')
                            if gt is not None:
                                gt = gt.to(self.device)
                            out = self.visual(pv, grid_thw=gt) if gt is not None else self.visual(pv)
                            if isinstance(out, tuple):
                                h = out[0]
                            elif hasattr(out, 'last_hidden_state'):
                                h = out.last_hidden_state
                            else:
                                h = out
                            emb = h.mean(dim=0) if len(h.shape) == 2 else h.mean(dim=(0, 1))
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                            all_embeddings.append(emb.detach().cpu().float().numpy())
                            if self.embed_dim is None:
                                self.embed_dim = all_embeddings[-1].shape[0]
                        except Exception as e2:
                            dim = self.embed_dim if self.embed_dim else 4096
                            all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)


class InternVLVisionEmbedder:
    """InternVL3.5-8B Vision Encoder"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3_5-8B", device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        print(f"Loading InternVL: {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # InternVL image preprocessing parameters
        self.input_size = 448
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        self.device = next(self.model.parameters()).device
        self.embed_dim = None
        
        # Try to accelerate vision encoder with torch.compile
        if hasattr(self.model, 'vision_model'):
            try:
                self.model.vision_model = torch.compile(self.model.vision_model, mode="reduce-overhead")
                print("   ✓ torch.compile enabled for vision encoder")
            except Exception as e:
                print(f"   torch.compile failed: {e}")
        
        print(f"✓ InternVL loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from a list of PIL Images - batch processing"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="InternVL embedding"):
            batch_images = images[i:i+batch_size]

            # Preprocessing: collect valid images
            valid_images = []
            valid_indices = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_indices.append(j)
            
            # If no valid images, fill all with zero vectors
            if not valid_images:
                dim = self.embed_dim if self.embed_dim else 3584  # InternVL3.5-8B hidden dimension
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
                continue
            
            try:
                # Batch preprocess images
                pixel_values_list = [self.transform(img) for img in valid_images]
                pixel_values = torch.stack(pixel_values_list, dim=0).to(
                    dtype=torch.bfloat16, device=self.device
                )
                
                # Debug info (only print for first batch)
                if i == 0:
                    print(f"   Batch 0: {len(valid_images)} images, pixel_values.shape={pixel_values.shape}")

                # Extract vision features
                if hasattr(self.model, 'vision_model'):
                    # Directly use vision_model
                    vision_outputs = self.model.vision_model(pixel_values)
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_hidden = vision_outputs.last_hidden_state
                    elif isinstance(vision_outputs, tuple):
                        vision_hidden = vision_outputs[0]
                    else:
                        vision_hidden = vision_outputs
                elif hasattr(self.model, 'extract_feature'):
                    # Use InternVL's extract_feature method
                    vision_hidden = self.model.extract_feature(pixel_values)
                else:
                    raise ValueError("Cannot find vision encoder in InternVL model")

                # vision_hidden: (batch, num_patches, hidden_dim) or (batch, hidden_dim)
                if len(vision_hidden.shape) == 3:
                    # Average over all patches
                    batch_emb = vision_hidden.mean(dim=1)
                elif len(vision_hidden.shape) == 2:
                    batch_emb = vision_hidden
                else:
                    # Other cases: flatten then average
                    batch_emb = vision_hidden.reshape(vision_hidden.shape[0], -1, vision_hidden.shape[-1]).mean(dim=1)
                
                # Normalize
                batch_emb = batch_emb.float()
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                batch_embeddings = batch_emb.detach().cpu().numpy()
                
                if self.embed_dim is None:
                    self.embed_dim = batch_embeddings.shape[1]
                
                # Fill results in original order (including zero vectors for None images)
                emb_idx = 0
                for j in range(len(batch_images)):
                    if j in valid_indices:
                        all_embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))

                # Clean up intermediate variables
                del pixel_values, vision_hidden, batch_emb, batch_embeddings
                
            except Exception as e:
                print(f"WARNING: Batch {i//batch_size} failed: {e}, falling back to per-image processing")
                # Fall back to per-image processing
                for j, img in enumerate(batch_images):
                    if img is None:
                        dim = self.embed_dim if self.embed_dim else 3584
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    else:
                        try:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            pixel_values = self.transform(img).unsqueeze(0).to(
                                dtype=torch.bfloat16, device=self.device
                            )
                            
                            if hasattr(self.model, 'vision_model'):
                                out = self.model.vision_model(pixel_values)
                                if hasattr(out, 'last_hidden_state'):
                                    h = out.last_hidden_state
                                elif isinstance(out, tuple):
                                    h = out[0]
                                else:
                                    h = out
                            elif hasattr(self.model, 'extract_feature'):
                                h = self.model.extract_feature(pixel_values)
                            else:
                                raise ValueError("Cannot find vision encoder")
                            
                            if len(h.shape) == 3:
                                emb = h.mean(dim=1).squeeze(0)
                            elif len(h.shape) == 2:
                                emb = h.squeeze(0)
                            else:
                                emb = h.reshape(-1, h.shape[-1]).mean(dim=0)
                            
                            emb = emb.float()
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                            all_embeddings.append(emb.detach().cpu().numpy())
                            
                            if self.embed_dim is None:
                                self.embed_dim = all_embeddings[-1].shape[0]
                        except Exception as e2:
                            print(f"WARNING: Image {i+j} failed: {e2}")
                            dim = self.embed_dim if self.embed_dim else 3584
                            all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Clean GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)


class SAILVLVisionEmbedder:
    """SAIL-VL2-8B Vision Encoder

    BytedanceDouyinContent/SAIL-VL2-8B
    Uses SAILViT-Huge as the vision encoder
    """
    
    def __init__(self, model_name: str = "BytedanceDouyinContent/SAIL-VL2-8B", device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        
        print(f"Loading SAIL-VL2: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        self.device = next(self.model.parameters()).device
        self.embed_dim = None
        
        # Get vision encoder
        if hasattr(self.model, 'vision_model'):
            self.vision_encoder = self.model.vision_model
        elif hasattr(self.model, 'visual'):
            self.vision_encoder = self.model.visual
        else:
            self.vision_encoder = None
            print("WARNING: Could not find vision encoder, will use full model")
        
        print(f"✓ SAIL-VL2 loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 8) -> np.ndarray:
        """Extract embeddings from a list of PIL Images"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="SAIL-VL2 embedding"):
            batch_images = images[i:i+batch_size]

            # Preprocessing: collect valid images
            valid_images = []
            valid_indices = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_indices.append(j)
            
            # If no valid images, fill all with zero vectors
            if not valid_images:
                dim = self.embed_dim if self.embed_dim else 3584  # SAIL-VL2 hidden dim
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
                continue
            
            try:
                batch_embeddings = []
                
                # SAIL-VL2 requires per-image processing (processor needs text argument)
                for img in valid_images:
                    # Build simple message for processor handling
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "describe"}
                        ]
                    }]
                    
                    text = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    
                    inputs = self.processor(
                        images=img,
                        text=text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device).to(torch.bfloat16)
                    
                    # Extract vision features
                    if self.vision_encoder is not None:
                        # Directly use vision encoder
                        pixel_values = inputs.get('pixel_values')
                        if pixel_values is not None:
                            vision_outputs = self.vision_encoder(pixel_values)
                            
                            if hasattr(vision_outputs, 'last_hidden_state'):
                                vision_hidden = vision_outputs.last_hidden_state
                            elif isinstance(vision_outputs, tuple):
                                vision_hidden = vision_outputs[0]
                            else:
                                vision_hidden = vision_outputs
                        else:
                            raise ValueError("No pixel_values found in inputs")
                    else:
                        # Use full model forward (only extract vision part)
                        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        # Try to extract vision hidden states from output
                        if hasattr(outputs, 'vision_outputs'):
                            vision_hidden = outputs.vision_outputs.last_hidden_state
                        else:
                            # Fallback: use first layer hidden states
                            vision_hidden = outputs.hidden_states[0]
                    
                    # vision_hidden: (batch, num_patches, hidden_dim) or other formats
                    if len(vision_hidden.shape) == 3:
                        emb = vision_hidden.mean(dim=1).squeeze(0)
                    elif len(vision_hidden.shape) == 2:
                        emb = vision_hidden.mean(dim=0)
                    else:
                        emb = vision_hidden.reshape(-1, vision_hidden.shape[-1]).mean(dim=0)
                    
                    # Normalize
                    emb = emb.float()
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    batch_embeddings.append(emb.detach().cpu().numpy())
                    
                    if self.embed_dim is None:
                        self.embed_dim = batch_embeddings[-1].shape[0]
                    
                    # Clean up
                    del inputs, vision_hidden, emb
                
                # Fill results in original order
                emb_idx = 0
                for j in range(len(batch_images)):
                    if j in valid_indices:
                        all_embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))

            except Exception as e:
                print(f"WARNING: Batch {i//batch_size} failed: {e}")
                import traceback
                traceback.print_exc()
                # Fill with zero vectors
                dim = self.embed_dim if self.embed_dim else 3584
                for j in range(len(batch_images)):
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Clean GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)


class Llama32VisionEmbedder:
    """Llama-3.2-Vision Encoder"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", device: str = "cuda"):
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        print(f"Loading Llama-3.2-Vision: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.embed_dim = None
        self.model_name = model_name
        
        print(f"✓ Llama-3.2-Vision loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 4) -> np.ndarray:
        """Extract embeddings from a list of PIL Images"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Llama-3.2-Vision embedding"):
            batch_images = images[i:i+batch_size]

            # Llama processor requires text argument, must process per-image
            for j, img in enumerate(batch_images):
                if img is None:
                    dim = self.embed_dim if self.embed_dim else 4096
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    continue
                
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    inputs = self.processor(
                        images=img,
                        text="<|image|>",
                        return_tensors="pt"
                    )
                    
                    pixel_values = inputs.get('pixel_values')
                    aspect_ratio_ids = inputs.get('aspect_ratio_ids')
                    aspect_ratio_mask = inputs.get('aspect_ratio_mask')
                    
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(self.device)
                        if aspect_ratio_ids is not None:
                            aspect_ratio_ids = aspect_ratio_ids.to(self.device)
                        if aspect_ratio_mask is not None:
                            aspect_ratio_mask = aspect_ratio_mask.to(self.device)
                        
                        vision_outputs = self.model.vision_model(
                            pixel_values=pixel_values,
                            aspect_ratio_ids=aspect_ratio_ids,
                            aspect_ratio_mask=aspect_ratio_mask,
                        )
                        
                        # Get hidden states
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            vision_hidden = vision_outputs.last_hidden_state
                        elif isinstance(vision_outputs, tuple):
                            vision_hidden = vision_outputs[0]
                        else:
                            vision_hidden = vision_outputs
                        
                        # Average all tokens to get image representation
                        # Llama-3.2-Vision output format: (batch, num_images, num_tiles, num_patches, hidden_dim)
                        # e.g. (1, 1, 4, 1601, 7680)

                        # Debug: print info for first few images
                        if i + j < 3:
                            print(f"  [DEBUG] img {i+j}: vision_hidden.shape={vision_hidden.shape}, "
                                  f"mean={vision_hidden.mean().item():.4f}, std={vision_hidden.std().item():.4f}")
                        
                        if len(vision_hidden.shape) == 5:
                            # (batch, num_images, num_tiles, num_patches, hidden_dim)
                            # Average over tiles and patches: dim 2,3
                            embedding = vision_hidden.mean(dim=(2, 3)).squeeze(0).squeeze(0)
                        elif len(vision_hidden.shape) == 4:
                            # (batch, num_aspects, num_patches, hidden_dim) -> (batch, hidden_dim)
                            embedding = vision_hidden.mean(dim=(1, 2))
                        elif len(vision_hidden.shape) == 3:
                            # (batch, num_patches, hidden_dim) -> (batch, hidden_dim)
                            embedding = vision_hidden.mean(dim=1)
                        elif len(vision_hidden.shape) == 2:
                            # (num_patches, hidden_dim) -> (hidden_dim,)
                            embedding = vision_hidden.mean(dim=0)
                        else:
                            # fallback
                            embedding = vision_hidden.reshape(-1, vision_hidden.shape[-1]).mean(dim=0)
                        
                        # Ensure 1D vector
                        embedding = embedding.squeeze()
                        
                        # Debug: print embedding info (before normalization)
                        if i + j < 3:
                            print(f"  [DEBUG] img {i+j}: embedding.shape={embedding.shape}, "
                                  f"norm={embedding.norm().item():.4f}, "
                                  f"first5={embedding[:5].tolist()}")
                        
                        # Convert to float32 before normalization (avoid bfloat16 precision issues)
                        embedding = embedding.float()
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        
                        # Debug: print first few values and norm after normalization
                        if i + j < 3:
                            actual_norm = embedding.norm().item()
                            print(f"  [DEBUG] img {i+j}: normalized first5={embedding[:5].tolist()}, "
                                  f"actual_norm={actual_norm:.6f}")
                        
                        # Compute cosine similarity after processing the 3rd image
                        if i + j == 2 and len(all_embeddings) >= 2:
                            e0, e1 = all_embeddings[0], all_embeddings[1]
                            e2 = embedding.detach().cpu().numpy()
                            sim_01 = float(np.dot(e0, e1))
                            sim_02 = float(np.dot(e0, e2))
                            sim_12 = float(np.dot(e1, e2))
                            print(f"  [DEBUG] Cosine similarity: sim(0,1)={sim_01:.6f}, sim(0,2)={sim_02:.6f}, sim(1,2)={sim_12:.6f}")
                        
                        embedding_np = embedding.detach().cpu().numpy()
                        
                        if self.embed_dim is None:
                            self.embed_dim = embedding_np.shape[0]
                        
                        all_embeddings.append(embedding_np)
                        
                        # Clean up intermediate variables
                        del vision_outputs, vision_hidden, pixel_values, embedding, embedding_np
                        if aspect_ratio_ids is not None:
                            del aspect_ratio_ids
                        if aspect_ratio_mask is not None:
                            del aspect_ratio_mask
                    else:
                        dim = self.embed_dim if self.embed_dim else 4096
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    
                    del inputs
                    
                except Exception as e:
                    print(f"WARNING: Failed to process image {i+j}: {e}")
                    dim = self.embed_dim if self.embed_dim else 4096
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Clean GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        
        return np.stack(all_embeddings, axis=0)


# ==================== Utility Functions ====================

def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list JSON.")
    return data


def load_hf_dataset(dataset_name: str, split: str = "test", num_samples: Optional[int] = None):
    """Load dataset from HuggingFace"""
    from datasets import load_dataset
    
    if dataset_name.lower() == "flickr30k":
        print(f"Loading Flickr30k from HuggingFace (split={split})...")
        dataset = load_dataset("nlphuji/flickr30k", split=split, trust_remote_code=True)
    elif dataset_name.lower() == "coco":
        print(f"Loading COCO-Caption from HuggingFace (split={split})...")
        dataset = load_dataset("lmms-lab/COCO-Caption", split=split, trust_remote_code=True)
    elif dataset_name.lower() == "cifar-10" or dataset_name.lower() == "cifar10":
        print(f"Loading CIFAR-10 from HuggingFace (split={split})...")
        dataset = load_dataset("cifar10", split=split, trust_remote_code=True)
    elif dataset_name.lower() == "cifar-100" or dataset_name.lower() == "cifar100":
        print(f"Loading CIFAR-100 from HuggingFace (split={split})...")
        dataset = load_dataset("cifar100", split=split, trust_remote_code=True)
    elif dataset_name.lower() == "imagenet" or dataset_name.lower() == "imagenet-1k":
        print(f"Loading ImageNet-1K from HuggingFace (split={split})...")
        # ImageNet-1k only has train and validation, no test
        actual_split = "validation" if split == "test" else split
        # Prefer local cache, set HF_DATASETS_OFFLINE=1 environment variable
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        dataset = load_dataset("ILSVRC/imagenet-1k", split=actual_split, trust_remote_code=True)
    elif dataset_name.lower() == "food-101" or dataset_name.lower() == "food101":
        print(f"Loading Food-101 from HuggingFace (split={split})...")
        actual_split = "validation" if split == "test" else split
        dataset = load_dataset("food101", split=actual_split, trust_remote_code=True)
    elif dataset_name.lower() == "stanford_cars" or dataset_name.lower() == "stanford-cars":
        print(f"Loading Stanford Cars from HuggingFace (split={split})...")
        dataset = load_dataset("tanganke/stanford_cars", split=split, trust_remote_code=True)
    else:
        # Try loading directly with the dataset name
        print(f"Loading {dataset_name} from HuggingFace (split={split})...")
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Unknown/unsupported dataset: {dataset_name}. Error: {e}")
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def get_image_from_hf(hf_dataset, idx: int) -> Optional[Image.Image]:
    """Get image from HuggingFace dataset

    Supports different dataset field names:
    - flickr30k/coco: 'image'
    - cifar10/cifar100: 'img'
    - imagenet: 'image'
    - food101: 'image'
    - stanford_cars: 'image'
    """
    try:
        item = hf_dataset[idx]
        # Try different field names
        for key in ['image', 'img', 'pixel_values']:
            image = item.get(key)
            if image is not None:
                return image
    except Exception as e:
        print(f"WARNING: Failed to get image at index {idx}: {e}")
    return None


def get_image_from_path(item: Dict, image_dir: str) -> Optional[str]:
    """Get image path from item"""
    for key in ["image_path", "image", "file_path", "path", "filename"]:
        if key in item:
            img_path = item[key]
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_dir, os.path.basename(img_path))
            if os.path.exists(img_path):
                return img_path
    
    if "metadata" in item:
        meta = item["metadata"]
        for key in ["image_path", "image", "file_path", "path", "filename", "image_id"]:
            if key in meta:
                img_id = str(meta[key])
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    potential = os.path.join(image_dir, f"{img_id}{ext}")
                    if os.path.exists(potential):
                        return potential
    
    if "index" in item:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            potential = os.path.join(image_dir, f"{item['index']}{ext}")
            if os.path.exists(potential):
                return potential
    
    return None


def get_embedder(method: str, device: str = "cuda"):
    """Get the corresponding embedder by method name"""
    method = method.lower()
    
    if method == "metaclip":
        return MetaCLIPEmbedder(device=device)
    elif method == "qwen3vl":
        return Qwen3VLVisionEmbedder(device=device)
    elif method == "llama32vl":
        return Llama32VisionEmbedder(device=device)
    elif method == "internvl":
        return InternVLVisionEmbedder(device=device)
    elif method == "sailvl":
        return SAILVLVisionEmbedder(device=device)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: metaclip, qwen3vl, llama32vl, internvl, sailvl")


def get_default_batch_size(method: str) -> int:
    """Get default batch size"""
    method = method.lower()
    if method == "metaclip":
        return 32
    elif method == "qwen3vl":
        return 16
    elif method == "llama32vl":
        return 2  # Llama-3.2-Vision is large, use very small batch to avoid OOM
    elif method == "internvl":
        return 16  # InternVL 8B model, similar to Qwen3-VL
    elif method == "sailvl":
        return 8  # SAIL-VL2 8B model
    return 16


# ==================== KNN Functions ====================

def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors within the dataset itself (excluding self)"""
    N, D = emb.shape
    nn_idx = np.empty((N, k), dtype=np.int32)
    nn_score = np.empty((N, k), dtype=np.float32)

    for start in range(0, N, query_block):
        end = min(start + query_block, N)
        q = emb[start:end]
        scores = q @ emb.T

        rows = np.arange(end - start)
        cols = np.arange(start, end)
        scores[rows, cols] = -np.inf

        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        top_sc = np.take_along_axis(scores, top_idx, axis=1)

        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)

        nn_idx[start:end] = top_idx.astype(np.int32)
        nn_score[start:end] = top_sc.astype(np.float32)

        print(f"Processed queries [{start}:{end}) / {N}")

    return nn_idx, nn_score


def topk_neighbors_query_vs_all(
    query_emb: np.ndarray,
    all_emb: np.ndarray,
    query_indices: List[int],
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors for query samples in the full image library"""
    Q = query_emb.shape[0]
    N = all_emb.shape[0]
    
    nn_idx = np.empty((Q, k), dtype=np.int32)
    nn_score = np.empty((Q, k), dtype=np.float32)
    
    for start in range(0, Q, query_block):
        end = min(start + query_block, Q)
        q = query_emb[start:end]
        scores = q @ all_emb.T
        
        for i, qi in enumerate(range(start, end)):
            original_idx = query_indices[qi]
            if 0 <= original_idx < N:
                scores[i, original_idx] = -np.inf
        
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        top_sc = np.take_along_axis(scores, top_idx, axis=1)
        
        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)
        
        nn_idx[start:end] = top_idx.astype(np.int32)
        nn_score[start:end] = top_sc.astype(np.float32)
        
        print(f"Processed queries [{start}:{end}) / {Q} (searching in {N} images)")
    
    return nn_idx, nn_score


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Find KNN for images using various VLM encoders")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input VLM results JSON file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (flickr30k/coco)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--full_dataset", action="store_true",
                        help="Find neighbors in the entire dataset (not just within input_json samples)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Local image directory (if not using HuggingFace dataset)")
    parser.add_argument("--out_jsonl", type=str, required=True,
                        help="Output neighbor JSONL file")
    parser.add_argument("--method", type=str, default="metaclip",
                        choices=["metaclip", "qwen3vl", "llama32vl", "internvl", "sailvl"],
                        help="Embedding method (default: metaclip)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Embedding batch size (auto-selected by method if not specified)")
    parser.add_argument("--query_block", type=int, default=256,
                        help="KNN query block size")
    parser.add_argument("--k", type=int, default=9,
                        help="Number of neighbors")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Set default batch size
    if args.batch_size is None:
        args.batch_size = get_default_batch_size(args.method)

    print("=" * 80)
    print("Image KNN Neighbor Search (VLM version)")
    print("=" * 80)
    print(f"Input file: {args.input_json}")
    print(f"Embedding method: {args.method}")
    if args.dataset:
        print(f"Dataset: {args.dataset} (split={args.split})")
        if args.full_dataset:
            print(f"Search scope: Entire dataset")
        else:
            print(f"Search scope: Only samples in input_json")
    elif args.image_dir:
        print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.out_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of neighbors: {args.k}")
    print("=" * 80)

    # 1) Load VLM results JSON
    print("\nLoading VLM results...")
    items = load_json(args.input_json)
    N = len(items)
    print(f"Loaded {N} samples")
    
    query_indices = [item.get("index", i) for i, item in enumerate(items)]

    # 2) Initialize embedder
    print(f"\nInitializing {args.method} embedder...")
    embedder = get_embedder(args.method, args.device)

    # 3) Load images and extract embeddings
    if args.dataset:
        if args.full_dataset:
            print(f"\nLoading full {args.dataset} dataset from HuggingFace...")
            hf_dataset = load_hf_dataset(args.dataset, split=args.split, num_samples=None)
            total_images = len(hf_dataset)
            print(f"Dataset has {total_images} images in total")

            print("\nLoading all images...")
            all_images: List[Optional[Image.Image]] = []
            for i in tqdm(range(total_images), desc="Loading all images"):
                image = get_image_from_hf(hf_dataset, i)
                all_images.append(image)
            
            print("\nExtracting all image embeddings...")
            all_emb = embedder.embed_images_pil(all_images, batch_size=args.batch_size)
            print(f"Embedding dimensions: {all_emb.shape}")
            
            query_emb = all_emb[query_indices]
            
            meta = []
            for i, item in enumerate(items):
                idx = item.get("index", i)
                meta.append({
                    "global_id": i,
                    "source_file": os.path.basename(args.input_json),
                    "row_in_file": i,
                    "index": idx,
                })
            
            print("\nSearching for nearest neighbors in the full dataset...")
            nn_idx, nn_score = topk_neighbors_query_vs_all(
                query_emb, all_emb, query_indices,
                k=args.k, query_block=args.query_block
            )
        else:
            print(f"\nLoading {args.dataset} dataset from HuggingFace...")
            hf_dataset = load_hf_dataset(args.dataset, split=args.split, num_samples=None)
            
            # Build meta info
            meta = []
            for i, item in enumerate(items):
                idx = item.get("index", i)
                meta.append({
                    "global_id": i,
                    "source_file": os.path.basename(args.input_json),
                    "row_in_file": i,
                    "index": idx,
                })
            
            # Load images
            images: List[Optional[Image.Image]] = []
            print("\nExtracting images...")
            for i, item in enumerate(tqdm(items, desc="Loading images")):
                idx = item.get("index", i)
                image = get_image_from_hf(hf_dataset, idx)
                images.append(image)
            
            valid_count = sum(1 for img in images if img is not None)
            print(f"\nValid images: {valid_count} / {N}")

            print("\nExtracting image embeddings...")
            full_emb = embedder.embed_images_pil(images, batch_size=args.batch_size)

            # Clean up image memory
            for img in images:
                if img is not None:
                    img.close()
            del images
            import gc
            gc.collect()
            
            print(f"Embedding dimensions: {full_emb.shape}")

            print("\nSearching for nearest neighbors...")
            nn_idx, nn_score = topk_neighbors_all(full_emb, k=args.k, query_block=args.query_block)

    elif args.image_dir:
        images: List[Optional[Image.Image]] = []
        meta = []
        print("\nLoading images from local directory...")
        for i, item in enumerate(tqdm(items, desc="Loading images")):
            img_path = get_image_from_path(item, args.image_dir)
            if img_path:
                try:
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"WARNING: Failed to load {img_path}: {e}")
                    images.append(None)
            else:
                images.append(None)
            
            meta.append({
                "global_id": i,
                "source_file": os.path.basename(args.input_json),
                "row_in_file": i,
                "index": item.get("index", i),
            })
        
        valid_count = sum(1 for img in images if img is not None)
        print(f"\nValid images: {valid_count} / {N}")

        print("\nExtracting image embeddings...")
        full_emb = embedder.embed_images_pil(images, batch_size=args.batch_size)
        print(f"Embedding dimensions: {full_emb.shape}")

        print("\nSearching for nearest neighbors...")
        nn_idx, nn_score = topk_neighbors_all(full_emb, k=args.k, query_block=args.query_block)
    else:
        raise ValueError("Must specify --dataset or --image_dir")

    # 4) Save results
    print("\nSaving results...")
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in range(N):
            neighbors = [
                {"global_id": int(nn_idx[i, j]), "cosine": float(nn_score[i, j])}
                for j in range(args.k)
            ]
            row = {
                **meta[i],
                "neighbors": neighbors,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Neighbor results saved to: {args.out_jsonl}")
    print(f"   Total samples: {N}")
    print(f"   Neighbors per sample: {args.k}")
    print(f"   Embedding method: {args.method}")

    # Statistics
    print("\n" + "=" * 80)
    print("📊 Statistics")
    print("=" * 80)
    print(f"Avg similarity:  {nn_score.mean():.4f}")
    print(f"Min similarity:  {nn_score.min():.4f}")
    print(f"Max similarity:  {nn_score.max():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

