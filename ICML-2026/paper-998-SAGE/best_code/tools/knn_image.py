"""
Compute image embeddings using various methods, then find top-9 nearest neighbors for each image.

Supported embedding methods:
- metaclip: facebook/metaclip-2-worldwide-huge-quickgelu (default, public encoder)
- sailvl: BytedanceDouyinContent/SAILViT-Huge-600M-448px (SAIL-VL vision encoder)
- internvl: OpenGVLab/InternVL3_5-8B (InternVL vision encoder)
- qwen: Qwen/Qwen3-VL-8B-Instruct (Qwen3-VL vision encoder)
- step3vl: stepfun-ai/Step3-VL-10B (Step3-VL vision encoder, based on Qwen2-VL architecture)

Install:
  pip install -U transformers torch pillow numpy datasets

Run (using MetaCLIP):
  python knn_image.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method metaclip \
    --k 9

Run (using Step3-VL):
  python knn_image.py \
    --input_json "/path/to/vlm_results.json" \
    --dataset flickr30k \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method step3vl \
    --k 9

Run (loading images from local directory):
  python knn_image.py \
    --input_json "/path/to/vlm_results.json" \
    --image_dir "/path/to/images" \
    --out_jsonl "/path/to/out/image_neighbors.jsonl" \
    --method metaclip \
    --k 9
"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON file"""
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
        if num_samples is None:
            num_samples = 10000
        print(f"   (Using streaming mode to load first {num_samples} samples from {actual_split})")
        # Use streaming mode to avoid downloading the entire dataset
        ds_stream = load_dataset("ILSVRC/imagenet-1k", split=actual_split, streaming=True, trust_remote_code=True)
        # Manually iterate and show progress
        samples = []
        for idx, item in enumerate(ds_stream):
            if idx >= num_samples:
                break
            samples.append(item)
            if (idx + 1) % 1000 == 0:
                print(f"   Loaded {idx + 1}/{num_samples} samples...")
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        print(f"   ✓ Loaded {len(dataset)} ImageNet samples")
        num_samples = None  # Already limited
    elif dataset_name.lower() == "food-101" or dataset_name.lower() == "food101":
        print(f"Loading Food-101 from HuggingFace (split={split})...")
        # Food-101 only has train and validation split
        actual_split = "validation" if split == "test" else split
        dataset = load_dataset("food101", split=actual_split, trust_remote_code=True)
    elif dataset_name.lower() == "stanford_cars" or dataset_name.lower() == "stanford-cars":
        print(f"Loading Stanford Cars from HuggingFace (split={split})...")
        dataset = load_dataset("tanganke/stanford_cars", split=split, trust_remote_code=True)
    else:
        # Try loading directly with dataset name
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

    Supports different field names for different datasets:
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
        print(f"⚠️  Failed to get image at index {idx}: {e}")
    return None


def get_image_from_path(item: Dict, image_dir: str) -> Optional[str]:
    """Get image path from item"""
    # Try multiple possible field names
    for key in ["image_path", "image", "file_path", "path", "filename"]:
        if key in item:
            img_path = item[key]
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_dir, os.path.basename(img_path))
            if os.path.exists(img_path):
                return img_path
    
    # Try getting from metadata
    if "metadata" in item:
        meta = item["metadata"]
        for key in ["image_path", "image", "file_path", "path", "filename", "image_id"]:
            if key in meta:
                img_id = str(meta[key])
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    potential = os.path.join(image_dir, f"{img_id}{ext}")
                    if os.path.exists(potential):
                        return potential
    
    # If index exists, try to construct path
    if "index" in item:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            potential = os.path.join(image_dir, f"{item['index']}{ext}")
            if os.path.exists(potential):
                return potential
    
    return None


class MetaCLIPEmbedder:
    """MetaCLIP image embedder"""
    
    def __init__(self, model_name: str = "facebook/metaclip-2-worldwide-huge-quickgelu", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel
        
        self.device = device
        self.embed_dim = None  # Will be determined on first forward pass
        print(f"Loading MetaCLIP model: {model_name}")
        # Only load image_processor to avoid tokenizer compatibility issues
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        print("✓ MetaCLIP loaded")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from a list of PIL Images
        images: can contain None (will be replaced with zero vectors)
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch_images = images[i:i+batch_size]
            
            # Filter valid images and record positions
            valid_images = []
            valid_positions = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    # Ensure RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_positions.append(j)
            
            if valid_images:
                # Process images
                inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)
                
                # Get image embeddings
                outputs = self.model.get_image_features(**inputs)
                
                # Normalize
                embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy().astype(np.float32)

                # Record embedding dimension
                if self.embed_dim is None:
                    self.embed_dim = embeddings.shape[1]
                
                # Fill back results
                emb_idx = 0
                for j, img in enumerate(batch_images):
                    if img is not None:
                        all_embeddings.append(embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
            else:
                # All None, use zero vectors
                dim = self.embed_dim if self.embed_dim else 1280  # MetaCLIP-H default dimension
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)
    
    @torch.no_grad()
    def embed_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract image embeddings from file paths (backward compatible)"""
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                images.append(None)
        return self.embed_images_pil(images, batch_size)


class SAILVLVisionEmbedder:
    """SAILViT-Huge Vision Encoder
    
    Uses the standalone SAILViT-Huge model directly to extract image features,
    rather than extracting from SAIL-VL2-8B (to avoid adapter contamination).

    Reference: https://huggingface.co/BytedanceDouyinContent/SAILViT-Huge-600M-448px

    Pooling strategies:
    - "mean_patches": average over patches dimension -> (batch, 1536)
    - "mean_hidden": average over hidden dimension -> (batch, 1024) preserving spatial structure
    - "concat": mean_patches + std_patches -> (batch, 3072) increasing discriminability
    """
    
    def __init__(self, model_name: str = "BytedanceDouyinContent/SAILViT-Huge-600M-448px", 
                 device: str = "cuda", pooling: str = "mean_patches"):
        from transformers import AutoModel, AutoImageProcessor
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        self.device = device
        self.embed_dim = None
        self.pooling = pooling
        
        # Directly load standalone SAILViT-Huge model (not SAIL-VL2-8B)
        print(f"Loading SAILViT-Huge: {model_name}...")
        print(f"   Pooling strategy: {pooling}")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        
        # Try loading image processor
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"   Using image_processor: {type(self.image_processor).__name__}")
            self.transform = None
        except Exception as e:
            print(f"   No image_processor found ({e}), using standard transforms")
            self.image_processor = None
            
            # Use standard ImageNet preprocessing (SAILViT uses 448x448)
            img_size = 448
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        self.vision_device = device
        print(f"✓ SAILViT-Huge loaded on {device}")
        
        # Print model info
        if hasattr(self.model.config, 'hidden_size'):
            print(f"   Hidden size: {self.model.config.hidden_size}")
        if hasattr(self.model.config, 'image_size'):
            print(f"   Image size: {self.model.config.image_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from a list of PIL Images - batch processing"""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="SAILViT embedding"):
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
                dim = self.embed_dim if self.embed_dim else 1536
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
                continue
            
            try:
                # Preprocess images
                if self.image_processor is not None:
                    inputs = self.image_processor(images=valid_images, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].to(
                        device=self.vision_device, dtype=torch.bfloat16
                    )
                else:
                    pixel_values_list = [self.transform(img) for img in valid_images]
                    pixel_values = torch.stack(pixel_values_list, dim=0).to(
                        device=self.vision_device, dtype=torch.bfloat16
                    )
                
                # Debug info (only print first batch)
                if i == 0:
                    print(f"   Batch 0: {len(valid_images)} images, pixel_values.shape={pixel_values.shape}")
                
                # Use SAILViT model directly (it is the vision encoder)
                vision_outputs = self.model(pixel_values)
                
                # Debug: print output info
                if i == 0:
                    print(f"   vision_outputs type: {type(vision_outputs)}")
                    if hasattr(vision_outputs, 'pooler_output'):
                        print(f"   has pooler_output: {vision_outputs.pooler_output is not None}")
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        print(f"   has last_hidden_state: {vision_outputs.last_hidden_state is not None}")
                
                # Prefer pooler_output if available
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    batch_emb = vision_outputs.pooler_output
                    if i == 0:
                        print(f"   Using pooler_output: {batch_emb.shape}")
                else:
                    # Get last_hidden_state
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_hidden = vision_outputs.last_hidden_state
                    elif isinstance(vision_outputs, tuple):
                        vision_hidden = vision_outputs[0]
                    else:
                        vision_hidden = vision_outputs
                    
                    if i == 0:
                        print(f"   vision_hidden.shape: {vision_hidden.shape}")
                    
                    # Process vision_hidden: (batch, num_patches, hidden_dim)
                    if i == 0:
                        print(f"   vision_hidden.shape: {vision_hidden.shape}")
                        print(f"   vision_hidden.ndim: {vision_hidden.ndim}")
                    
                    if vision_hidden.ndim == 3:
                        # vision_hidden: (batch, num_patches=1024, hidden_dim=1536)
                        if self.pooling == "mean_patches":
                            # Average over patches dimension -> (batch, 1536)
                            batch_emb = vision_hidden.mean(dim=1)
                        elif self.pooling == "mean_hidden":
                            # Average over hidden dimension -> (batch, 1024) preserving spatial structure
                            batch_emb = vision_hidden.mean(dim=-1)
                        elif self.pooling == "concat":
                            # mean + std concatenation -> (batch, 3072) increasing discriminability
                            mean_emb = vision_hidden.mean(dim=1)
                            std_emb = vision_hidden.std(dim=1)
                            batch_emb = torch.cat([mean_emb, std_emb], dim=-1)
                        else:
                            batch_emb = vision_hidden.mean(dim=1)
                        
                        if i == 0:
                            print(f"   After pooling='{self.pooling}': batch_emb.shape = {batch_emb.shape}")
                    elif vision_hidden.ndim == 2:
                        batch_emb = vision_hidden
                        if i == 0:
                            print(f"   Already 2D: batch_emb.shape = {batch_emb.shape}")
                    else:
                        batch_emb = vision_hidden.reshape(len(valid_images), -1, vision_hidden.shape[-1]).mean(dim=1)
                        if i == 0:
                            print(f"   Other dimensions, after reshape: batch_emb.shape = {batch_emb.shape}")
                
                # Normalize
                batch_emb = batch_emb.float()
                batch_emb = batch_emb / (batch_emb.norm(dim=-1, keepdim=True) + 1e-8)
                batch_embeddings = batch_emb.detach().cpu().numpy()
                
                if self.embed_dim is None:
                    self.embed_dim = batch_embeddings.shape[1]
                    print(f"   Embedding dim: {self.embed_dim}")
                
                # Fill results in original order
                emb_idx = 0
                for j in range(len(batch_images)):
                    if j in valid_indices:
                        all_embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
                
                del pixel_values, batch_emb, batch_embeddings
                
            except Exception as e:
                print(f"⚠️  Batch {i//batch_size} failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to per-image processing
                for j, img in enumerate(batch_images):
                    if img is None or j not in valid_indices:
                        dim = self.embed_dim if self.embed_dim else 1536
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    else:
                        try:
                            emb = self._embed_single_image(img)
                            all_embeddings.append(emb)
                        except Exception as e2:
                            print(f"⚠️  Single image {i+j} failed: {e2}")
                            dim = self.embed_dim if self.embed_dim else 1536
                            all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)
    
    @torch.no_grad()
    def _embed_single_image(self, img: Image.Image) -> np.ndarray:
        """Process single image (fallback method)"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.image_processor is not None:
            inputs = self.image_processor(images=img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(
                device=self.vision_device, dtype=torch.bfloat16
            )
        else:
            pixel_values = self.transform(img).unsqueeze(0).to(
                device=self.vision_device, dtype=torch.bfloat16
            )
        
        vision_outputs = self.model(pixel_values)
        
        # Prefer pooler_output
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            emb = vision_outputs.pooler_output.squeeze(0)
        else:
            # Get last_hidden_state
            if hasattr(vision_outputs, 'last_hidden_state'):
                h = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, tuple):
                h = vision_outputs[0]
            else:
                h = vision_outputs

            # Pooling (SAILViT has no CLS token)
            if h.ndim == 3:
                # h: (1, num_patches=1024, hidden_dim=1536)
                if self.pooling == "mean_patches":
                    emb = h.mean(dim=1).squeeze(0)
                elif self.pooling == "mean_hidden":
                    emb = h.mean(dim=-1).squeeze(0)
                elif self.pooling == "concat":
                    mean_emb = h.mean(dim=1)
                    std_emb = h.std(dim=1)
                    emb = torch.cat([mean_emb, std_emb], dim=-1).squeeze(0)
                else:
                    emb = h.mean(dim=1).squeeze(0)
            elif h.ndim == 2:
                emb = h[0] if h.shape[0] > 1 else h.squeeze(0)
            else:
                emb = h.reshape(-1, h.shape[-1]).mean(dim=0)
        
        emb = emb.float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        emb_np = emb.detach().cpu().numpy()
        
        if self.embed_dim is None:
            self.embed_dim = emb_np.shape[0]
        
        return emb_np


class InternVLVisionEmbedder:
    """InternVL3.5-8B Vision Encoder (the model's own vision encoder)

    Loads the full InternVL3.5-8B model and uses its vision_model to extract image features.
    Reference: knn_image_vlm.py
    """
    
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
                
                # Debug info (only print first batch)
                if i == 0:
                    print(f"   Batch 0: {len(valid_images)} images, pixel_values.shape={pixel_values.shape}")
                
                # Extract vision features
                if hasattr(self.model, 'vision_model'):
                    # Use vision_model directly
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
                    print(f"   Embedding dim: {self.embed_dim}")
                
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
                print(f"⚠️  Batch {i//batch_size} failed: {e}, falling back to per-image processing")
                import traceback
                traceback.print_exc()
                # Fall back to per-image processing
                for j, img in enumerate(batch_images):
                    if img is None or j not in valid_indices:
                        dim = self.embed_dim if self.embed_dim else 3584
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    else:
                        try:
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
                            print(f"⚠️  Image {i+j} failed: {e2}")
                            dim = self.embed_dim if self.embed_dim else 3584
                            all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Clean GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)


class Qwen3VLVisionEmbedder:
    """Qwen3-VL Vision Encoder (the model's own vision encoder)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading Qwen3-VL: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.eval()
        self.visual = self.model.visual
        
        self.image_processor = self.processor.image_processor
        
        # Limit image resolution to avoid generating too many vision tokens
        self.image_processor.min_pixels = 128 * 28 * 28  # ~100K pixels
        self.image_processor.max_pixels = 384 * 28 * 28  # ~300K pixels
        print(f"   Image processor: min_pixels={self.image_processor.min_pixels}, max_pixels={self.image_processor.max_pixels}")
        
        self.device = self.model.device
        self.embed_dim = None
        
        print(f"✓ Qwen3-VL loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from a list of PIL Images - batch processing"""
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
                
                # Debug info (only print first batch)
                if i == 0:
                    total_tokens = sum(g[0].item() * g[1].item() * g[2].item() for g in image_grid_thw) if image_grid_thw is not None else 0
                    print(f"   Batch 0: {len(valid_images)} images, total_tokens={total_tokens}")
                
                # Batch forward
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
                # Need to split by token count per image
                if image_grid_thw is not None:
                    token_counts_raw = [g[0].item() * g[1].item() * g[2].item() for g in image_grid_thw]
                    total_raw = sum(token_counts_raw)
                    
                    # Compute merge ratio and adjust token counts
                    if vision_hidden.shape[0] != total_raw:
                        merge_ratio = total_raw / vision_hidden.shape[0]
                        token_counts = [int(t / merge_ratio) for t in token_counts_raw]
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
                    print(f"   Embedding dim: {self.embed_dim}")
                
                # Fill results in original order
                emb_idx = 0
                for j in range(len(batch_images)):
                    if j in valid_indices:
                        all_embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
                
            except Exception as e:
                print(f"⚠️  Batch {i//batch_size} failed: {e}, falling back to per-image processing")
                # Fall back to per-image processing
                for j, img in enumerate(batch_images):
                    if img is None or j not in valid_indices:
                        dim = self.embed_dim if self.embed_dim else 4096
                        all_embeddings.append(np.zeros(dim, dtype=np.float32))
                    else:
                        try:
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
                            print(f"⚠️  Image {i+j} failed: {e2}")
                            dim = self.embed_dim if self.embed_dim else 4096
                            all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Clean GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)


class Step3VLVisionEmbedder:
    """Step3-VL-10B Vision Encoder
    
    Step3-VL is based on the Qwen2-VL architecture, using its vision_model to extract image features.
    https://huggingface.co/stepfun-ai/Step3-VL-10B
    
    Features:
    - Requires KEY_MAPPING to correctly load model
    - vision_model output needs to go through projector processing
    - Supports dynamic resolution (similar to Qwen2-VL)
    """
    
    # Key mapping required by Step3-VL
    KEY_MAPPING = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "vit_large_projector": "model.vit_large_projector",
    }
    
    def __init__(self, model_name: str = "stepfun-ai/Step3-VL-10B", device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"Loading Step3-VL: {model_name}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Load full model (requires key_mapping)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            key_mapping=self.KEY_MAPPING
        ).eval()
        
        self.device = next(self.model.parameters()).device
        self.embed_dim = None
        
        # Get vision encoder and projector
        self._setup_vision_components()
        
        print(f"✓ Step3-VL loaded on {self.device}")
    
    def _setup_vision_components(self):
        """Set up vision component references"""
        # Step3-VL's vision_model is at model.model.vision_model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            self.vision_model = self.model.model.vision_model
            print(f"   Found vision_model at model.model.vision_model")
        elif hasattr(self.model, 'vision_model'):
            self.vision_model = self.model.vision_model
            print(f"   Found vision_model at model.vision_model")
        else:
            # Try to find by traversal
            self.vision_model = None
            for name, module in self.model.named_modules():
                if 'vision' in name.lower() and hasattr(module, 'forward'):
                    self.vision_model = module
                    print(f"   Found vision_model at {name}")
                    break
        
        if self.vision_model is None:
            print("   ⚠️ Could not find vision_model, will use full model forward")
        
        # Find projector (optional, for better embeddings)
        self.projector = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vit_large_projector'):
            self.projector = self.model.model.vit_large_projector
            print(f"   Found vit_large_projector")
        
        # Print vision config
        if hasattr(self.model.config, 'vision_config'):
            vc = self.model.config.vision_config
            if hasattr(vc, 'hidden_size'):
                print(f"   Vision hidden size: {vc.hidden_size}")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 8) -> np.ndarray:
        """Extract embeddings from a list of PIL Images - per-image processing (to avoid dynamic resolution issues)"""
        all_embeddings = []
        
        for i in tqdm(range(len(images)), desc="Step3-VL embedding"):
            img = images[i]
            
            if img is None:
                dim = self.embed_dim if self.embed_dim else 3584
                all_embeddings.append(np.zeros(dim, dtype=np.float32))
                continue
            
            try:
                emb = self._embed_single_image(img)
                all_embeddings.append(emb)
                
                if self.embed_dim is None:
                    self.embed_dim = emb.shape[0]
                    print(f"   Embedding dim: {self.embed_dim}")
                    
            except Exception as e:
                print(f"⚠️  Image {i} failed: {e}")
                dim = self.embed_dim if self.embed_dim else 3584
                all_embeddings.append(np.zeros(dim, dtype=np.float32))
            
            # Periodically clean GPU cache
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.stack(all_embeddings, axis=0)
    
    @torch.no_grad()
    def _embed_single_image(self, img: Image.Image) -> np.ndarray:
        """Process single image"""
        import io
        import base64
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert PIL Image to base64 URL (Step3-VL input format)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_base64}"
        
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": "Describe."}  # Simple prompt
                ]
            }
        ]
        
        # Apply chat template to get inputs
        try:
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            # Fallback: process image directly
            inputs = self.processor(images=img, return_tensors="pt")
        
        # Move to device and convert data types
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Extract vision embedding
        if self.vision_model is not None and 'pixel_values' in inputs:
            # Convert to bfloat16 (consistent with model weights)
            pixel_values = inputs['pixel_values'].to(dtype=torch.bfloat16)
            
            # Handle possible grid_thw parameter
            grid_thw = inputs.get('image_grid_thw')
            
            # Forward pass to get vision features
            if grid_thw is not None:
                try:
                    vision_outputs = self.vision_model(pixel_values, grid_thw=grid_thw)
                except TypeError:
                    vision_outputs = self.vision_model(pixel_values)
            else:
                vision_outputs = self.vision_model(pixel_values)
            
            # Get hidden states
            if hasattr(vision_outputs, 'last_hidden_state'):
                vision_hidden = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, tuple):
                vision_hidden = vision_outputs[0]
            else:
                vision_hidden = vision_outputs
            
            # Optional: pass through projector
            if self.projector is not None:
                try:
                    vision_hidden = self.projector(vision_hidden)
                except:
                    pass  # Skip if projector is incompatible
            
            # Pooling: average
            if vision_hidden.ndim == 3:
                # (batch, num_patches, hidden_dim)
                emb = vision_hidden.mean(dim=1).squeeze(0)
            elif vision_hidden.ndim == 2:
                # (num_patches, hidden_dim) or (batch, hidden_dim)
                emb = vision_hidden.mean(dim=0) if vision_hidden.shape[0] > 1 else vision_hidden.squeeze(0)
            else:
                emb = vision_hidden.reshape(-1, vision_hidden.shape[-1]).mean(dim=0)
        else:
            # Fallback: use model's encode method (if available)
            pixel_values = inputs['pixel_values'].to(dtype=torch.bfloat16)
            if hasattr(self.model, 'encode_images'):
                vision_hidden = self.model.encode_images(pixel_values)
            elif hasattr(self.model, 'get_image_features'):
                inputs['pixel_values'] = pixel_values
                vision_hidden = self.model.get_image_features(**inputs)
            else:
                # Last fallback: use full forward, extract intermediate layer
                inputs['pixel_values'] = pixel_values
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the last layer's hidden state
                vision_hidden = outputs.hidden_states[-1]
            
            if vision_hidden.ndim == 3:
                emb = vision_hidden.mean(dim=1).squeeze(0)
            else:
                emb = vision_hidden.mean(dim=0) if vision_hidden.ndim == 2 else vision_hidden.flatten()
        
        # Normalize
        emb = emb.float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

        return emb.detach().cpu().numpy()


def get_embedder(method: str, device: str = "cuda", pooling: str = "mean_patches"):
    """Get corresponding embedder by method name"""
    method = method.lower()
    
    if method == "metaclip":
        return MetaCLIPEmbedder(device=device)
    elif method == "sailvl":
        return SAILVLVisionEmbedder(device=device, pooling=pooling)
    elif method == "internvl":
        return InternVLVisionEmbedder(device=device)
    elif method == "qwen" or method == "qwen3vl":
        return Qwen3VLVisionEmbedder(device=device)
    elif method == "step3vl" or method == "step3-vl":
        return Step3VLVisionEmbedder(device=device)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: metaclip, sailvl, internvl, qwen, step3vl")


def get_default_batch_size(method: str) -> int:
    """Get default batch size"""
    method = method.lower()
    if method == "metaclip":
        return 32
    elif method == "sailvl":
        return 32
    elif method == "internvl":
        return 16  # InternVL 8B model is larger
    elif method == "qwen" or method == "qwen3vl":
        return 16  # Qwen3-VL 8B model
    elif method == "step3vl" or method == "step3-vl":
        return 8  # Step3-VL 10B model, per-image processing
    return 16


def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find neighbors within the dataset itself (excluding self)
    emb: (N, D) float32, already normalized.
    Returns:
      nn_idx: (N, k) int32
      nn_score: (N, k) float32
    """
    N, D = emb.shape
    nn_idx = np.empty((N, k), dtype=np.int32)
    nn_score = np.empty((N, k), dtype=np.float32)

    for start in range(0, N, query_block):
        end = min(start + query_block, N)
        q = emb[start:end]                 # (B, D)
        scores = q @ emb.T                 # (B, N)

        # Exclude self: set diagonal positions to -inf for this block
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        scores[rows, cols] = -np.inf

        # Get top-k indices per row (unsorted), then sort them by score desc
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]  # (B, k)
        top_sc = np.take_along_axis(scores, top_idx, axis=1)        # (B, k)

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
    """
    Find neighbors for query samples (query_emb) in the full image database (all_emb)

    Args:
        query_emb: (Q, D) query sample embeddings
        all_emb: (N, D) all image embeddings
        query_indices: original indices of query samples in the full image set (for self-exclusion)
        k: number of neighbors
        query_block: query batch size

    Returns:
        nn_idx: (Q, k) neighbor indices in the full image set
        nn_score: (Q, k) similarity scores
    """
    Q = query_emb.shape[0]
    N = all_emb.shape[0]
    
    nn_idx = np.empty((Q, k), dtype=np.int32)
    nn_score = np.empty((Q, k), dtype=np.float32)
    
    for start in range(0, Q, query_block):
        end = min(start + query_block, Q)
        q = query_emb[start:end]           # (B, D)
        scores = q @ all_emb.T             # (B, N)
        
        # Exclude self: set original index positions for each query to -inf
        for i, qi in enumerate(range(start, end)):
            original_idx = query_indices[qi]
            if 0 <= original_idx < N:
                scores[i, original_idx] = -np.inf
        
        # Get top-k
        top_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        top_sc = np.take_along_axis(scores, top_idx, axis=1)
        
        order = np.argsort(-top_sc, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sc = np.take_along_axis(top_sc, order, axis=1)
        
        nn_idx[start:end] = top_idx.astype(np.int32)
        nn_score[start:end] = top_sc.astype(np.float32)
        
        print(f"Processed queries [{start}:{end}) / {Q} (searching in {N} images)")
    
    return nn_idx, nn_score


def main():
    parser = argparse.ArgumentParser(description="Find KNN for images using various embedding methods")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input VLM result JSON file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (flickr30k/coco), takes priority")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--full_dataset", action="store_true",
                        help="Search for neighbors in the entire dataset (not just samples in input_json)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Local image directory (if not using HuggingFace dataset)")
    parser.add_argument("--out_jsonl", type=str, required=True,
                        help="Output neighbor JSONL file")
    parser.add_argument("--method", type=str, default="metaclip",
                        choices=["metaclip", "sailvl", "internvl", "qwen", "step3vl"],
                        help="Embedding method: metaclip(public), sailvl(SAIL-VL native), internvl(InternVL native), qwen(Qwen3-VL native), step3vl(Step3-VL native)")
    parser.add_argument("--pooling", type=str, default="concat",
                        choices=["mean_patches", "mean_hidden", "concat"],
                        help="SAILViT pooling strategy: mean_patches(1536), mean_hidden(1024), concat(3072, default)")
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
    print("Image KNN Neighbor Search")
    print("=" * 80)
    print(f"Input file: {args.input_json}")
    print(f"Embedding method: {args.method}")
    if args.method == "sailvl":
        print(f"Pooling strategy: {args.pooling}")
    if args.dataset:
        print(f"Dataset: {args.dataset} (split={args.split})")
        if args.full_dataset:
            print(f"Search scope: entire dataset")
        else:
            print(f"Search scope: only samples in input_json")
    elif args.image_dir:
        print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.out_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of neighbors: {args.k}")
    print("=" * 80)

    # 1) Load VLM result JSON
    print("\nLoading VLM results...")
    items = load_json(args.input_json)
    N = len(items)
    print(f"Loaded {N} samples")

    # Get index list of samples in input_json
    query_indices = [item.get("index", i) for i, item in enumerate(items)]

    # 2) Load images
    if args.dataset:
        if args.full_dataset:
            # Load entire dataset as search library
            print(f"\nLoading full {args.dataset} dataset from HuggingFace...")
            hf_dataset = load_hf_dataset(args.dataset, split=args.split, num_samples=None)
            total_images = len(hf_dataset)
            print(f"Dataset has {total_images} images in total")

            # Load all images
            print("\nLoading all images...")
            all_images: List[Optional[Image.Image]] = []
            for i in tqdm(range(total_images), desc="Loading all images"):
                image = get_image_from_hf(hf_dataset, i)
                all_images.append(image)
            
            # Extract all image embeddings
            print(f"\nExtracting all image embeddings using {args.method}...")
            embedder = get_embedder(args.method, device=args.device, pooling=args.pooling)
            all_emb = embedder.embed_images_pil(all_images, batch_size=args.batch_size)
            print(f"Embedding shape: {all_emb.shape}")
            
            # Get query sample embeddings (extract from full embeddings)
            query_emb = all_emb[query_indices]
            
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
            
            # KNN search: find neighbors for query samples in all images
            print("\nSearching for nearest neighbors in full dataset...")
            nn_idx, nn_score = topk_neighbors_query_vs_all(
                query_emb, all_emb, query_indices,
                k=args.k, query_block=args.query_block
            )
        else:
            # Only load samples from input_json
            print(f"\nLoading {args.dataset} dataset from HuggingFace...")
            hf_dataset = load_hf_dataset(args.dataset, split=args.split, num_samples=None)
            
            images: List[Optional[Image.Image]] = []
            meta = []
            print("\nExtracting images...")
            for i, item in enumerate(tqdm(items, desc="Loading images")):
                idx = item.get("index", i)
                image = get_image_from_hf(hf_dataset, idx)
                images.append(image)
                meta.append({
                    "global_id": i,
                    "source_file": os.path.basename(args.input_json),
                    "row_in_file": i,
                    "index": idx,
                })
            
            # Count valid images
            valid_count = sum(1 for img in images if img is not None)
            print(f"\nValid images: {valid_count} / {N}")
            
            # Extract embeddings
            print(f"\nExtracting image embeddings using {args.method}...")
            embedder = get_embedder(args.method, device=args.device, pooling=args.pooling)
            full_emb = embedder.embed_images_pil(images, batch_size=args.batch_size)
            print(f"Embedding shape: {full_emb.shape}")
            
            # KNN search
            print("\nSearching for nearest neighbors...")
            nn_idx, nn_score = topk_neighbors_all(full_emb, k=args.k, query_block=args.query_block)
    
    elif args.image_dir:
        # Load images from local directory
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
                    print(f"⚠️  Failed to load {img_path}: {e}")
                    images.append(None)
            else:
                images.append(None)
            
            meta.append({
                "global_id": i,
                "source_file": os.path.basename(args.input_json),
                "row_in_file": i,
                "index": item.get("index", i),
            })
        
        # Count valid images
        valid_count = sum(1 for img in images if img is not None)
        print(f"\nValid images: {valid_count} / {N}")

        # Extract embeddings
        print(f"\nExtracting image embeddings using {args.method}...")
        embedder = get_embedder(args.method, device=args.device, pooling=args.pooling)
        full_emb = embedder.embed_images_pil(images, batch_size=args.batch_size)
        print(f"Embedding shape: {full_emb.shape}")

        # KNN search
        print("\nSearching for nearest neighbors...")
        nn_idx, nn_score = topk_neighbors_all(full_emb, k=args.k, query_block=args.query_block)
    else:
        raise ValueError("Must specify --dataset or --image_dir")

    # 5) Save results
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

    print(f"\nNeighbor results saved to: {args.out_jsonl}")
    print(f"   Total samples: {N}")
    print(f"   Neighbors per sample: {args.k}")
    print(f"   Embedding method: {args.method}")


if __name__ == "__main__":
    main()
