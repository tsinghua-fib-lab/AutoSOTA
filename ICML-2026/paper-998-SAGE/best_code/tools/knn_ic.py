"""
unified_image_knn.py

Unified image KNN computation script (for Image Classification tasks)

Supported embedding methods:
- Qwen3-VL: Qwen/Qwen3-VL-8B-Instruct
- CLIP: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- MetaCLIP: facebook/metaclip-2-worldwide-huge-quickgelu

Supported datasets:
- CIFAR-10
- CIFAR-100
- ImageNet-1k
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import torch
from datetime import datetime


# ==================== Embedding Methods ====================

class Qwen3VLVisionEmbedder:
    """Qwen3-VL Vision Encoder (official implementation)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading Qwen3-VL: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.visual = self.model.visual
        self.image_processor = self.processor.image_processor
        self.device = self.model.device
        self.embed_dim = None
        
        print(f"✓ Qwen3-VL loaded")
        if hasattr(self.model.config, 'vision_config'):
            print(f"   Vision hidden size: {self.model.config.vision_config.hidden_size}")
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from a list of PIL Images"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Qwen3-VL embedding"):
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
                inputs = self.image_processor(images=valid_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                image_grid_thw = inputs.get('image_grid_thw')
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)
                
                if image_grid_thw is not None:
                    vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    vision_outputs = self.visual(pixel_values)
                
                # Handle different output formats
                if isinstance(vision_outputs, tuple):
                    vision_hidden = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_hidden = vision_outputs.last_hidden_state
                else:
                    vision_hidden = vision_outputs
                
                num_images = len(valid_images)
                
                # Check vision_hidden shape
                # Qwen3-VL output: (total_patches, hidden_dim) where total_patches = sum(t*h*w for each image)
                if image_grid_thw is not None and len(vision_hidden.shape) == 2:
                    # Compute patches count per image (t * h * w)
                    patches_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
                    total_expected = sum(patches_per_image)
                    
                    # Verify dimension match
                    if vision_hidden.shape[0] != total_expected:
                        # Qwen3-VL compresses vision features (e.g., through pooling), this is normal
                        patches_per_image_fallback = vision_hidden.shape[0] // num_images
                        compression_ratio = total_expected / vision_hidden.shape[0]
                        print(f"Qwen3-VL vision feature compression: {total_expected} -> {vision_hidden.shape[0]} patches ({compression_ratio:.1f}x)")
                        print(f"    Per image: {patches_per_image_fallback} tokens (hidden_dim={vision_hidden.shape[1]})")
                        # Evenly distribute then average
                        vision_hidden_reshaped = vision_hidden[:num_images * patches_per_image_fallback].reshape(
                            num_images, patches_per_image_fallback, -1
                        )
                        embeddings = vision_hidden_reshaped.mean(dim=1)
                    else:
                        # Normal processing: split by patches
                        embeddings_list = []
                        start_idx = 0
                        for num_patches in patches_per_image:
                            end_idx = start_idx + int(num_patches)
                            image_patches = vision_hidden[start_idx:end_idx]
                            image_embedding = image_patches.mean(dim=0)
                            embeddings_list.append(image_embedding)
                            start_idx = end_idx
                        embeddings = torch.stack(embeddings_list)
                else:
                    # Fallback: assume batch dimension or uniform distribution
                    if len(vision_hidden.shape) == 3:
                        # (batch, patches, hidden_dim)
                        embeddings = vision_hidden.mean(dim=1)
                    else:
                        # (total_patches, hidden_dim) evenly distribute
                        total_patches = vision_hidden.shape[0]
                        patches_per_img = total_patches // num_images
                        if patches_per_img > 0:
                            vision_hidden_reshaped = vision_hidden[:num_images * patches_per_img].reshape(
                                num_images, patches_per_img, -1
                            )
                            embeddings = vision_hidden_reshaped.mean(dim=1)
                        else:
                            # Edge case: patches count less than image count
                            embeddings = vision_hidden[:num_images]
                
                # Normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embeddings = embeddings.cpu().float().numpy()
                
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
                dim = self.embed_dim if self.embed_dim else 4096  # Use actual dimension
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)
    # @torch.no_grad()
    # def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 16) -> np.ndarray:
    #     """Extract embeddings from a list of PIL Images"""
    #     all_embeddings = []
        
    #     for i in tqdm(range(0, len(images), batch_size), desc="Qwen3-VL embedding"):
    #         batch_images = images[i:i+batch_size]
            
    #         # Filter valid images and record positions
    #         valid_images = []
    #         valid_positions = []
    #         for j, img in enumerate(batch_images):
    #             if img is not None:
    #                 if img.mode != 'RGB':
    #                     img = img.convert('RGB')
    #                 valid_images.append(img)
    #                 valid_positions.append(j)
            
    #         if valid_images:
    #             inputs = self.image_processor(images=valid_images, return_tensors="pt")
    #             pixel_values = inputs['pixel_values'].to(self.device)
    #             image_grid_thw = inputs.get('image_grid_thw')
    #             if image_grid_thw is not None:
    #                 image_grid_thw = image_grid_thw.to(self.device)
                
    #             if image_grid_thw is not None:
    #                 vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
    #             else:
    #                 vision_outputs = self.visual(pixel_values)
                
    #             if isinstance(vision_outputs, tuple):
    #                 vision_hidden = vision_outputs[0]
    #             elif hasattr(vision_outputs, 'last_hidden_state'):
    #                 vision_hidden = vision_outputs.last_hidden_state
    #             else:
    #                 vision_hidden = vision_outputs
                
    #             embeddings = vision_hidden.mean(dim=1)
    #             embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    #             embeddings = embeddings.cpu().float().numpy()
                
    #             if self.embed_dim is None:
    #                 self.embed_dim = embeddings.shape[1]
                
    #             emb_idx = 0
    #             for j, img in enumerate(batch_images):
    #                 if img is not None:
    #                     all_embeddings.append(embeddings[emb_idx])
    #                     emb_idx += 1
    #                 else:
    #                     all_embeddings.append(np.zeros(self.embed_dim, dtype=np.float32))
    #         else:
    #             dim = self.embed_dim if self.embed_dim else 3584
    #             for _ in batch_images:
    #                 all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
    #     return np.stack(all_embeddings, axis=0)


class CLIPEmbedder:
    """CLIP Image Encoder"""
    
    def __init__(self, model_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        from transformers import CLIPModel, CLIPProcessor
        
        print(f"Loading CLIP: {model_name}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to("cuda")
        self.model.eval()
        self.device = self.model.device
        self.embed_dim = None
        print(f"✓ CLIP loaded")
    
    @torch.no_grad()
    def embed_images_pil(self, images: List[Optional[Image.Image]], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from a list of PIL Images"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP embedding"):
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
                embeddings = embeddings.cpu().float().numpy()
                
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
                dim = self.embed_dim if self.embed_dim else 1024
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)


class MetaCLIPEmbedder:
    """MetaCLIP Image Encoder"""
    
    def __init__(self, model_name: str = "facebook/metaclip-2-worldwide-huge-quickgelu"):
        from transformers import AutoModel, AutoProcessor
        
        print(f"Loading MetaCLIP: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.model.eval()
        self.device = self.model.device
        self.embed_dim = None
        print(f"✓ MetaCLIP loaded")
    
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

                embeddings = embeddings.cpu().float().numpy()
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
                dim = self.embed_dim if self.embed_dim else 1152
                for _ in batch_images:
                    all_embeddings.append(np.zeros(dim, dtype=np.float32))
        
        return np.stack(all_embeddings, axis=0)


# ==================== Dataset Loading ====================

def load_dataset_images(dataset_name: str, max_samples: int = 10000) -> Tuple[List[Optional[Image.Image]], str]:
    """Load the first N images from HuggingFace dataset (using streaming to avoid downloading the entire dataset)"""
    from datasets import load_dataset
    
    print(f"\nLoading dataset: {dataset_name} (max={max_samples})...")
    
    # Dataset configuration
    dataset_configs = {
        "cifar100": {"path": "cifar100", "split": "test", "image_field": "img", "use_streaming": False},
        "cifar10": {"path": "cifar10", "split": "test", "image_field": "img", "use_streaming": False},
        "imagenet-1k": {"path": "imagenet-1k", "split": "validation", "image_field": "image", "use_streaming": True},
    }
    
    config = dataset_configs.get(dataset_name.lower())
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    image_field = config["image_field"]
    
    # ImageNet uses streaming mode (to avoid downloading the entire dataset)
    if config["use_streaming"]:
        print(f"  Using streaming mode (only downloading {max_samples} samples)...")
        dataset = load_dataset(config["path"], split=config["split"], streaming=True, trust_remote_code=True)
        
        images = []
        for i, item in enumerate(tqdm(dataset, total=max_samples, desc="Loading images")):
            if i >= max_samples:
                break
            try:
                img = item[image_field]
                images.append(img)
            except Exception as e:
                print(f"⚠️  Failed to load image {i}: {e}")
                images.append(None)
        
        valid_count = sum(1 for img in images if img is not None)
        print(f"✓ Loaded {len(images)} images ({valid_count} valid)")
        return images, dataset_name
    
    # Small datasets like CIFAR are loaded directly
    dataset = load_dataset(config["path"], split=config["split"], trust_remote_code=True)
    
    total = len(dataset)
    n_samples = min(max_samples, total)
    
    print(f"Loading {n_samples} images (out of {total} total)...")
    images = []
    for i in tqdm(range(n_samples), desc="Loading images"):
        try:
            img = dataset[i][image_field]
            images.append(img)
        except Exception as e:
            print(f"⚠️  Failed to load image {i}: {e}")
            images.append(None)
    
    valid_count = sum(1 for img in images if img is not None)
    print(f"✓ Loaded {len(images)} images ({valid_count} valid)")
    
    return images, dataset_name


# ==================== KNN Computation ====================

def topk_neighbors_all(
    emb: np.ndarray,
    k: int = 9,
    query_block: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors within the dataset itself (excluding self)"""
    print(f"\nComputing KNN (k={k})...")
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
        if (start // query_block) % 10 == 0:
            print(f"  Processed queries [{start}:{end}) / {N}")
    
    return nn_idx, nn_score


# ==================== Output Formatting ====================

def save_neighbors_jsonl(
    knn_indices: np.ndarray,
    knn_scores: np.ndarray,
    dataset_name: str,
    method_name: str,
    output_path: str
):
    """Save as standard JSONL format"""
    print(f"\nSaving to {output_path}...")
    
    N, k = knn_indices.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_file = f"{dataset_name}_{method_name}_{timestamp}.jsonl"
    
    with open(output_path, 'w') as f:
        for i in range(N):
            item = {
                'global_id': i,
                'source_file': source_file,
                'row_in_file': i,
                'index': i,
                'neighbors': [
                    {
                        'global_id': int(knn_indices[i, j]),
                        'cosine': float(knn_scores[i, j])
                    }
                    for j in range(k)
                ]
            }
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Saved {N} samples")


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Image Classification KNN (knn_ic)")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar100', 'cifar10', 'imagenet-1k'],
                       help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                       choices=['qwen3vl', 'clip', 'metaclip'],
                       help='Embedding method')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples (default: 10000)')
    parser.add_argument('--k', type=int, default=9,
                       help='Number of neighbors (default: 9)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: outputs/image_classification)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto if not specified)')
    parser.add_argument('--query-block', type=int, default=256,
                       help='KNN query block size')
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent.parent
        args.output_dir = script_dir / "outputs" / "image_classification"
    
    if args.batch_size is None:
        if args.method == 'qwen3vl':
            args.batch_size = 16
        else:
            args.batch_size = 32
    
    print("=" * 80)
    print("🖼️  Image Classification KNN Calculator")
    print("=" * 80)
    print(f"Dataset:     {args.dataset}")
    print(f"Method:      {args.method}")
    print(f"Samples:     {args.max_samples}")
    print(f"Neighbors:   {args.k}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 80)
    
    # Load dataset
    images, dataset_name = load_dataset_images(args.dataset, args.max_samples)
    
    # Initialize embedder
    print(f"\nInitializing {args.method} embedder...")
    if args.method == 'qwen3vl':
        embedder = Qwen3VLVisionEmbedder("Qwen/Qwen3-VL-8B-Instruct")
    elif args.method == 'clip':
        embedder = CLIPEmbedder("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    elif args.method == 'metaclip':
        embedder = MetaCLIPEmbedder("facebook/metaclip-2-worldwide-huge-quickgelu")
    
    # Extract embeddings
    print(f"\n🎨 Extracting embeddings...")
    embeddings = embedder.embed_images_pil(images, batch_size=args.batch_size)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    # Compute KNN
    knn_indices, knn_scores = topk_neighbors_all(
        embeddings,
        k=args.k,
        query_block=args.query_block
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_{args.method}_neighbors.jsonl"
    
    save_neighbors_jsonl(knn_indices, knn_scores, dataset_name, args.method, str(output_path))
    
    # Statistics
    print("\n" + "=" * 80)
    print("📊 Statistics")
    print("=" * 80)
    print(f"Total samples:        {len(knn_indices)}")
    print(f"Neighbors per sample: {args.k}")
    print(f"Avg similarity:       {knn_scores.mean():.4f}")
    print(f"Min similarity:       {knn_scores.min():.4f}")
    print(f"Max similarity:       {knn_scores.max():.4f}")
    print("=" * 80)
    
    print(f"\n✅ Done! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
