"""
Memory-efficient utilities for handling embeddings and training optimization.
"""

import torch
from typing import Tuple
import gc

def save_embeddings_compressed(embeddings: torch.Tensor, labels: torch.Tensor, 
                              embeddings_path: str, labels_path: str, 
                              compression: str = 'none') -> None:
    """
    Save embeddings with optional compression to reduce storage space.
    
    Args:
        embeddings: Tensor of embeddings to save
        labels: Tensor of labels to save
        embeddings_path: Path to save embeddings
        labels_path: Path to save labels
        compression: Type of compression ('none', 'half_precision', 'quantized')
    """
    if compression == 'half_precision':
        # Convert to half precision to save memory
        embeddings = embeddings.half()
        print(f'Converted embeddings to half precision (float16)')
    elif compression == 'quantized':
        # Simple quantization (could be improved with more sophisticated methods)
        embeddings = torch.quantize_per_tensor(embeddings, scale=1e-3, zero_point=0, dtype=torch.qint8)
        print(f'Quantized embeddings to int8')
    
    torch.save(embeddings, embeddings_path)
    torch.save(labels, labels_path)
    
    print(f'Embeddings saved to {embeddings_path} with shape {embeddings.shape}')
    print(f'Labels saved to {labels_path} with shape {labels.shape}')

def load_embeddings_efficient(embeddings_path: str, labels_path: str, 
                             device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load embeddings efficiently with proper memory management.
    
    Args:
        embeddings_path: Path to embeddings file
        labels_path: Path to labels file
        device: Device to load tensors on initially
        
    Returns:
        Tuple of (embeddings, labels)
    """
    # Load on CPU first to avoid GPU memory issues
    embeddings = torch.load(embeddings_path, map_location='cpu')
    labels = torch.load(labels_path, map_location='cpu')
    
    # Handle quantized embeddings
    if embeddings.dtype == torch.qint8:
        embeddings = embeddings.dequantize()
        print('Dequantized embeddings from int8')
    
    # Convert half precision back to float if needed
    if embeddings.dtype == torch.float16:
        embeddings = embeddings.float()
        print('Converted embeddings from half to full precision')
    
    return embeddings, labels

def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print('Cleared GPU cache and ran garbage collection')

def get_memory_info() -> dict:
    """Get current memory usage information. SLURM-compatible version."""
    info = {}
    
    # GPU memory info - usually available even on SLURM
    if torch.cuda.is_available():
        try:
            info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
            info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        except Exception as e:
            print(f"Warning: Could not get GPU memory info - {e}")
    
    # CPU memory info with SLURM compatibility
    try:
        import psutil
        process = psutil.Process()
        info['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
    except ImportError:
        info['cpu_memory'] = 0.0
        print("Warning: psutil not available, CPU memory info unavailable")
    except Exception as e:
        info['cpu_memory'] = 0.0
        print(f"Warning: Could not get CPU memory info - {e}")
    
    return info

def print_memory_usage(stage: str = "", detailed: bool = False):
    """
    Print current memory usage for both CPU and GPU.
    SLURM-compatible version with graceful fallbacks.
    
    Args:
        stage: Description of the current stage (e.g., "before training")
        detailed: Whether to print detailed memory breakdown
    """
    print(f"\n{'='*50}")
    print(f"Memory Usage - {stage}")
    print(f"{'='*50}")
    
    # CPU Memory - SLURM compatible
    try:
        import psutil
        process = psutil.Process()
        cpu_memory = process.memory_info()
        
        print(f"CPU Memory:")
        print(f"  Current Process: {cpu_memory.rss / 1024**3:.2f} GB")
        
        try:
            cpu_percent = process.memory_percent()
            print(f"  Process %: {cpu_percent:.1f}%")
        except:
            print(f"  Process %: Not available (SLURM restriction)")
        
        if detailed:
            try:
                virtual_memory = psutil.virtual_memory()
                print(f"  System Total: {virtual_memory.total / 1024**3:.2f} GB")
                print(f"  System Available: {virtual_memory.available / 1024**3:.2f} GB")
                print(f"  System Used: {virtual_memory.used / 1024**3:.2f} GB ({virtual_memory.percent:.1f}%)")
            except:
                print(f"  System info: Not available (SLURM restriction)")
                
    except ImportError:
        print("CPU Memory: psutil not available")
        # Fallback: try to read /proc/meminfo if available
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                for line in lines[:3]:  # MemTotal, MemFree, MemAvailable
                    if any(x in line for x in ['MemTotal', 'MemFree', 'MemAvailable']):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            gb = kb / 1024**2
                            print(f"  {parts[0]}: {gb:.2f} GB")
        except:
            print("  Fallback memory info not available")
    except Exception as e:
        print(f"CPU Memory: Error accessing system info - {e}")
    
    # GPU Memory - Always available on CUDA systems
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                
                print(f"  GPU {i}:")
                print(f"    Allocated: {allocated:.2f} GB")
                print(f"    Reserved: {reserved:.2f} GB")
                print(f"    Max Allocated: {max_allocated:.2f} GB")
                
                if detailed:
                    try:
                        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        print(f"    Total: {total_memory:.2f} GB")
                        print(f"    Utilization: {(allocated/total_memory)*100:.1f}%")
                    except:
                        pass
            except Exception as e:
                print(f"  GPU {i}: Error accessing memory info - {e}")
    else:
        print("\nGPU Memory: CUDA not available")
    
    print(f"{'='*50}\n")

class EfficientEmbeddingDataset:
    """
    A memory-efficient dataset class that loads embeddings on-demand.
    Useful for very large embedding files that don't fit in memory.
    """
    
    def __init__(self, embeddings_path: str, labels_path: str):
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        
        # Load labels (usually much smaller)
        self.labels = torch.load(labels_path, map_location='cpu')
        
        # Get embedding dimensions without loading the full tensor
        sample_embeddings = torch.load(embeddings_path, map_location='cpu')
        self.embedding_dim = sample_embeddings.shape[1]
        self.num_samples = len(self.labels)
        del sample_embeddings  # Free memory
        
        print(f"Initialized efficient dataset with {self.num_samples} samples, "
              f"embedding dim: {self.embedding_dim}")
    
    def __len__(self):
        return self.num_samples
    
    def load_embeddings(self):
        """Load embeddings when needed."""
        if not hasattr(self, 'embeddings'):
            self.embeddings = torch.load(self.embeddings_path, map_location='cpu')
            print("Loaded embeddings into memory")
        return self.embeddings
    
    def get_batch(self, indices):
        """Get a batch of embeddings by indices."""
        embeddings = self.load_embeddings()
        return embeddings[indices], self.labels[indices]
