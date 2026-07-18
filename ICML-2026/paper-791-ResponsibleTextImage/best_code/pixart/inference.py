"""
External Heads Guided Image Generation with PixArt-Alpha - FIXED VERSION
=========================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
PROMPT = "a photo of a doctor"
BASE_SEED = 42
NUM_SEEDS = 1 # Generate with 5 different seeds
NUM_INFERENCE_STEPS = 20

# External heads settings
EXTERNAL_HEADS_PATH = "./external_heads.pt"
TARGET_LAYERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
TARGET_HEADS = [11] # None = all heads, or list like [0, 1, 2] for specific heads
COEFFICIENT_LIST = [0, 10]

OUTPUT_DIR = "images_out"


# ============================================================================
# EXTERNAL HEADS MODULE - FIXED VERSION
# ============================================================================
class LoadedExternalHeads:
    """
    Container for loaded external heads with per-key sequence length support.
    
    FIXES:
    - Normalizes keys on load (strips "external_heads." prefix)
    - Uses per-tensor sequence length instead of global max
    - Converts device/dtype immediately
    - Validates sequence length matches at runtime (no silent interpolation)
    """
    
    def __init__(self, state_dict, target_layers, num_heads=16, head_dim=72):
        self.target_layers = target_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.external_heads = {}
        
        # Load and normalize keys from state dict
        for key, value in state_dict.items():
            # Strip module prefix if present (e.g., "external_heads.layer_11_head_0" -> "layer_11_head_0")
            normalized_key = key
            if normalized_key.startswith("external_heads."):
                normalized_key = normalized_key[len("external_heads."):]
            
            # Store tensor (will convert device/dtype on access)
            self.external_heads[normalized_key] = value
        
        print(f"✓ Loaded external heads:")
        print(f"  Target layers: {target_layers}")
        print(f"  Heads per layer: {num_heads}")
        print(f"  Head dimension: {head_dim}")
        print(f"  Total head tensors: {len(self.external_heads)}")
        
        # Show some example shapes for verification
        if len(self.external_heads) > 0:
            example_keys = list(self.external_heads.keys())[:3]
            print(f"  Example shapes:")
            for k in example_keys:
                print(f"    {k}: {self.external_heads[k].shape}")
    
    def get_external_head(self, layer_idx, head_idx, seq_len, device, dtype, target_heads=None):
        """
        Get external head for a specific layer and head index.

        
        Args:
            layer_idx: Layer index
            head_idx: Head index (0-15)
            seq_len: Actual sequence length in this batch
            device: Device for tensor
            dtype: Data type for tensor
            target_heads: None (all heads) or list of head indices to use
            
        Returns:
            External head tensor [seq_len, head_dim]
        """
        # Check if this head should be modified
        if target_heads is not None and head_idx not in target_heads:
            # Return zeros for heads not in target list
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        
        if layer_idx not in self.target_layers:
            # Return zeros for layers we're not using
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        
        key = f"layer_{layer_idx}_head_{head_idx}"
        
        if key not in self.external_heads:
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        
        # Get the stored tensor and convert device/dtype IMMEDIATELY
        full_head = self.external_heads[key].to(device=device, dtype=dtype)  # [S_l, head_dim]
        S_l = full_head.shape[0]
        
        # CRITICAL: Enforce exact sequence length match
        # This prevents silent errors from mismatched training/inference resolutions
        if S_l != seq_len:
            raise ValueError(
                f"Sequence length mismatch for {key}: "
                f"stored S_l={S_l}, runtime seq_len={seq_len}. "
                f"This indicates a mismatch between training and inference grid sizes. "
                f"If you want to allow interpolation, implement 2D bilinear resizing here."
            )
        
        return full_head
    
    def get_all_heads_for_layer(self, layer_idx, seq_len, device, dtype, target_heads=None):
        """
        Get all external heads for a specific layer.
        
        Args:
            layer_idx: Layer index
            seq_len: Actual sequence length in this batch
            device: Device for tensor
            dtype: Data type for tensor
            target_heads: None (all heads) or list of head indices to use
            
        Returns:
            Tensor [num_heads, seq_len, head_dim]
        """
        heads = []
        for head_idx in range(self.num_heads):
            head = self.get_external_head(layer_idx, head_idx, seq_len, device, dtype, target_heads)
            heads.append(head)
        
        # Stack: [num_heads, seq_len, head_dim]
        return torch.stack(heads, dim=0)


# ============================================================================
# TRUE DELEGATION EXTERNAL HEADS PROCESSOR WITH BIAS-FREE PROJECTION - FIXED
# ============================================================================
class TrueDelegationExternalHeadsProcessor:
    """
    Attention processor using TRUE DELEGATION pattern with BIAS-FREE projection:

    """

    def __init__(self, original_processor, layer_idx: int, attn_module, 
                 external_heads_module, coefficient: float, target_heads=None):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.attn_module = attn_module
        self.external_heads_module = external_heads_module
        self.coefficient = coefficient
        self.target_heads = target_heads  # None = all heads, or list of specific head indices
        self.first_call = True
        
        # Cache attention configuration
        self.num_heads = getattr(attn_module, 'heads', None)
        if self.num_heads is None:
            raise ValueError(f"Layer {layer_idx}: Attention module missing 'heads' attribute")
        
        # Infer head dimension from to_q output
        if hasattr(attn_module.to_q, 'out_features'):
            inner_dim = attn_module.to_q.out_features
        elif hasattr(attn_module.to_q, 'weight'):
            inner_dim = attn_module.to_q.weight.shape[0]
        else:
            raise ValueError(f"Layer {layer_idx}: Cannot determine inner_dim from to_q")
        
        self.head_dim = inner_dim // self.num_heads
        
        # CRITICAL: Validate head dimension matches external heads module
        if self.external_heads_module.head_dim != self.head_dim:
            raise ValueError(
                f"Layer {layer_idx}: Head dimension mismatch! "
                f"External heads module: {self.external_heads_module.head_dim}, "
                f"Attention module: {self.head_dim}"
            )
        
        # Print info once
        heads_info = f"heads {target_heads}" if target_heads else "all heads"
        print(f"  Layer {layer_idx}: {self.num_heads} heads ({heads_info}), head_dim={self.head_dim}, coef={coefficient}")

    def _apply_to_out_bias_free(self, attn, delta_concat):
        """
        Apply the to_out projection to the delta WITHOUT bias.
        
        This ensures that when delta_concat is zero (or coefficient=0), 
        the output is exactly zero, preventing spurious offsets from bias terms.
        
        Args:
            attn: Attention module
            delta_concat: Delta in concatenated form [B, N, H*d_h]
            
        Returns:
            Projected delta (bias-free)
        """
        if not hasattr(attn, 'to_out') or attn.to_out is None:
            return delta_concat
        
        # ---- BIAS-FREE PROJECTION OF delta_concat ----
        to_out = attn.to_out
        _delta_proj = delta_concat
        
        if isinstance(to_out, torch.nn.ModuleList):
            # Handle ModuleList by applying each module sequentially
            for i, _m in enumerate(to_out):
                if isinstance(_m, torch.nn.Linear):
                    # Apply Linear layer WITHOUT bias
                    _W = _m.weight
                    _delta_proj = F.linear(_delta_proj, _W, bias=None)
                else:
                    # Apply non-Linear layers normally (Dropout, LayerNorm, etc.)
                    _delta_proj = _m(_delta_proj)
                    
        elif isinstance(to_out, torch.nn.Sequential):
            # Handle Sequential container (e.g., Linear + Dropout)
            _first = to_out[0]
            if isinstance(_first, torch.nn.Linear):
                # Apply Linear layer WITHOUT bias
                _W = _first.weight
                _delta_proj = F.linear(delta_concat, _W, bias=None)
                # Apply remaining layers (Dropout, etc.)
                for _m in list(to_out)[1:]:
                    _delta_proj = _m(_delta_proj)
            else:
                # First layer is not Linear, apply normally
                _delta_proj = to_out(delta_concat)
                
        elif isinstance(to_out, torch.nn.Linear):
            # Single Linear layer - apply WITHOUT bias
            _W = to_out.weight
            _delta_proj = F.linear(delta_concat, _W, bias=None)
            
        else:
            # Unknown type, apply normally
            _delta_proj = to_out(delta_concat)
        
        return _delta_proj

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        """
        Process attention using TRUE DELEGATION with BIAS-FREE projection:

        

        """
        
        # 1. DELEGATE ENTIRELY TO ORIGINAL PROCESSOR
        orig_out = self.original_processor(
            attn, 
            hidden_states, 
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 2. GET EXTERNAL HEADS FOR THIS LAYER
        # Infer dimensions from hidden_states
        B, N, _ = hidden_states.shape
        H = self.num_heads
        d_h = self.head_dim
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if self.first_call:
            heads_info = f"target heads {self.target_heads}" if self.target_heads else "all heads"
            print(f"  Layer {self.layer_idx}: Building external head delta with B={B}, N={N}, H={H}, d_h={d_h} ({heads_info})")
            print(f"  Layer {self.layer_idx}: Using BIAS-FREE projection for delta")
            self.first_call = False  # FIXED: Set to False after first call
        
        # Get external heads: [H, N, d_h]
        external_heads = self.external_heads_module.get_all_heads_for_layer(
            self.layer_idx, N, device, dtype, target_heads=self.target_heads
        )
        
        # Expand for batch dimension: [B, H, N, d_h]
        external_heads = external_heads.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Reshape to concatenated form: [B, N, H*d_h]
        delta_concat = external_heads.transpose(1, 2).reshape(B, N, H * d_h)
        
        # 3. PROJECT EXTERNAL HEADS THROUGH to_out WITHOUT BIAS
        delta_proj = self._apply_to_out_bias_free(attn, delta_concat)
        
        # 4. ADD SCALED PROJECTED DELTA TO ORIGINAL OUTPUT
        return orig_out + self.coefficient * delta_proj


# ============================================================================
# SETUP FUNCTIONS - FIXED VERSION
# ============================================================================
def load_external_heads(checkpoint_path, target_layers):
    """
    Load trained external heads from checkpoint.

    
    Args:
        checkpoint_path: Path to external_heads_full.pt file
        target_layers: List of layer indices to use
        
    Returns:
        LoadedExternalHeads object
    """
    # FIXED: Check file existence
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"External heads checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Loading external heads from: {checkpoint_path}")
    raw_state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # FIXED: Normalize keys to strip module prefix
    normalized_state_dict = {}
    for k, v in raw_state_dict.items():
        normalized_key = k
        if normalized_key.startswith("external_heads."):
            normalized_key = normalized_key[len("external_heads."):]
        normalized_state_dict[normalized_key] = v
    
    # Create LoadedExternalHeads object with normalized keys
    external_heads = LoadedExternalHeads(
        state_dict=normalized_state_dict,
        target_layers=target_layers,
        num_heads=16,
        head_dim=72
    )
    
    return external_heads


def setup_external_heads_processors(pipe, target_layers, baseline_processors, 
                                    external_heads_module, coefficient, target_heads=None):
    """
    Install external heads processors on target layers.
    
    Args:
        pipe: PixArt pipeline
        target_layers: List of layer indices
        baseline_processors: Dictionary of original processors
        external_heads_module: LoadedExternalHeads object
        coefficient: Scaling coefficient for external heads
        target_heads: None (all heads) or list of specific head indices to modify
    """
    for layer_idx in target_layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        cross_attn = block.attn2  # Cross-attention layer
        
        # Get baseline processor
        original_processor = baseline_processors[layer_idx]
        
        # Create custom processor with external heads
        custom_processor = TrueDelegationExternalHeadsProcessor(
            original_processor=original_processor,
            layer_idx=layer_idx,
            attn_module=cross_attn,
            external_heads_module=external_heads_module,
            coefficient=coefficient,
            target_heads=target_heads
        )
        
        # Install custom processor
        cross_attn.set_processor(custom_processor)


def reset_to_baseline_processors(pipe, target_layers, baseline_processors):
    """
    Reset all target layers to use baseline (unmodified) processors.
    
    Args:
        pipe: PixArt pipeline
        target_layers: List of layer indices
        baseline_processors: Dictionary of original processors
    """
    for layer_idx in target_layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        cross_attn = block.attn2
        cross_attn.set_processor(baseline_processors[layer_idx])


def save_baseline_processors(pipe, target_layers):
    """
    Save the current (baseline) processors for all target layers.
    
    Args:
        pipe: PixArt pipeline
        target_layers: List of layer indices
        
    Returns:
        Dictionary mapping layer_idx to baseline processor
    """
    baseline_processors = {}
    for layer_idx in target_layers:
        block = pipe.transformer.transformer_blocks[layer_idx]
        cross_attn = block.attn2
        baseline_processors[layer_idx] = cross_attn.get_processor()
    return baseline_processors


# ============================================================================
# IMAGE GENERATION
# ============================================================================
def generate_image(pipe, prompt, seed, num_inference_steps):
    """
    Generate a single image with the given parameters.
    
    Args:
        pipe: PixArt pipeline
        prompt: Text prompt
        seed: Random seed
        num_inference_steps: Number of denoising steps
        
    Returns:
        PIL Image
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=4.5,
    ).images[0]
    
    return image


# ============================================================================
# MAIN FUNCTION - FIXED VERSION
# ============================================================================
def main():
    """
    Main generation function with all critical issues fixed.
    
    FIXES:
    - Proper try-finally for processor cleanup
    - Better error handling and reporting
    """
    print("="*80)
    print("EXTERNAL HEADS GUIDED IMAGE GENERATION - FIXED VERSION")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Target Layers: {TARGET_LAYERS}")
    print(f"Target Heads: {TARGET_HEADS if TARGET_HEADS else 'All heads'}")
    print(f"Base Seed: {BASE_SEED}")
    print(f"Number of Seeds: {NUM_SEEDS}")
    print(f"Coefficients: {COEFFICIENT_LIST}")
    print(f"Total Images: {NUM_SEEDS * (1 + len(COEFFICIENT_LIST))} ({NUM_SEEDS} seeds × {1 + len(COEFFICIENT_LIST)} images)")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe.to("cuda")
    
    # Optional: Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    print(f"✓ Loaded model: {MODEL_ID}")
    
    # ========================================================================
    # LOAD EXTERNAL HEADS
    # ========================================================================
    print(f"\n{'='*80}")
    print("LOADING EXTERNAL HEADS")
    print(f"{'='*80}")
    
    external_heads = load_external_heads(EXTERNAL_HEADS_PATH, TARGET_LAYERS)
    
    # ========================================================================
    # SAVE BASELINE PROCESSORS
    # ========================================================================
    print(f"\n{'='*80}")
    print("SAVING BASELINE PROCESSORS")
    print(f"{'='*80}")
    
    baseline_processors = save_baseline_processors(pipe, TARGET_LAYERS)
    print(f"✓ Saved baseline processors for {len(TARGET_LAYERS)} layers")
    
    # ========================================================================
    # GENERATE IMAGES FOR EACH SEED
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"GENERATING IMAGES FOR {NUM_SEEDS} SEEDS")
    print(f"{'='*80}")
    
    total_images = 0
    
    # Loop over seeds
    for seed_idx in range(NUM_SEEDS):
        current_seed = BASE_SEED + seed_idx
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{current_seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"SEED {current_seed} ({seed_idx + 1}/{NUM_SEEDS})")
        print(f"{'='*80}")
        print(f"Output directory: {seed_dir}")
        
        # ====================================================================
        # GENERATE BASELINE IMAGE (coefficient = 0.0)
        # ====================================================================
        print(f"\n  Generating baseline (coefficient=0.0)...")
        try:
            baseline_image = generate_image(pipe, PROMPT, current_seed, NUM_INFERENCE_STEPS)
            baseline_filename = os.path.join(seed_dir, "baseline_coef_0.0.png")
            baseline_image.save(baseline_filename)
            print(f"  ✓ Saved: baseline_coef_0.0.png")
            total_images += 1
        except Exception as e:
            print(f"  ✗ ERROR: Generating baseline: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # ====================================================================
        # GENERATE IMAGES WITH EXTERNAL HEADS FOR EACH COEFFICIENT
        # ====================================================================
        print(f"\n  Generating images with external heads...")
        
        for coef in COEFFICIENT_LIST:
            # FIXED: Use try-finally to ensure cleanup even on errors
            try:
                # Setup external heads processors with current coefficient
                setup_external_heads_processors(
                    pipe,
                    TARGET_LAYERS,
                    baseline_processors,
                    external_heads,
                    coefficient=coef,
                    target_heads=TARGET_HEADS
                )
                
                # Generate image
                image = generate_image(pipe, PROMPT, current_seed, NUM_INFERENCE_STEPS)
                
                # Save image
                image_filename = os.path.join(seed_dir, f"image_coef_{coef}.png")
                image.save(image_filename)
                print(f"  ✓ Saved: image_coef_{coef}.png")
                total_images += 1
                
            except Exception as e:
                print(f"  ✗ ERROR: Generating image with coefficient {coef}: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                # FIXED: Always reset to baseline, even if generation failed
                try:
                    reset_to_baseline_processors(pipe, TARGET_LAYERS, baseline_processors)
                except Exception as e:
                    print(f"  ✗ ERROR: Resetting processors: {e}")
        
        print(f"\n  ✓ Completed seed {current_seed}: {total_images - (seed_idx * (1 + len(COEFFICIENT_LIST)))} images this seed")
    
    print(f"\n{'='*80}")
    print(f"✓ Generated {total_images} total images across {NUM_SEEDS} seeds")
    print(f"{'='*80}")
    
    # ========================================================================
    # CREATE SUMMARY FILE
    # ========================================================================
    print(f"\n{'='*80}")
    print("CREATING SUMMARY FILE")
    print(f"{'='*80}")
    
    summary_filename = os.path.join(OUTPUT_DIR, "generation_summary.txt")
    try:
        with open(summary_filename, 'w') as f:
            f.write("External Heads Guided Generation - FIXED VERSION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"FIXES APPLIED:\n")
            f.write(f"  1. Key normalization (strips 'external_heads.' prefix)\n")
            f.write(f"  2. Per-key sequence length handling\n")
            f.write(f"  3. Immediate device/dtype conversion\n")
            f.write(f"  4. Head dimension validation\n")
            f.write(f"  5. File existence check\n")
            f.write(f"  6. Try-finally for processor cleanup\n")
            f.write(f"  7. first_call flag properly reset\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Model: {MODEL_ID}\n")
            f.write(f"  Prompt: '{PROMPT}'\n")
            f.write(f"  Base Seed: {BASE_SEED}\n")
            f.write(f"  Number of Seeds: {NUM_SEEDS}\n")
            f.write(f"  Seeds Used: {BASE_SEED} to {BASE_SEED + NUM_SEEDS - 1}\n")
            f.write(f"  Target Layers: {TARGET_LAYERS}\n")
            f.write(f"  Target Heads: {TARGET_HEADS if TARGET_HEADS else 'All heads'}\n")
            f.write(f"  Inference Steps: {NUM_INFERENCE_STEPS}\n")
            f.write(f"  Coefficients Tested: {COEFFICIENT_LIST}\n")
            f.write(f"  External Heads Path: {EXTERNAL_HEADS_PATH}\n\n")
            f.write(f"Approach (TRUE DELEGATION + BIAS-FREE):\n")
            f.write(f"  1. Load trained external heads from checkpoint\n")
            f.write(f"  2. For each target layer:\n")
            f.write(f"     - Call original_processor(attn, ...) → get unmodified output\n")
            f.write(f"     - Get external heads for this layer [H, N, d_h]\n")
            f.write(f"     - Reshape to [B, N, H*d_h] and project through to_out WITHOUT bias\n")
            f.write(f"     - Add coefficient * projected_delta to original output\n")
            f.write(f"  ✓ No attention re-implementation\n")
            f.write(f"  ✓ Bias-free projection ensures coefficient=0 matches baseline\n")
            f.write(f"  ✓ Inherits all model behavior (dtype, scaling, masks, etc.)\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Images: {NUM_SEEDS * (1 + len(COEFFICIENT_LIST))}\n")
            f.write(f"  Images per Seed: {1 + len(COEFFICIENT_LIST)} (1 baseline + {len(COEFFICIENT_LIST)} with external heads)\n")
            f.write(f"  Total Seeds: {NUM_SEEDS}\n\n")
            f.write(f"Directory Structure:\n")
            for seed_idx in range(NUM_SEEDS):
                seed = BASE_SEED + seed_idx
                f.write(f"  seed_{seed}/\n")
                f.write(f"    - baseline_coef_0.0.png\n")
                for coef in COEFFICIENT_LIST:
                    f.write(f"    - image_coef_{coef}.png\n")
                f.write(f"\n")
            f.write(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✓ Saved summary: {summary_filename}")
    except Exception as e:
        print(f"✗ ERROR: Creating summary file: {e}")
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images generated: {total_images}")
    print(f"Seeds: {BASE_SEED} to {BASE_SEED + NUM_SEEDS - 1}")
    print(f"Coefficients: {COEFFICIENT_LIST}")
    print(f"✓ All done!")


if __name__ == "__main__":
    main()
