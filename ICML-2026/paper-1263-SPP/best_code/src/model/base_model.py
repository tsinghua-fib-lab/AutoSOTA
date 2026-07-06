"""
Base model wrapper module
Supports loading, quantization, forward pass, activation extraction, and more
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple, Any
import logging
from pathlib import Path

from ..utils import get_device, get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module):
    """
    Base model wrapper class
    Supports quantization, mixed precision, mask application, and more
    """

    def __init__(
            self,
            model_name: str,
            quantization: str = "none",
            mixed_precision: bool = False,
            torch_dtype: str = "float16",
            device: Optional[torch.device] = None,
            **kwargs
    ):
        """
        Initialize the base model

        Args:
            model_name: model name or path
            quantization: quantization method (none, int4, int8, fp8)
            mixed_precision: whether to use mixed precision
            torch_dtype: torch data type (float16, float32, bfloat16)
            device: compute device
            **kwargs: additional arguments
        """
        super().__init__()

        self.model_name = model_name
        self.quantization = quantization
        # Flag: whether quantization was already completed during the from_pretrained load stage
        self._quantization_in_load = False
        self.mixed_precision = mixed_precision
        self.device = device or get_device()
        
        # Key fix: explicitly set the current CUDA device before loading the model
        # This ensures all CUDA operations (including transformers' internal operations) run on the correct GPU
        if self.device.type == 'cuda':
            # Extract the index from device; default to 0 if not set
            device_idx = self.device.index if self.device.index is not None else 0
            # When CUDA_VISIBLE_DEVICES is set, the device index gets remapped
            # For example: with CUDA_VISIBLE_DEVICES=3, PyTorch only sees 1 GPU (index 0)
            # So we need to check whether the index exceeds the visible device range
            if device_idx >= torch.cuda.device_count():
                device_idx = 0
                logger.warning(f"Device index {self.device.index} exceeds the visible device range, using GPU 0")
            if torch.cuda.device_count() > 0:
                torch.cuda.set_device(device_idx)
                logger.info(f"Set current CUDA device to: cuda:{device_idx}")
            else:
                logger.warning("No available CUDA device")

        # Set the data type
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Quantization method: {quantization}")
        logger.info(f"Mixed precision: {mixed_precision}")
        logger.info(f"Data type: {torch_dtype}")

        # Check whether this is a LoRA adapter path (must be checked before loading the tokenizer)
        model_path = Path(model_name)
        is_lora_path = (model_path.exists() and 
                      ((model_path / "adapter_model.safetensors").exists() or 
                       (model_path / "adapter_config.json").exists()))
        
        # Determine the base model name (for LoRA, it needs to be read from adapter_config.json)
        # If it is a local path and a tokenizer file exists, use it directly; otherwise use the default model name
        base_model_name_for_tokenizer = model_name
        if model_path.exists():
            # Check whether a tokenizer file is present
            has_tokenizer = (
                (model_path / "tokenizer.json").exists() or
                (model_path / "tokenizer_config.json").exists() or
                (model_path / "vocab.json").exists()
            )
            if not has_tokenizer:
                # Local path but no tokenizer file, use the default model name
                base_model_name_for_tokenizer = 'Qwen/Qwen2.5-7B-Instruct'
                logger.info(f"Local model path {model_name} has no tokenizer file, using default tokenizer: {base_model_name_for_tokenizer}")
        
        if is_lora_path:
            try:
                import json
                adapter_config_path = model_path / "adapter_config.json"
                if adapter_config_path.exists():
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path')
                        # If base_model_name_or_path is None or empty, use the default value
                        if base_model_name and base_model_name.strip():
                            base_model_name_for_tokenizer = base_model_name
                        else:
                            base_model_name_for_tokenizer = 'Qwen/Qwen2.5-7B-Instruct'
                            logger.warning(f"⚠️  base_model_name_or_path in adapter_config.json is empty, using default model: {base_model_name_for_tokenizer}")
            except Exception as e:
                logger.warning(f"⚠️  Unable to read adapter_config.json, using default model name: {e}")
                base_model_name_for_tokenizer = 'Qwen/Qwen2.5-7B-Instruct'

        # Ensure base_model_name_for_tokenizer is not None
        if not base_model_name_for_tokenizer or base_model_name_for_tokenizer == 'None':
            base_model_name_for_tokenizer = 'Qwen/Qwen2.5-7B-Instruct'
            logger.warning(f"⚠️  Base model name is empty, using default model: {base_model_name_for_tokenizer}")

        # Load the tokenizer (using the base model name)
        # If it is a local path, use it directly; otherwise use the HuggingFace model name
        if Path(base_model_name_for_tokenizer).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(Path(base_model_name_for_tokenizer)), trust_remote_code=True, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_for_tokenizer, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        self._load_model(**kwargs)

        # Apply quantization
        # Only run the post-load quantization logic if quantization was not already completed during the load stage
        if quantization != "none" and not getattr(self, "_quantization_in_load", False):
            self._apply_quantization(quantization)

        # Set up mixed precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.eval()  # Evaluation mode by default

    def _load_model(self, **kwargs):
        """Load the model"""
        try:
            # Memory optimization config for CPU environments
            import torch
            from pathlib import Path
            device = self.device

            # Filter out arguments not supported by from_pretrained
            model_kwargs = {k: v for k, v in kwargs.items() if k != 'device_idx'}

            # Check the model path type
            model_path = Path(self.model_name)

            # Check for a complete model first (if both a complete model and LoRA exist, prefer the complete model)
            # Markers of a complete model: config.json exists and model weight files exist
            is_complete_model = False
            if model_path.exists():
                config_file = model_path / "config.json"
                has_model_weights = (
                    (model_path / "model.safetensors").exists() or
                    (model_path / "pytorch_model.bin").exists() or
                    (model_path / "model.safetensors.index.json").exists() or
                    any((model_path / f).exists() for f in ["model-00001-of-00002.safetensors", "pytorch_model-00001-of-00002.bin"])
                )
                if config_file.exists() and has_model_weights:
                    is_complete_model = True
                    logger.info(f"Detected complete model path: {model_path}")

            # If it is not a complete model, check whether it is a LoRA adapter path
            is_lora_path = False
            if not is_complete_model and model_path.exists():
                is_lora_path = (
                    (model_path / "adapter_model.safetensors").exists() or
                    (model_path / "adapter_config.json").exists()
                )
                if is_lora_path:
                    logger.info(f"Detected LoRA adapter path: {model_path}")

            if is_lora_path:
                # Load the LoRA adapter
                logger.info(f"Detected LoRA adapter path: {model_path}")
                # First we need to find the base model path (read from adapter_config.json)
                try:
                    import json
                    adapter_config_path = model_path / "adapter_config.json"
                    if adapter_config_path.exists():
                        with open(adapter_config_path, 'r') as f:
                            adapter_config = json.load(f)
                            base_model_name = adapter_config.get('base_model_name_or_path')
                            # If base_model_name_or_path is None or empty, use the default value
                            if not base_model_name or base_model_name.strip() == '' or base_model_name == 'None':
                                base_model_name = 'Qwen/Qwen2.5-7B-Instruct'
                                logger.warning(f"⚠️  base_model_name_or_path in adapter_config.json is empty, using default model: {base_model_name}")
                    else:
                        base_model_name = 'Qwen/Qwen2.5-7B-Instruct'
                        logger.warning(f"⚠️  adapter_config.json not found, using default model: {base_model_name}")

                    logger.info(f"Loading base model: {base_model_name}")
                    # Load the base model first
                    if device.type == 'cpu':
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            **model_kwargs
                        )
                        base_model = base_model.to(device)
                    else:
                        # GPU environment: explicitly specify the device to avoid device mismatch caused by device_map="auto"
                        # Ensure the current CUDA device is correct again before loading
                        if device.type == 'cuda':
                            # When CUDA_VISIBLE_DEVICES is set, the device index gets remapped
                            device_idx = device.index if device.index is not None else 0
                            if device_idx >= torch.cuda.device_count():
                                device_idx = 0
                                logger.warning(f"Device index {device.index} exceeds the visible device range, using GPU 0")
                            torch.cuda.set_device(device_idx)
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=True,
                            device_map=None,  # Do not use device_map="auto" to avoid device mismatch
                            **model_kwargs
                        )
                        # Explicitly move the base model to the specified device
                        base_model = base_model.to(device)

                    # Load the LoRA adapter
                    from peft import PeftModel
                    # PeftModel.from_pretrained inherits the device of base_model
                    self.model = PeftModel.from_pretrained(base_model, str(model_path))
                    # Ensure the model is on the correct device again (double safeguard)
                    if device.type != 'cpu' and hasattr(self.model, 'to'):
                        model_device = next(self.model.parameters()).device
                        if model_device != device:
                            logger.warning(f"⚠️  Model device ({model_device}) differs from target device ({device}), moving to target device...")
                            self.model = self.model.to(device)
                    logger.info(f"✅ LoRA adapter loaded successfully: {model_path}, device: {device}")
                except Exception as e:
                    logger.warning(f"⚠️  LoRA loading failed, trying to load as a regular model: {e}")
                    is_lora_path = False  # Fall back to regular loading

            if not is_lora_path:
                # Regular model loading
                if device.type == 'cpu':
                    logger.info("CPU environment: loading the model with a memory optimization strategy")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,  # Reduce CPU memory usage
                        **model_kwargs
                    )
                    # Explicitly move to CPU (it is already on CPU, but to be sure)
                    self.model = self.model.to(device)
                else:
                    # GPU environment: explicitly specify the device to avoid device mismatch caused by device_map="auto"
                    # Ensure the current CUDA device is correct again before loading
                    if device.type == 'cuda':
                        # When CUDA_VISIBLE_DEVICES is set, the device index gets remapped
                        device_idx = device.index if device.index is not None else 0
                        if device_idx >= torch.cuda.device_count():
                            device_idx = 0
                            logger.warning(f"Device index {device.index} exceeds the visible device range, using GPU 0")
                        torch.cuda.set_device(device_idx)
                    # If using bitsandbytes quantization, let from_pretrained load directly onto the target GPU according to the quantization config
                    # This avoids the extra .to(device) copy whose peak memory could cause OOM.
                    import os
                    if device.type == 'cuda' and "CUDA_VISIBLE_DEVICES" in os.environ:
                        actual_device = torch.device('cuda:0')
                    else:
                        actual_device = device

                    quantization_config = None
                    device_map_for_load = None
                    if self.quantization != "none":
                        try:
                            from transformers import BitsAndBytesConfig
                            if self.quantization == "int4":
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                )
                            elif self.quantization == "int8":
                                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        except Exception as e:
                            logger.warning(f"bitsandbytes quantization config failed (will fall back to loading without quantization): {e}")
                            quantization_config = None

                    if quantization_config is not None:
                        device_map_for_load = {"": actual_device}
                        self._quantization_in_load = True

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                        device_map=device_map_for_load if device_map_for_load is not None else None,
                        quantization_config=quantization_config,
                        **model_kwargs
                    )

                    # For non-quantized loading, still need to explicitly move to the target GPU
                    if quantization_config is None:
                        self.model = self.model.to(actual_device)
            # Get the number of parameters
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded successfully: {num_params:,} parameters")
            except:
                logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _apply_quantization(self, quantization: str):
        """Apply quantization"""
        from .quantization import apply_quantization
        self.model = apply_quantization(self.model, quantization)
        logger.info(f"Quantization applied successfully: {quantization}")

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            mask: Optional[Dict[int, List[int]]] = None,
            output_activations: bool = False,
            layer_indices: Optional[List[int]] = None,
            extract_post_attn_residual: bool = True,
            retain_grad: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass

        Args:
            inputs: input dictionary (input_ids, attention_mask, etc.)
            mask: mask dictionary {layer_idx: [retained head/channel indices]}
            output_activations: whether to output activations
            layer_indices: indices of layers from which to extract activations
            extract_post_attn_residual: whether to extract the post-attention residual
            retain_grad: whether to retain gradients (True for PPD training, False for probe training)

        Returns:
            output dictionary containing logits, hidden_states, etc.
        """
        # Apply the mask (if any)
        if mask is not None:
            self._apply_mask_hooks(mask)

        # Hooks for extracting activations
        activations = {}
        hooks = []
        if output_activations and layer_indices:
            hooks = self._register_activation_hooks(
                layer_indices,
                activations,
                extract_post_attn_residual=extract_post_attn_residual,
                retain_grad=retain_grad
            )
            # If hooks is empty, activation hooks could not be registered; log a warning
            if not hooks and not hasattr(self, '_activation_hook_failed_warning'):
                logger.warning("⚠️  Unable to register activation hooks; activations cannot be extracted. This may be a model structure detection issue.")
                self._activation_hook_failed_warning = True

        # Ensure all inputs are on the device where the model resides
        # Get the model device (from the first parameter, to ensure accuracy)
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            # If there are no parameters, try to get it from embed_tokens
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                model_device = next(self.model.model.embed_tokens.parameters()).device
            else:
                # Use self.device instead of hardcoding cuda:0, to stay consistent with CUDA_VISIBLE_DEVICES
                model_device = self.device if torch.cuda.is_available() else torch.device('cpu')

        # Force all input tensors to move to the model device
        inputs_on_device = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Ensure the tensor is on the correct device; move it if not
                if v.device != model_device:
                    inputs_on_device[k] = v.to(model_device, non_blocking=True)
                else:
                    inputs_on_device[k] = v
            else:
                inputs_on_device[k] = v

        # Forward pass
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.mixed_precision):
            outputs = self.model(**inputs_on_device)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Remove mask hooks
        if mask is not None:
            self._remove_mask_hooks()

        result = {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        }

        if output_activations:
            result['activations'] = activations

        return result

    def _apply_mask_hooks(self, mask: Dict[int, List[int]]):
        """
        Apply mask hooks to gate attention heads and FFN channels

        Apply masks during the forward pass via a hook mechanism:
        - For attention heads: zero out the attention outputs of non-retained heads
        - For FFN channels: zero out the outputs of non-retained channels

        Supports multiple model architectures:
        - Llama/Qwen: model.model.layers[i]
        - GPT-2: model.transformer.h[i]
        - GPT-J/CodeGen: model.transformer.h[i]
        """
        self.mask_hooks = []
        self.current_mask = mask
        # Used to track the total overhead time of hooks (subtracted from inference_time)
        self.mask_hooks_time = 0.0

        # Handle LoRA-wrapped models (PeftModel)
        actual_model = self.model
        if hasattr(self.model, 'base_model'):
            # PeftModel structure: model.base_model.model.model.layers
            if hasattr(self.model.base_model, 'model'):
                if hasattr(self.model.base_model.model, 'model'):
                    actual_model = self.model.base_model.model.model
                    logger.debug("Detected PeftModel structure, using base_model.model.model")
                else:
                    actual_model = self.model.base_model.model
                    logger.debug("Detected PeftModel structure, using base_model.model")
            elif hasattr(self.model.base_model, 'layers'):
                actual_model = self.model.base_model
                logger.debug("Detected PeftModel structure, using base_model")

        # Detect the model structure and get the list of layers
        layers = None
        model_type = None

        # 1. Llama/Qwen/Bloom structure: model.model.layers or model.layers
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            layers = actual_model.model.layers
            model_type = 'llama'
            logger.debug("Detected Llama/Qwen structure")
        elif hasattr(actual_model, 'layers'):
            # If actual_model is itself a Qwen2Model, access layers directly
            layers = actual_model.layers
            model_type = 'llama'
            logger.debug("Detected Llama/Qwen structure (accessing layers directly)")

        # 2. GPT-2/GPT-J structure: model.transformer.h
        elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            layers = actual_model.transformer.h
            model_type = 'gpt2'
            logger.debug("Detected GPT-2/GPT-J structure")

        # 3. GPT-NeoX/Opt structure: model.gpt_neox.layers or model.model.decoder.layers
        elif hasattr(actual_model, 'gpt_neox') and hasattr(actual_model.gpt_neox, 'layers'):
            layers = actual_model.gpt_neox.layers
            model_type = 'gpt_neox'
            logger.debug("Detected GPT-NeoX structure")
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'decoder'):
            if hasattr(actual_model.model.decoder, 'layers'):
                layers = actual_model.model.decoder.layers
                model_type = 'opt'
                logger.debug("Detected OPT structure")

        if layers is None:
            logger.warning("Unable to recognize the model structure, skipping mask application")
            return

        def create_attention_mask_hook(layer_idx, retained_heads, layer_module):
            """Create an attention mask hook"""
            if layer_idx not in mask or len(retained_heads) == 0:
                return None

            # Get the name of the attention module for this layer
            attn_module_name = None
            if model_type == 'llama':
                # Llama: layer.self_attn
                if hasattr(layer_module, 'self_attn'):
                    attn_module_name = 'self_attn'
            elif model_type == 'gpt2':
                # GPT-2: layer.attn
                if hasattr(layer_module, 'attn'):
                    attn_module_name = 'attn'

            if attn_module_name is None:
                logger.warning(f"Attention module not found in layer {layer_idx}, skipping mask application")
                return None

            def hook(module, input, output):
                """
                Apply the head mask to the attention output

                Attention output formats for different models:
                - Llama/Qwen: (hidden_states, attention_weights, ...)
                - GPT-2: (hidden_states, attention_weights, ...) or just hidden_states
                """
                import time
                hook_start_time = time.time()  # Record the hook start time

                # Get the model config (number of heads)
                config = self.model.config if hasattr(self.model, 'config') else None
                if config:
                    num_heads = getattr(config, 'num_attention_heads', None) or \
                               getattr(config, 'n_head', None) or \
                               getattr(config, 'num_heads', None) or 12
                else:
                    num_heads = 12  # GPT-2 defaults to 12 heads

                # Process the output
                if isinstance(output, tuple):
                    hidden_states = output[0]  # (batch, seq_len, hidden_dim)
                    other_outputs = output[1:] if len(output) > 1 else ()
                else:
                    hidden_states = output
                    other_outputs = ()

                if len(retained_heads) >= num_heads:
                    # All heads are retained, no mask needed
                    hook_time = time.time() - hook_start_time
                    self.mask_hooks_time += hook_time  # Record the time even when no mask is needed (check overhead)
                    return output

                # Get hidden_dim and compute head_dim
                hidden_dim = hidden_states.shape[-1]
                head_dim = hidden_dim // num_heads

                if head_dim == 0:
                    logger.warning(f"Unable to compute head_dim, skipping mask")
                    hook_time = time.time() - hook_start_time
                    self.mask_hooks_time += hook_time
                    return output

                # Create the head mask
                head_mask = torch.zeros(num_heads, dtype=hidden_states.dtype,
                                       device=hidden_states.device)
                for head_idx in retained_heads:
                    if head_idx < num_heads:
                        head_mask[head_idx] = 1.0

                # Reshape hidden_states: (batch, seq_len, num_heads, head_dim)
                batch_size, seq_len, _ = hidden_states.shape
                try:
                    hidden_states_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)

                    # Apply the head mask: (1, 1, num_heads, 1)
                    head_mask_expanded = head_mask.view(1, 1, num_heads, 1)
                    masked_hidden = hidden_states_reshaped * head_mask_expanded

                    # Reshape back to: (batch, seq_len, hidden_dim)
                    masked_hidden = masked_hidden.view(batch_size, seq_len, -1)

                    hook_time = time.time() - hook_start_time
                    self.mask_hooks_time += hook_time  # Accumulate hook overhead time

                    # Return the masked output
                    if isinstance(output, tuple):
                        return (masked_hidden,) + other_outputs
                    else:
                        return masked_hidden
                except Exception as e:
                    logger.warning(f"Unable to reshape hidden_states to apply mask: {e}, skipping mask")
                    hook_time = time.time() - hook_start_time
                    self.mask_hooks_time += hook_time
                    return output

            return hook

        def create_ffn_mask_hook(layer_idx, retained_channels):
            """Create an FFN mask hook"""
            if layer_idx not in mask:
                return None

            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output

                # Apply the channel mask to the FFN output
                # Assume output is (batch, seq, hidden_dim)
                if len(output_tensor.shape) == 3:
                    hidden_dim = output_tensor.shape[-1]

                    if len(retained_channels) < hidden_dim:
                        # Create the channel mask
                        channel_mask = torch.zeros(hidden_dim, dtype=output_tensor.dtype,
                                                 device=output_tensor.device)
                        for ch_idx in retained_channels:
                            if ch_idx < hidden_dim:
                                channel_mask[ch_idx] = 1.0

                        # Apply the mask
                        masked_output = output_tensor * channel_mask.unsqueeze(0).unsqueeze(0)

                        if isinstance(output, tuple):
                            return (masked_output,) + output[1:]
                        else:
                            return masked_output

                return output

            return hook

        # Register hooks for each layer
        for layer_idx, retained in mask.items():
            if 0 <= layer_idx < len(layers):
                layer = layers[layer_idx]

                # Register a hook for the attention layer
                attn_module = None
                if model_type == 'llama':
                    # Llama/Qwen: layer.self_attn
                    if hasattr(layer, 'self_attn'):
                        attn_module = layer.self_attn
                elif model_type == 'gpt2':
                    # GPT-2: layer.attn
                    if hasattr(layer, 'attn'):
                        attn_module = layer.attn

                if attn_module is not None:
                    hook = create_attention_mask_hook(layer_idx, retained, layer)
                    if hook:
                        registered_hook = attn_module.register_forward_hook(hook)
                        self.mask_hooks.append(registered_hook)
                        logger.debug(f"Layer {layer_idx}: attention mask hook registered")

                # Register a hook for the FFN layer (only when mask_type is ffn_channels)
                # Simplified here; the FFN mask feature can be extended later

        logger.info(f"Registered mask hooks for {len([k for k in mask.keys() if 0 <= k < len(layers)])} layers")

    def _remove_mask_hooks(self):
        """Remove mask hooks"""
        if hasattr(self, 'mask_hooks'):
            for hook in self.mask_hooks:
                hook.remove()
            self.mask_hooks = []
        if hasattr(self, 'current_mask'):
            del self.current_mask
        # Reset the hooks time counter
        if hasattr(self, 'mask_hooks_time'):
            self.mask_hooks_time = 0.0
        # Reset the hooks time counter
        if hasattr(self, 'mask_hooks_time'):
            self.mask_hooks_time = 0.0

    def _register_activation_hooks(
            self,
            layer_indices: List[int],
            activations: Dict[int, torch.Tensor],
            extract_post_attn_residual: bool = True,
            retain_grad: bool = False
    ) -> List:
        """
        Register activation extraction hooks

        Args:
            layer_indices: list of layer indices
            activations: dictionary storing the activations
            extract_post_attn_residual: whether to extract the post-attention residual (True) or the entire layer output (False)
            retain_grad: whether to retain gradients (True for PPD training, False for probe training)
        """
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    raw = output[0]
                else:
                    raw = output
                # Cast to float32 to prevent NaN from fp16 overflow in deep layers
                if retain_grad:
                    activations[layer_idx] = raw.float()
                else:
                    activations[layer_idx] = raw.detach().float()

            return hook

        # Dictionary used to store pre-attention states (class level, to avoid GC)
        pre_attn_states = {}

        def make_post_attn_hook(layer_idx, layer_module):
            """
            Create a post-attention residual extraction hook
            In the Llama/Qwen structure, we need to hook after the output of self_attn
            """
            def attn_pre_hook(module, input):
                """Save hidden_states before attention"""
                if isinstance(input, tuple) and len(input) > 0:
                    pre_attn_states[layer_idx] = input[0].clone()
                elif isinstance(input, torch.Tensor):
                    pre_attn_states[layer_idx] = input.clone()
                # If input is an empty tuple or another type, skip

            def attn_post_hook(module, input, output):
                """Extract the post-attention residual after attention"""
                # Get the hidden_states from before attention
                if layer_idx not in pre_attn_states:
                    # If the pre_hook did not run, fall back to the entire layer output
                    if isinstance(output, tuple):
                        raw = output[0]
                    else:
                        raw = output
                    if retain_grad:
                        activations[layer_idx] = raw.float()
                    else:
                        activations[layer_idx] = raw.detach().float()
                    return

                pre_hidden = pre_attn_states[layer_idx].float()

                # Get the attention output
                if isinstance(output, tuple):
                    attn_output = output[0].float()  # (batch, seq_len, hidden_dim)
                else:
                    attn_output = output.float()

                # Post-attention residual = pre_attn_hidden + attn_output
                post_attn_residual = pre_hidden + attn_output
                if retain_grad:
                    activations[layer_idx] = post_attn_residual
                else:
                    activations[layer_idx] = post_attn_residual.detach()

                # Cleanup
                if layer_idx in pre_attn_states:
                    del pre_attn_states[layer_idx]

            return attn_pre_hook, attn_post_hook

        # Detect the model structure and get the list of layers
        # Handle LoRA-wrapped models (PeftModel)
        # Key fix: re-detect the model structure on every call instead of caching
        # because the model structure may change after fine-tuning (PEFT merge/unload, etc.)
        layers = None
        path_trace = []

        # Check for the LoRA-wrapped model structure first (PeftModel)
        if hasattr(self.model, 'base_model'):
            # PeftModel structure: model.base_model.model.model.layers
            # Path: PeftModel -> LoraModel -> Qwen2ForCausalLM -> Qwen2Model -> layers
            # Need to recursively find the actual model layers (handling nested PeftModel)
            # Key fix: use the get_base_model() method to obtain the underlying model, avoiding nested PeftModel issues
            current = self.model
            depth = 0
            max_depth = 10  # Increase the depth limit to handle nested PeftModel
            path_trace = [type(current).__name__]

            while depth < max_depth:
                # Prefer the get_base_model() method (PEFT-recommended approach)
                if hasattr(current, 'get_base_model'):
                    try:
                        base_model = current.get_base_model()
                        # Check that we actually obtained a different model (to avoid an infinite loop)
                        if base_model is not current and base_model is not None:
                            current = base_model
                            depth += 1
                            path_trace.append(f"get_base_model({type(current).__name__})")
                            continue
                    except Exception as e:
                        logger.debug(f"get_base_model() failed: {e}")

                # Try to find the layers attribute (highest priority)
                if hasattr(current, 'layers'):
                    layers = current.layers
                    path_trace.append('layers')
                    logger.debug(f"✅ Detected PeftModel structure, path: {' -> '.join(path_trace)}")
                    break

                # Try to access the model attribute
                if hasattr(current, 'model'):
                    next_model = current.model
                    # Avoid circular references
                    if next_model is not current:
                        current = next_model
                    depth += 1
                    path_trace.append(type(current).__name__)
                    continue

                # If base_model is a PeftModel, keep descending
                if hasattr(current, 'base_model'):
                    next_base = current.base_model
                    # Avoid circular references
                    if next_base is not current:
                        current = next_base
                    depth += 1
                    path_trace.append(f"base_model({type(current).__name__})")
                    continue

                # If nothing was found, break out of the loop
                    break

            # If still not found, try the standard paths (for backward compatibility)
            if layers is None:
                # Try the standard path starting from the current model
                try:
                    if hasattr(current, 'model') and hasattr(current.model, 'model'):
                        # base_model.model.model is Qwen2Model, which has a layers attribute directly
                        if hasattr(current.model.model, 'layers'):
                            layers = current.model.model.layers
                            logger.debug("✅ Detected PeftModel structure, using model.model.layers")
                    elif hasattr(current, 'model') and hasattr(current.model, 'layers'):
                        layers = current.model.layers
                        logger.debug("✅ Detected PeftModel structure, using model.layers")
                    elif hasattr(current, 'layers'):
                        layers = current.layers
                        logger.debug("✅ Detected PeftModel structure, using layers directly")
                except:
                    pass

        # If it is not a LoRA model, check the standard model structure
        if layers is None:
            actual_model = self.model
            # 1. Llama/Qwen structure: model.model.layers or model.layers
            if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
                layers = actual_model.model.layers
            elif hasattr(actual_model, 'layers'):
                # If actual_model is itself a Qwen2Model, access layers directly
                layers = actual_model.layers
            # 2. GPT-2 structure: model.transformer.h
            elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
                layers = actual_model.transformer.h
            # 3. GPT-NeoX structure
            elif hasattr(actual_model, 'gpt_neox') and hasattr(actual_model.gpt_neox, 'layers'):
                layers = actual_model.gpt_neox.layers
            # 4. OPT structure
            elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'decoder'):
                if hasattr(actual_model.model.decoder, 'layers'):
                    layers = actual_model.model.decoder.layers

        if layers is None:
            # Last attempt: use PEFT's get_base_model() method
            if hasattr(self.model, 'get_base_model'):
                try:
                    base_model = self.model.get_base_model()
                    # Recursively try to find layers within base_model
                    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                        layers = base_model.model.layers
                        logger.debug("✅ Found layers via get_base_model()")
                    elif hasattr(base_model, 'layers'):
                        layers = base_model.layers
                        logger.debug("✅ Found layers directly via get_base_model()")
                except Exception as e:
                    logger.debug(f"get_base_model() failed: {e}")

            if layers is None:
                # Key fix: log a warning every time, because the model structure may have changed
                logger.error(f"❌ Unable to recognize the model structure, cannot register activation hooks. Model type: {type(self.model)}")
                if hasattr(self.model, 'base_model'):
                    logger.error(f"  base_model type: {type(self.model.base_model)}")
                    if hasattr(self.model.base_model, 'model'):
                        logger.error(f"  base_model.model type: {type(self.model.base_model.model)}")
                        if hasattr(self.model.base_model.model, 'model'):
                            logger.error(f"  base_model.model.model type: {type(self.model.base_model.model.model)}")
                # Try printing the model structure for debugging
                logger.error(f"  Model attributes: {dir(self.model)[:10]}...")
                return hooks

        # Validate that layers is valid
        if not isinstance(layers, (list, torch.nn.ModuleList)):
            logger.error(f"❌ layers is not a valid list of layers, type: {type(layers)}")
            return hooks

        # Register hooks
        registered_count = 0
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                try:
                    if extract_post_attn_residual:
                        # Extract the post-attention residual: need to hook into the self_attn module
                        layer_module = layers[idx]
                        # Detect the model type and find the self_attn module
                        attn_module = None
                        if hasattr(layer_module, 'self_attn'):
                            attn_module = layer_module.self_attn
                        elif hasattr(layer_module, 'attn'):
                            attn_module = layer_module.attn

                        if attn_module is not None:
                            # Register pre and post hooks on the attention module
                            attn_pre_hook, attn_post_hook = make_post_attn_hook(idx, layer_module)
                            pre_hook_handle = attn_module.register_forward_pre_hook(attn_pre_hook)
                            post_hook_handle = attn_module.register_forward_hook(attn_post_hook)
                            hooks.extend([pre_hook_handle, post_hook_handle])
                            registered_count += 1
                            logger.debug(f"✅ Registered post-attention residual hook for layer {idx}")
                        else:
                            # Fallback: if the attention module cannot be found, use the entire layer output
                            logger.warning(f"⚠️  Attention module not found in layer {idx}, using the entire layer output")
                            hook = layers[idx].register_forward_hook(make_hook(idx))
                            hooks.append(hook)
                            registered_count += 1
                    else:
                        # Extract the entire layer output
                        hook = layers[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)
                        registered_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to register hook for layer {idx}: {e}")

        if registered_count == 0:
            logger.error(f"❌ Failed to register any activation hooks (requested {len(layer_indices)} layers, layers length {len(layers)})")
        elif registered_count < len(layer_indices):
            logger.warning(f"⚠️  Only registered hooks for {registered_count}/{len(layer_indices)} layers")
        else:
            logger.debug(f"✅ Successfully registered {registered_count} activation hooks")

        return hooks

    def get_activations(
            self,
            inputs: Dict[str, torch.Tensor],
            layer_indices: List[int],
            extract_post_attn_residual: bool = True,
            retain_grad: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        Get the activations of the specified layers

        Following the paper's design, by default extract the post-attention residual (the residual stream after attention is written back)
        This is the unified cross-section location for both probe training and inference

        Args:
            inputs: input dictionary
            layer_indices: list of layer indices
            extract_post_attn_residual: whether to extract the post-attention residual (default True)
            retain_grad: whether to retain gradients (True for PPD training, False for probe training)

        Returns:
            {layer_idx: activation_tensor}
        """
        outputs = self.forward(
            inputs,
            output_activations=True,
            layer_indices=layer_indices,
            extract_post_attn_residual=extract_post_attn_residual,
            retain_grad=retain_grad
        )
        return outputs.get('activations', {})

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            **kwargs
    ) -> str:
        """
        Generate text

        Args:
            prompt: input prompt
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling parameter
            **kwargs: additional generation arguments

        Returns:
            the generated text
        """
        # Encode the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        # Generate
        # Following the previous project's implementation exactly: pass temperature and top_p directly and let transformers handle them automatically
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )

        # Decode: return only the newly generated part (dropping the input portion)
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def get_num_layers(self) -> int:
        """Get the number of model layers"""
        # Handle LoRA-wrapped models (PeftModel)
        actual_model = self.model
        if hasattr(self.model, 'base_model'):
            # PeftModel structure: model.base_model.model.model.layers
            if hasattr(self.model.base_model, 'model'):
                if hasattr(self.model.base_model.model, 'model'):
                    actual_model = self.model.base_model.model.model
                else:
                    actual_model = self.model.base_model.model
            elif hasattr(self.model.base_model, 'layers'):
                actual_model = self.model.base_model

        # 1. Llama/Qwen structure: model.model.layers or model.layers
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return len(actual_model.model.layers)
        elif hasattr(actual_model, 'layers'):
            # If actual_model is itself a Qwen2Model, access layers directly
            return len(actual_model.layers)

        # 2. GPT-2 structure: model.transformer.h
        elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return len(actual_model.transformer.h)

        # 3. GPT-NeoX structure: model.gpt_neox.layers
        elif hasattr(actual_model, 'gpt_neox') and hasattr(actual_model.gpt_neox, 'layers'):
            return len(actual_model.gpt_neox.layers)

        # 4. OPT structure: model.model.decoder.layers
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'decoder'):
            if hasattr(actual_model.model.decoder, 'layers'):
                return len(actual_model.model.decoder.layers)

        # 5. Get it from the config
        elif hasattr(actual_model, 'config'):
            config = actual_model.config
            if hasattr(config, 'num_hidden_layers'):
                return config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                return config.n_layer
            elif hasattr(config, 'num_layers'):
                return config.num_layers

        raise ValueError("Unable to determine the number of model layers, please check the model structure")

    def get_config(self):
        """Get the model config"""
        actual_model = self.model
        if hasattr(actual_model, 'base_model'):
            # PeftModel structure
            if hasattr(actual_model.base_model, 'model'):
                return actual_model.base_model.model.config
            return actual_model.base_model.config
        elif hasattr(actual_model, 'config'):
            return actual_model.config

        raise ValueError("Unable to determine the model config, please check the model structure")