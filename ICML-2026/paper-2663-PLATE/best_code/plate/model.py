from __future__ import annotations
import re
from typing import Optional
import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .config import PLATEConfig
from .layer import PLATELinear

try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    Conv1D = None


class PLATETuner(BaseTuner):
    """
    PEFT-style tuner that wraps target Linear/Conv1D modules with PLATELinear.
    
    Key differences from LoRA:
    - PLATELinear uses 3 components: B (frozen selection), A (trainable), Q_in (frozen basis)
    - Only plate_A is trainable; plate_B is a selection matrix
    - Q_in is reconstructed from saved buffers (C_candidates, U_polish)
    """
    prefix: str = "plate_"
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    
    def __init__(self, model, config, adapter_name):
        # Progress tracking for initialization
        self._init_progress = {"current": 0, "total": 0, "show": True}
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _check_target_module_exists(peft_config, key: str) -> bool | re.Match[str] | None:
        """Check if the module key matches target modules in the config."""
        return check_target_module_exists(peft_config, key)

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module):
        """Replace a module with a new PLATE module."""
        setattr(parent, child_name, new_module)
        # Move to correct device
        if hasattr(child, "weight") and isinstance(child.weight, torch.Tensor):
            device = child.weight.device
            new_module = new_module.to(device)

    def _create_and_replace(
        self,
        peft_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        parameter_name: Optional[str] = None,
    ):
        """Create and replace target module with PLATELinear."""
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Get config for this adapter
        if isinstance(peft_config, PLATEConfig):
            cfg = peft_config
        else:
            cfg_obj = self.peft_config.get(adapter_name)
            if cfg_obj is None or not isinstance(cfg_obj, PLATEConfig):
                raise ValueError(f"Config for adapter {adapter_name} not found or not PLATEConfig")
            cfg = cfg_obj

        # Check if we should replace this module and create appropriate wrapper
        if isinstance(target, nn.Linear) or (Conv1D is not None and isinstance(target, Conv1D)):
            wrapped = PLATELinear(base_layer=target)
        else:
            return
        
        # Get rank and plate_alpha for this specific module (from patterns or use global)
        r = cfg.r
        plate_alpha = cfg.plate_alpha
        col_tau = cfg.col_tau
        if cfg.rank_pattern and current_key in cfg.rank_pattern:
            r = cfg.rank_pattern[current_key]
        if cfg.alpha_pattern and current_key in cfg.alpha_pattern:
            plate_alpha = cfg.alpha_pattern[current_key]
        if cfg.col_tau_pattern and current_key in cfg.col_tau_pattern:
            col_tau = cfg.col_tau_pattern[current_key]
        
        # Update progress display
        self._init_progress["current"] += 1
        if self._init_progress["show"] and self._init_progress["total"] > 0:
            cur = self._init_progress["current"]
            tot = self._init_progress["total"]
            pct = 100 * cur / tot
            bar_len = 20
            filled = int(bar_len * cur / tot)
            bar = "█" * filled + "░" * (bar_len - filled)
            # Extract short layer name (e.g., "layer.0.q_proj" -> "q_proj")
            short_name = current_key.split(".")[-1] if "." in current_key else current_key
            print(f"\r[PLATE] Initializing adapters: {bar} {pct:5.1f}% ({cur}/{tot}) {short_name:<12}", end="", flush=True)
        
        # Initialize the adapter using update_layer
        wrapped.update_layer(
            adapter_name=adapter_name,
            r=r,
            plate_alpha=plate_alpha,
            plate_dropout=cfg.plate_dropout,
            col_tau=col_tau,
            max_rank=cfg.max_rank,
            init_lora_weights=cfg.init_lora_weights,
            inference_mode=cfg.inference_mode,
            num_hutch_probes=cfg.num_hutch_probes,
            pool_multiple=cfg.pool_multiple,
        )
        
        self._replace_module(parent, target_name, wrapped, target)

    def _count_target_modules(self, model: nn.Module, peft_config) -> int:
        """Count how many modules will be replaced (for progress bar)."""
        count = 0
        for key, module in model.named_modules():
            if self._check_target_module_exists(peft_config, key):
                if isinstance(module, nn.Linear) or (Conv1D is not None and isinstance(module, Conv1D)):
                    count += 1
        return count
    
    def inject_adapter(self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, **kwargs):
        """Override to add progress tracking during adapter injection."""
        peft_config = self.peft_config.get(adapter_name)
        
        # Count target modules for progress bar (skip for inference_mode)
        if peft_config and not peft_config.inference_mode:
            self._init_progress["total"] = self._count_target_modules(model, peft_config)
            self._init_progress["current"] = 0
            self._init_progress["show"] = self._init_progress["total"] > 0
        else:
            self._init_progress["show"] = False
        
        # Call parent implementation (pass through any extra kwargs like low_cpu_mem_usage)
        super().inject_adapter(model, adapter_name, autocast_adapter_dtype, **kwargs)
        
        # Print completion summary
        if self._init_progress["show"] and self._init_progress["total"] > 0:
            print(f"\r[PLATE] Initialized {self._init_progress['total']} adapters" + " " * 40)
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only PLATE adapter params (plate_A) as trainable."""
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        for _, m in model.named_modules():
            if isinstance(m, PLATELinear):
                for pname, pp in m.named_parameters():
                    if "plate_a" in pname.lower():
                        pp.requires_grad_(True)
    
    def enable_adapter_layers(self) -> None:
        """Enable all PLATE adapters."""
        self._set_adapter_layers(enabled=True)
    
    def disable_adapter_layers(self) -> None:
        """Disable all PLATE adapters."""
        self._set_adapter_layers(enabled=False)
    
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s).
        
        Args:
            adapter_name: Name of the adapter(s) to be activated.
            inference_mode: If True, freeze adapter parameters (requires_grad=False).
                           If False, set active adapter to trainable (requires_grad=True).
        
        Note: We override BaseTuner.set_adapter because PLATELinear inherits from
        BaseTunerLayer, not PLATETunerLayer, so the base implementation's isinstance
        check would fail.
        """
        # Handle auxiliary modules (modules_to_save) - if method exists (peft >= 0.18.0)
        if hasattr(self, 'set_auxiliary_adapters'):
            self.set_auxiliary_adapters(adapter_name, inference_mode=inference_mode)
        
        # Set adapter on all PLATE layers
        for module in self.model.modules():
            if isinstance(module, PLATELinear):
                if module.merged:
                    import warnings
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                # Try with inference_mode (newer peft), fall back to without (older peft)
                try:
                    module.set_adapter(adapter_name, inference_mode=inference_mode)
                except TypeError:
                    module.set_adapter(adapter_name)
        
        # Track active adapter - same as BaseTuner
        self.active_adapter = adapter_name
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """Prepare adapter config, auto-inferring target_modules if needed."""
        if peft_config.target_modules is None:
            if model_config["model_type"] in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                peft_config.target_modules = set(
                    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
                )
            else:
                raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config
    
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        """Unload and optionally merge adapter weights into base model."""
        from peft.utils import _get_submodules
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable
        
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            
            if isinstance(target, PLATELinear):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
        
        return self.model
    
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None
    ) -> nn.Module:
        """
        Merge the PLATE adapter weights into the base model and unload the adapter.
        
        This returns the base model with adapter weights merged in, ready to use
        as a standalone model without PEFT.
        
        Args:
            progressbar (bool): Show progress bar during merge.
            safe_merge (bool): Check for NaN values in adapter weights.
            adapter_names (Optional[list[str]]): Specific adapters to merge.
        
        Returns:
            nn.Module: The merged base model.
        
        Example:
            >>> from peft import get_peft_model
            >>> from plate import PLATEConfig
            >>> 
            >>> model = YourModel()
            >>> config = PLATEConfig(r=64, target_modules=["fc1"])
            >>> peft_model = get_peft_model(model, config)
            >>> # ... train ...
            >>> merged_model = peft_model.merge_and_unload()
            >>> # Now merged_model is a regular nn.Module with adapters merged
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar,
            safe_merge=safe_merge,
            adapter_names=adapter_names,
            merge=True
        )
    
    def unload(self) -> nn.Module:
        """
        Unload the adapter without merging, returning the original base model.
        
        Returns:
            nn.Module: The original base model without adapters.
        """
        return self._unload_and_optionally_merge(merge=False)

def get_plate_model(model: nn.Module, config: PLATEConfig) -> nn.Module:
    """
    Public entry point: patch the model and mark only adapters as trainable.
    Usage:
        from plate import PLATEConfig, get_plate_model
        model = get_plate_model(model, PLATEConfig(...))
    
    Note: PLATE does not support gradient checkpointing. If gradient checkpointing
    is enabled on the model, it will be automatically disabled.
    """
    # Check for gradient checkpointing and disable if enabled
    if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
        import warnings
        warnings.warn(
            "Gradient checkpointing is enabled but not supported by PLATE. "
            "Disabling gradient checkpointing. The Q_in projection uses a fixed buffer "
            "which breaks gradient flow during activation recomputation.",
            UserWarning,
            stacklevel=2
        )
        model.gradient_checkpointing_disable()
    
    tuner = PLATETuner(model, config, adapter_name="plate")
    tuner.inject_adapter(model, "plate")
    return model

