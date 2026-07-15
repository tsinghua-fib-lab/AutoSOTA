import os
import json
from typing import Dict, Any
import torch

import contextlib
import transformer_lens.loading_from_pretrained as tl_loading
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig
from sae_lens.saes.sae import SAEMetadata

def load_hooked_transformer_offline(
    model_name: str,
    model_path: str,
    device: str = "cuda"
) -> HookedTransformer:
    """
    Load the hooked transformer model from the local model path.
    """

    print(f">>> Loading: {model_name} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=True
    )

    @contextlib.contextmanager
    def force_local_config_loading(target_local_path):
        original_from_pretrained = tl_loading.AutoConfig.from_pretrained
        def patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
            print(f"Interceptor active: Redirecting config load from '{pretrained_model_name_or_path}' to '{target_local_path}'")
            return original_from_pretrained(target_local_path, **kwargs)

        tl_loading.AutoConfig.from_pretrained = patched_from_pretrained

        try:
            yield
        finally:
            tl_loading.AutoConfig.from_pretrained = original_from_pretrained
            print("Config loader restored to original state.")

    print(f">>> Loading HookedTransformer with forced local config...")
    with force_local_config_loading(model_path):
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            local_files_only=True 
        )

        print(">>> Model loaded successfully.")
    
    del hf_model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(">>> hf_model released from memory.")
    
    return tokenizer, model


def load_custom_batch_topk_as_jumprelu(
    model_name: str,
    sae_id: str,
    sae_path: str,
    layer: int,
    device: str = "cuda",
    dtype: str = "bfloat16"
) -> JumpReLUSAE:
    """
    Load the original weight BatchTopK model, and convert it to the native JumpReLU instance of SAE Lens.
    """
    
    print(f">>> Loading: SAE {sae_id} from {sae_path}...")

    config_path = sae_path.replace("ae.pt", "config.json")
    with open(config_path) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    assert model_name in config["trainer"]["lm_name"]

    k = config["trainer"]["k"]

    pt_params = torch.load(sae_path, map_location="cpu")
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
           "threshold": "threshold",
    }
    
    renamed_params: Dict[str, Any] = {
        (key_mapping[k] if k in key_mapping else k): v
        for k, v in pt_params.items()
    }
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T
    print("Renamed keys in state_dict:", renamed_params.keys())
            
    metadata = SAEMetadata(
        model_name=model_name,
        hook_layer=layer
    )
            
    d_in=renamed_params["b_dec"].shape[0]
    d_sae=renamed_params["b_enc"].shape[0]

    if renamed_params['threshold'] > 0:
        sae_cfg = JumpReLUSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            device=device,
            dtype=dtype,
            apply_b_dec_to_input=True,
            metadata=metadata
        )
        sae = JumpReLUSAE(sae_cfg)

        if "threshold" in renamed_params:
            current_threshold = renamed_params["threshold"]
            renamed_params.pop("k")
            if current_threshold.dim() == 0:
                print(f"Converting scalar threshold {current_threshold.item()} to vector...")
                renamed_params["threshold"] = current_threshold.expand(sae_cfg.d_sae)
            
            elif current_threshold.dim() == 1 and current_threshold.shape[0] == 1:
                renamed_params["threshold"] = current_threshold.expand(sae_cfg.d_sae)

        sae.load_state_dict(renamed_params)
        sae.to(device, dtype=getattr(torch, dtype))
    else:
        raise ValueError("The provided model does not use JumpReLU activation (threshold <= 0).")
            
    d_sae, d_in = sae.W_dec.data.shape
    assert d_sae >= d_in

    print(">>> SAE loaded successfully.")
    for key, value in vars(sae.cfg).items():
        print(f"{key}: {value}")
    return sae


def load_decoder_matrix(file_path):
    """
    Load the SAE checkpoint and extract the decoder matrix.
    """
    print(f">>> Loading checkpoint from: {file_path}")
    
    # 1. Load state_dict (map to CPU to avoid memory overflow)
    try:
        state_dict = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # 2. Determine the Decoder key name
    target_key = None
    if "decoder.weight" in state_dict:
        target_key = "decoder.weight"
        print(">>> Found key: 'decoder.weight' (Standard Format)")
    elif "W_dec" in state_dict:
        target_key = "W_dec"
        print(">>> Found key: 'W_dec' (SAE Lens Format)")
    else:
        print(f"Error: Could not find decoder weight. Available keys: {list(state_dict.keys())}")
        return None

    # 3. Extract and process the matrix
    W_dec = state_dict[target_key]
    
    # 4. Execute the transpose logic (re-implement the logic in the reference code: renamed_params["W_dec"] = renamed_params["W_dec"].T)
    # In the original code, decoder.weight is usually [d_model, d_sae] (Linear layer default), and after transpose, it becomes [d_sae, d_model]
    if target_key == "decoder.weight":
        print(">>> Applying transpose to match SAE Lens format (W.T)...")
        W_dec = W_dec.T
    
    return W_dec