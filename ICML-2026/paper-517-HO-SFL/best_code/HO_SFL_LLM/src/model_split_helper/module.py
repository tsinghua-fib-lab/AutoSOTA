import torch
from transformers import AutoConfig
from typing import Optional


class ModelSplitHelper:
    def __init__(self):
        pass

    def split(
        self,
        model_name: str,
        split_point: int,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
    ):
        """
        Split the model into client_side_model and server_side_model.

        Args:
            model_name: Model name, e.g., "facebook/opt-125m"
            split_point: Number of transformer blocks for client side
            torch_dtype: Data type for the model
            device_map: Device mapping

        Returns:
            client_side_model, server_side_model
        """
        config = AutoConfig.from_pretrained(model_name)
        if "opt" in model_name.lower():
            from .opt_split import split_opt

            return split_opt(config, split_point, torch_dtype, device_map)
        elif "gemma" in model_name.lower():
            from .gemma3_split import split_gemma3

            return split_gemma3(config, split_point, torch_dtype, device_map)
        elif "llama" in model_name.lower():
            from .llama_split import split_llama

            return split_llama(config, split_point, torch_dtype, device_map)
        else:
            raise NotImplementedError(f"Model {model_name} not supported yet")
