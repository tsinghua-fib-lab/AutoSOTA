import os
from typing import Dict, Optional, Union

import safetensors
import torch
from diffusers.utils import _get_model_file, logging
from safetensors import safe_open

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomAdapterMixin:
    def init_custom_adapter(self, *args, **kwargs):
        self._init_custom_adapter(*args, **kwargs)

    def _init_custom_adapter(self, *args, **kwargs):
        raise NotImplementedError

    def load_custom_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        self._load_custom_adapter(state_dict)

    def _load_custom_adapter(self, state_dict):
        raise NotImplementedError

    def save_custom_adapter(
        self,
        save_directory: Union[str, os.PathLike],
        weight_name: str,
        safe_serialization: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        if safe_serialization:

            def save_function(weights, filename):
                return safetensors.torch.save_file(
                    weights, filename, metadata={"format": "pt"}
                )

        else:
            save_function = torch.save

        # Save the model
        state_dict = self._save_custom_adapter(**kwargs)
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(
            f"Custom adapter weights saved in {os.path.join(save_directory, weight_name)}"
        )

    def _save_custom_adapter(self):
        raise NotImplementedError
