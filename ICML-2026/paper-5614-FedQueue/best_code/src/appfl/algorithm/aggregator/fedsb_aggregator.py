import os
import json
import torch
import shutil
from omegaconf import DictConfig
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from typing import Union, Dict, Any, Optional
from appfl.algorithm.aggregator import BaseAggregator
from fed_sb.fed.fed_agg import (
    aggregate_models_ffa,
    aggregate_models_fedex,
    aggregate_models_fed_it,
    aggregate_models_fed_sb,
)
from fed_sb.utils.initialization_utils import find_and_initialize
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class FedSBAggregator(BaseAggregator):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.aggregator_configs = aggregator_configs
        self.logger = logger

    def get_parameters(self, **kwargs):
        raise NotImplementedError("FedSB does not support get_parameters.")

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[str, Dict]], **kwargs
    ) -> Dict:
        client_ids = list(local_models.keys())
        first_client_data = list(local_models.values())[0]

        # Determine approach once with a flag
        is_file_based = isinstance(first_client_data, str)
        is_direct_content = (
            isinstance(first_client_data, dict)
            and "adapter_weights" in first_client_data
        )

        if not (is_file_based or is_direct_content):
            raise ValueError(f"Unexpected data format: {type(first_client_data)}")

        # Extract metadata from kwargs
        aggregator_kwargs = {
            key: value[client_ids[0]] for key, value in kwargs.items()
        }  # assume all clients have the same kwargs
        agg_type = aggregator_kwargs["agg_type"]

        # Load adapter weights and apply transformations
        client_models = []
        for client_data in local_models.values():
            if is_file_based:
                adapter_path = os.path.join(client_data, "adapter_model.safetensors")
                adapter_weights = load_file(
                    adapter_path, device=self.aggregator_configs.device
                )
            else:
                adapter_weights = {
                    k: v.to(self.aggregator_configs.device)
                    for k, v in client_data["adapter_weights"].items()
                }

            new_weights = {}
            for key, value in adapter_weights.items():
                if agg_type == "fed-sb":
                    if "lora_A" in key:
                        new_key = key.replace("lora_A", "lora_A.default")
                    elif "lora_B" in key:
                        new_key = key.replace("lora_B", "lora_B.default")
                    elif "lora_latent" in key:
                        new_key = key.replace("_lora_latent", ".default_lora_latent")
                    else:
                        new_key = key
                else:
                    if "lora_A" in key:
                        new_key = key.replace("lora_A", "lora_A.default")
                    elif "lora_B" in key:
                        new_key = key.replace("lora_B", "lora_B.default")
                    else:
                        new_key = key
                new_weights[new_key] = value
            client_models.append(new_weights)

        if is_file_based:
            config_path = os.path.join(
                list(local_models.values())[0], "adapter_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path) as f:
                    adapter_config = json.load(f)
            else:
                adapter_config = {}
        else:
            adapter_config = list(local_models.values())[0].get("adapter_config", {})

        if agg_type == "fed-sb":
            global_model, tokenizer = self.load_model_with_lora_sb(
                aggregator_kwargs["model_name"],
                client_models[0],
                adapter_config,
                aggregator_kwargs,
            )
        else:
            lora_weights_path = (
                list(local_models.values())[0] if is_file_based else None
            )
            global_model, tokenizer = self.load_model_with_lora(
                aggregator_kwargs["model_name"],
                adapter_config,
                aggregator_kwargs,
                lora_weights_path,
            )

        if agg_type == "fedex":
            global_model = aggregate_models_fedex(
                global_model, client_models, aggregator_kwargs
            )
        elif agg_type == "fed-it":
            global_model = aggregate_models_fed_it(global_model, client_models)
        elif agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)
        elif agg_type == "fed-sb":
            global_model = aggregate_models_fed_sb(global_model, client_models)

        if agg_type == "fed-sb":
            for param in global_model.parameters():
                param.data = param.data.contiguous()

        save_directory_final_model = os.path.join(
            self.aggregator_configs.global_model_dir, "final_model"
        )
        global_model.save_pretrained(save_directory_final_model)

        # Clean up files only if using file-based approach
        if is_file_based:
            for client_model in local_models.values():
                if os.path.exists(client_model):
                    shutil.rmtree(client_model)
                    self.logger.info(f"Deleted {client_model}")

        global_model = global_model.merge_and_unload()

        save_directory_merged_model = os.path.join(
            self.aggregator_configs.global_model_dir, "merged_model"
        )
        global_model.save_pretrained(save_directory_merged_model)

        save_directory_tokenizer = os.path.join(
            self.aggregator_configs.global_model_dir, "merged_model"
        )
        tokenizer.save_pretrained(save_directory_tokenizer)
        self.logger.info(f"Model saved to {save_directory_merged_model}")
        return {"merged_model_path": save_directory_merged_model}

    def load_model_with_lora_sb(
        self,
        base_model_name,
        lora_weights,
        adapter_config,
        args,
    ):
        """
        Load a base model with LoRA weights for Fed-SB (unified function)

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            lora_weights (dict): LoRA weights dictionary (for direct approach)
            adapter_config (dict): Adapter configuration dictionary
            args: Arguments containing LoRA parameters

        Returns:
            model: Combined model with LoRA weights
            tokenizer: Associated tokenizer
        """
        # 1. Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": self.aggregator_configs.device},
            torch_dtype=torch.bfloat16,
        )

        # 2. Load the tokenizer
        if "llama" in base_model_name:
            if "Llama-3" in base_model_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": self.aggregator_configs.device},
                )
            else:
                tokenizer = LlamaTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": self.aggregator_configs.device},
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": self.aggregator_configs.device},
            )

        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(**adapter_config)
        model = get_peft_model(base_model, lora_config)

        reconstr_config = self.aggregator_configs.reconstruction_configs
        adapter_name = "default"
        peft_config_dict = {adapter_name: lora_config}
        reconstr_config["svd"]["rank"] = args["lora_r"]

        find_and_initialize(
            model,
            peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type="svd",
            writer=None,
            reconstruct_config=reconstr_config,
        )

        model_state_dict = model.state_dict()
        for key in model_state_dict.keys():
            if ("lora_A" in key) or ("lora_B" in key):
                model_state_dict[key] = lora_weights[key]
        model.load_state_dict(model_state_dict)

        return model, tokenizer

    def load_model_with_lora(
        self,
        base_model_name,
        adapter_config,
        args,
    ):
        """
        Load a base model with LoRA weights (unified function for both direct config and file path)

        Args:
            base_model_name (str): Hugging Face model name or path to base model
            adapter_config (dict): Adapter configuration dictionary
            args: Arguments containing LoRA parameters

        Returns:
            model: Combined model with LoRA weights
            tokenizer: Associated tokenizer
        """
        # 1. Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": self.aggregator_configs.device},
            torch_dtype=torch.bfloat16,
        )

        if "llama" in base_model_name:
            if "Llama-3" in base_model_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": self.aggregator_configs.device},
                )
            else:
                tokenizer = LlamaTokenizer.from_pretrained(
                    base_model_name,
                    use_fast=True,
                    device_map={"": self.aggregator_configs.device},
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": self.aggregator_configs.device},
            )

        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(**adapter_config)
        model = get_peft_model(base_model, lora_config)

        return model, tokenizer
