import torch

from typing import List, Dict
# from data.pdb2feature import batch_coords2feature
from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification
# from module.esm.structure_module import (
#     EsmForMaskedLMWithStructure as EsmForMaskedLM,
#     EsmForSequenceClassificationWithStructure as EsmForSequenceClassification,
# )
from ..abstract_model import AbstractModel


class SaprotBaseModel(AbstractModel):
    """
    ESM base model. It cannot be used directly but provides model initialization for downstream tasks.
    """

    def __init__(self,
                 task: str,
                 config_path: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 use_lora: bool = False,
                 lora_config_path: str = None,
                 **kwargs):
        """
        Args:
            task: Task name. Must be one of ['classification', 'regression', 'lm', 'base']

            config_path: Path to the config file of huggingface esm model

            extra_config: Extra config for the model

            load_pretrained: Whether to load pretrained weights of base model

            freeze_backbone: Whether to freeze the backbone of the model

            use_lora: Whether to use LoRA on downstream tasks

            lora_config_path: Path to the config file of LoRA. If not None, LoRA model is for inference only.
            Otherwise, LoRA model is for training.

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ['classification', 'regression', 'lm', 'base']
        self.task = task
        self.config_path = config_path
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        super().__init__(**kwargs)

        # After all initialization done, lora technique is applied if needed
        self.use_lora = use_lora
        if use_lora:
            self._init_lora(lora_config_path)

    def _init_lora(self, lora_config_path):
        from peft import (
            PeftModelForSequenceClassification,
            get_peft_model,
            LoraConfig,
        )

        if lora_config_path:
            # Note that the model is for inference only
            self.model = PeftModelForSequenceClassification.from_pretrained(self.model, lora_config_path)
            self.model.merge_and_unload()
            print("LoRA model is initialized for inference.")

        else:
            lora_config = {
                "task_type": "SEQ_CLS",
                "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
                "modules_to_save": ["classifier"],
                "inference_mode": False,
                "lora_dropout": 0.1,
                "lora_alpha": 8,
            }

            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
            # original_module is not needed for training
            self.model.classifier.original_module = None

            print("LoRA model is initialized for training.")
            self.model.print_trainable_parameters()

        # After LoRA model is initialized, add trainable parameters to optimizer
        self.init_optimizers()

    def initialize_model(self):
        # Initialize tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(self.config_path)

        # Initialize different models according to task
        config = EsmConfig.from_pretrained(self.config_path)

        # Add extra config if needed
        if self.extra_config is None:
            self.extra_config = {}

        for k, v in self.extra_config.items():
            setattr(config, k, v)

        if self.task == 'classification':
            # Note that self.num_labels should be set in child classes
            if self.load_pretrained:
                self.model = EsmForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=self.num_labels, **self.extra_config)

            else:
                config.num_labels = self.num_labels
                self.model = EsmForSequenceClassification(config)

        elif self.task == 'regression':
            if self.load_pretrained:
                self.model = EsmForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=1, **self.extra_config)

            else:
                config.num_labels = 1
                self.model = EsmForSequenceClassification(config)

        elif self.task == 'lm':
            if self.load_pretrained:
                self.model = EsmForMaskedLM.from_pretrained(self.config_path, **self.extra_config)

            else:
                self.model = EsmForMaskedLM(config)

        elif self.task == 'base':
            if self.load_pretrained:
                self.model = EsmForMaskedLM.from_pretrained(self.config_path, **self.extra_config)

            else:
                self.model = EsmForMaskedLM(config)

            # Remove lm_head as it is not needed for PPI task
            self.model.lm_head = None

        # Freeze the backbone of the model
        if self.freeze_backbone:
            for param in self.model.esm.parameters():
                param.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}

    def get_hidden_states(self, inputs, reduction: str = None) -> list:
        """
        Get hidden representations of the model.

        Args:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        inputs["output_hidden_states"] = True
        outputs = self.model.esm(**inputs)

        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)

        repr_list = []
        hidden_states = outputs["hidden_states"][-1]
        for i, idx in enumerate(indices):
            if reduction == "mean":
                repr = hidden_states[i][1:idx].mean(dim=0)
            else:
                repr = hidden_states[i][1:idx]

            repr_list.append(repr)

        return repr_list

    # def add_bias_feature(self, inputs, coords: List[Dict]) -> torch.Tensor:
    #     """
    #     Add structure information as biases to attention map. This function is used to add structure information
    #     to the model as Evoformer does.
    #
    #     Args:
    #         inputs: A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
    #         coords: Coordinates of backbone atoms. Each element is a dictionary with keys ["N", "CA", "C", "O"].
    #
    #     Returns
    #         pair_feature: A tensor of shape [B, L, L, 407]. Here 407 is the RBF of distance(400) + angle(7).
    #     """
    #     inputs["pair_feature"] = batch_coords2feature(coords, self.model.device)
    #     return inputs

    def save_checkpoint(self, save_info: dict = None) -> None:
        """
        Rewrite this function for saving LoRA parameters
        """
        if not self.use_lora:
            return super().save_checkpoint(save_info)

        else:
            self.model.save_pretrained(self.save_path)


