"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        if self.padding_side == "right":  
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)  
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)  
        elif self.padding_side == "left":  
            # Manually pad sequences on the left  
            max_len = max(len(seq) for seq in input_ids)  
            input_ids = [torch.cat((torch.full((max_len - len(seq),), self.pad_token_id, dtype=seq.dtype), seq)) for seq in input_ids]  
            labels = [torch.cat((torch.full((max_len - len(seq),), IGNORE_INDEX, dtype=seq.dtype), seq)) for seq in labels]  
            input_ids = torch.stack(input_ids)  
            labels = torch.stack(labels)  
        else:  
            raise ValueError(f"Invalid padding_side: {self.padding_side}")  
  
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]  
  
        attention_mask = input_ids.ne(self.pad_token_id)  

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        action_masks = [instance["action_masks"] for instance in instances]
        action_masks = torch.stack(action_masks)
        # Add continuous actions
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:]) 
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


@dataclass
class PaddedCollatorForHandPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        # assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        # if all([label is not None for label in labels]):
        #     labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # else:
        #     labels = torch.zeros_like(input_ids)
        if self.padding_side == "right":  
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            if all([label is not None for label in labels]):
                labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            else:
                labels = torch.zeros_like(input_ids)
        elif self.padding_side == "left":  
            # Manually pad sequences on the left  
            max_len = max(len(seq) for seq in input_ids)  
            input_ids = [torch.cat((torch.full((max_len - len(seq),), self.pad_token_id, dtype=seq.dtype), seq)) for seq in input_ids]  
            input_ids = torch.stack(input_ids)  
            if all([label is not None for label in labels]):
                labels = [torch.cat((torch.full((max_len - len(seq),), IGNORE_INDEX, dtype=seq.dtype), seq)) for seq in labels]  
                labels = torch.stack(labels)
            else:
                labels = torch.zeros_like(input_ids)
        else:  
            raise ValueError(f"Invalid padding_side: {self.padding_side}") 

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        if "action_masks" in instances[0]:
            action_masks = [instance["action_masks"] for instance in instances]
            action_masks = torch.stack(action_masks)
        else:
            action_masks = None
        # Add continuous actions 
        if "current_state_mask" in instances[0]:
            current_state_mask = [instance["current_state_mask"] for instance in instances]
            current_state_mask = torch.stack(current_state_mask)
            current_state = [instance["current_state"] for instance in instances]
            current_state = torch.stack(current_state)
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:]) 
        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
            current_state_mask=current_state_mask,
            current_state=current_state,
        )

        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output