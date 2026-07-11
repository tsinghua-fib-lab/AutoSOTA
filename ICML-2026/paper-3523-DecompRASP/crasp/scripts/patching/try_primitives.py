import torch
import json
from functools import partial
import copy
import re
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any
import torch.nn.functional as F

from logits_primitives_hook import lm_head_hook
from attention_primitives_hook import primitives_attention_forward

def save_wte_inputs(module, input, output, model):
    model.wte_inputs = input[0].detach()
def save_wpe_inputs(module, input, output, model):
    model.wpe_inputs = input[0].detach()

def try_primitives(hooked_model, original_model, primitives, iterable_dataset, batch_size,
                      num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp,
                      replace_attention=True, replace_logits=True):
    hooks = []
    hooks.append(hooked_model.model.transformer.wte.register_forward_hook(partial(save_wte_inputs, model=hooked_model)))
    hooks.append(hooked_model.model.transformer.wpe.register_forward_hook(partial(save_wpe_inputs, model=hooked_model)))
    if replace_attention:
        prev_attn_forward = hooked_model.attention_forward
        hooked_model.attention_forward = lambda module, layer: primitives_attention_forward(hooked_model, module, layer, primitives=primitives, tokenizer=tokenizer, converted_mlp=converted_mlp, oa_vecs=oa_vecs)
    if replace_logits:
        hooks_to_remove = [h for h in hooked_model.hooks if h.id in h.__getstate__()[0] and h.__getstate__()[0][h.id] == hooked_model.lm_head_hook]
        assert len(hooks_to_remove) == 1, hooks_to_remove
        hooks_to_remove[0].remove()
        hooks.append(hooked_model.model.lm_head.register_forward_hook(partial(
            lm_head_hook,
            hooked_model=hooked_model, primitives=primitives, tokenizer=tokenizer, converted_mlp=converted_mlp,
            oa_vecs=oa_vecs
        )))
    inputs = []
    current_step = 0
    num_correct_items = 0  # Count of items where all predictions are correct
    sum_kl = 0
    num_match_items = 0    # Count of items that exactly match original model
    sum_task_loss = 0
    with torch.no_grad():
        for item in iterable_dataset:
            inputs.append(item)
            if len(inputs) == batch_size:
                batch = collator(inputs)
                batch = {k: v.to(hooked_model.device) for k, v in batch.items()}
                labels = batch["labels"]
                batch.pop("labels")

                masks = mask_sampler.sample_binary_masks(batch_size)
                hooked_model.input_ids = batch["input_ids"]
                hooked_model.position_ids = batch["position_ids"]
                
                result = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch)
                logits = result.logits.detach()

                assert not torch.isnan(logits).any()

                target_logits = original_model(**batch).logits.detach()

                assert not torch.isnan(target_logits).any()
                
                if not hasattr(iterable_dataset, "BCE") or not iterable_dataset.BCE:
                    sum_task_loss += F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()

                    target_shift_logits = target_logits[:, :-1].detach()
                    shift_logits = logits[:, :-1].detach()
                    shift_labels = labels[:, 1:].detach()

                    target_predictions = target_shift_logits.argmax(dim=-1)
                    predictions = shift_logits.argmax(dim=-1)

                    match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                    num_match_items += match.sum().item()
                    correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                    num_correct_items += correct.sum().item()

                    sum_kl += F.kl_div(F.log_softmax(shift_logits[shift_labels != -100], dim=-1),
                                       F.log_softmax(target_shift_logits[shift_labels != -100], dim=-1),
                                       log_target=True).item()

                else:
                    mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)
                    
                    task_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                    sum_task_loss += task_loss[mask].mean().item()
            
                    loss = F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")
                    sum_kl += loss[mask].mean().item()

                    target_predictions = (target_logits > 0).long()
                    predictions = (logits > 0).long()

                    mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                    match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                    num_match_items += match.sum().item()
                    correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                    num_correct_items += correct.sum().item()

                del batch, labels, masks, logits, target_logits, result
                if 'target_shift_logits' in locals():
                    del target_shift_logits, shift_logits, shift_labels
                if 'target_predictions' in locals():
                    del target_predictions, predictions, match, correct
                if 'mask' in locals():
                    del mask
                if 'task_loss' in locals():
                    del task_loss
                if 'loss' in locals():
                    del loss
                
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                inputs = []
                current_step += 1
                
                if current_step == num_test_step:
                    break
    
    total_items = num_test_step * batch_size
    acc_per_item = num_correct_items / total_items
    acc_match = num_match_items / total_items
    avg_kl = sum_kl / num_test_step
    avg_loss = sum_task_loss / num_test_step

    if replace_attention:
        hooked_model.attention_forward = prev_attn_forward
    for h in hooks:
        h.remove()
    if replace_logits:
        hooked_model.hooks.append(hooked_model.model.lm_head.register_forward_hook(
            hooked_model.lm_head_hook
        ))
    return acc_per_item, avg_kl, acc_match, avg_loss