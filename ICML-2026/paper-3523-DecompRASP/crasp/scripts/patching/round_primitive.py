from collections import defaultdict
import os
import json
from math import isnan
import torch
import copy
from pathlib import Path
from functools import partial
import torch.nn.functional as F

from try_primitives import save_wpe_inputs, save_wte_inputs
from attention_primitives_hook import primitives_attention_forward
from logits_primitives_hook import lm_head_hook

class MatrixRounder(torch.nn.Module):
    possible_params_and_default_values = {
        "train_scalar": [False],
        "round_loss_coef": 1.,
        "scalar_loss_coef": 1.,
        "to_zero_loss_coef": 1.,
        "two_stages": [True],
        "average_over_pixels": [True],
        "to_zero_loss_penalty": "l1",
        "round_loss_penalty": "l1",
    }

    def __init__(self, indep_prod, params=None):
        super().__init__()
        self.indep_prod = torch.nn.Parameter(copy.deepcopy(indep_prod))
        self.scalar = torch.nn.Parameter(torch.tensor(1.))
        self.do_round = False

        for attr, value in MatrixRounder.possible_params_and_default_values.items():
            setattr(self, attr, value)
        if params:
            for attr in MatrixRounder.possible_params_and_default_values:
                if attr in params:
                    setattr(self, attr, params[attr])

        if not self.train_scalar:
            self.scalar.requires_grad = False
        
        if self.two_stages:
            self.stage = 1
        else:
            self.stage = None

    def set_stage(self, stage):
        self.stage = stage

    def enable_rounding(self):
        self.do_round = True
    def disable_rounding(self):
        self.do_round = False

    def get_matrix(self):
        if self.do_round:
            return self.scalar * torch.round(self.indep_prod)
        else:
            return self.scalar * self.indep_prod
    
    def get_penalty(self):
        rounded_diff = torch.round(self.indep_prod) - self.indep_prod
        if self.round_loss_penalty == "l2":
            if self.average_over_pixels:
                reg_round = self.round_loss_coef * (rounded_diff ** 2).sum()
            else:
                reg_round = self.round_loss_coef * (rounded_diff ** 2).mean()
        elif self.round_loss_penalty == "l1":
            if self.average_over_pixels:
                reg_round = self.round_loss_coef * torch.abs(rounded_diff).sum()
            else:
                reg_round = self.round_loss_coef * torch.abs(rounded_diff).mean()
        else:
            raise NotImplementedError()
        
        if self.to_zero_loss_penalty == "l2":
            if self.average_over_pixels:
                reg_to_zero = self.to_zero_loss_coef * (self.indep_prod ** 2).sum()
            else:
                reg_to_zero = self.to_zero_loss_coef * (self.indep_prod ** 2).mean()
        elif self.to_zero_loss_penalty == "l1":
            if self.average_over_pixels:
                reg_to_zero = self.to_zero_loss_coef * torch.abs(self.indep_prod).sum()
            else:
                reg_to_zero = self.to_zero_loss_coef * torch.abs(self.indep_prod).mean()
        else:
            raise NotImplementedError()
        
        if self.stage is None:
            return reg_round + reg_to_zero
        elif self.stage == 1:
            return reg_to_zero
        elif self.stage == 2:
            return reg_round
        raise NotImplementedError(f"Stage {self.stage} not defined")


def base_learning_loop(hooked_model, original_model, collator, iterable_dataset, mask_sampler, oa_vecs,
                  primitives_to_try, tokenizer, converted_mlp,
                  all_params, samplers_dict,
                  lamb = 0.05, num_steps = 100, batch_size = 120, log_interval = 10,
                  lr=0.1, save_dir=None, tuple_of_hyperparams=None, match_acc_threshold=0.95):
    hooks = []
    hooks_to_remove = [h for h in hooked_model.hooks if h.id in h.__getstate__()[0] and h.__getstate__()[0][h.id] == hooked_model.lm_head_hook]
    assert len(hooks_to_remove) == 1, hooks_to_remove
    hooks_to_remove[0].remove()
    hooks.append(hooked_model.model.lm_head.register_forward_hook(partial(
        lm_head_hook,
        hooked_model=hooked_model, primitives=primitives_to_try, tokenizer=tokenizer, converted_mlp=converted_mlp,
        oa_vecs=oa_vecs
    )))
    hooks.append(hooked_model.model.transformer.wte.register_forward_hook(partial(save_wte_inputs, model=hooked_model)))
    hooks.append(hooked_model.model.transformer.wpe.register_forward_hook(partial(save_wpe_inputs, model=hooked_model)))
    prev_attn_forward = hooked_model.attention_forward
    hooked_model.attention_forward = lambda module, layer: primitives_attention_forward(hooked_model, module, layer, primitives=primitives_to_try, tokenizer=tokenizer, converted_mlp=converted_mlp, oa_vecs=oa_vecs)
        
    sampling_optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0, betas=(0.9, 0.995))
    device = hooked_model.model.device
    current_step = 0
    inputs = []
    training_logs = defaultdict(list)
    save_training_logs = []
    
    valid_checkpoint = None
    
    steps_below_match_threshold = 0
    match_acc_patience = 1
    
    def save_valid_checkpoint(step):
        """Save checkpoint only if current state is valid"""
        return {
            'samplers_state': {name: {k: v.detach().clone() for k, v in sampler.state_dict().items()} 
                             for name, sampler in samplers_dict.items()},
            'step': step
        }
    
    def revert_to_valid_checkpoint():
        """Revert to the latest valid checkpoint if available"""
        if valid_checkpoint is not None:
            hooked_model.logger(f"Reverting to valid checkpoint from step {valid_checkpoint['step']}")
            for name, sampler in samplers_dict.items():
                sampler.load_state_dict(valid_checkpoint['samplers_state'][name])
            sampling_optimizer.zero_grad()
            return True
        else:
            hooked_model.logger("No valid checkpoint available to revert to")
            return False
        
    valid_checkpoint = save_valid_checkpoint(0)
    
    for item in iterable_dataset:
        inputs.append(item)
        if len(inputs) == batch_size:
            batch = collator(inputs)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            batch.pop("labels")
            masks = mask_sampler.sample_binary_masks(batch_size)
            hooked_model.input_ids = batch["input_ids"]
            hooked_model.position_ids = batch["position_ids"]
            result = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch)
            logits = result.logits

            with torch.no_grad():
                target_logits = original_model(**batch).logits.detach()

            if not hasattr(iterable_dataset, "BCE") or not iterable_dataset.BCE:
                task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()

                target_shift_logits = target_logits[:, :-1]
                shift_logits = logits[:, :-1]
                shift_labels = labels[:, 1:]

                with torch.no_grad():
                    target_predictions = target_shift_logits.argmax(dim=-1)
                    predictions = shift_logits.argmax(dim=-1)

                    match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                    correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                    
                    match_acc = match.sum().item() / batch_size
                    acc = correct.sum().item() / batch_size

                loss = F.kl_div(F.log_softmax(shift_logits[shift_labels != -100], dim=-1),
                                F.log_softmax(target_shift_logits[shift_labels != -100], dim=-1),
                                log_target=True)

            else:
                mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)
                
                task_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                task_loss = task_loss[mask].mean().item()
        
                loss = F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")
                loss = loss[mask].mean()

                with torch.no_grad():
                    target_predictions = (target_logits > 0).long()
                    predictions = (logits.detach() > 0).long()

                    mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)

                    match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                    correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                    
                    match_acc = match.sum().item() / batch_size
                    acc = correct.sum().item() / batch_size
            
            if isnan(loss):
                hooked_model.logger(f"Stop learning because loss is nan")
                revert_to_valid_checkpoint()
                break

            training_logs["task_loss"].append(task_loss)
            training_logs["loss"].append(loss.item())
            training_logs["acc"].append(acc)
            training_logs["match_acc"].append(match_acc)

            penalty_tensor = sum([sampler.get_penalty() for sampler in samplers_dict.values()])
            for sampler in samplers_dict.values():
                assert sampler.average_over_pixels == list(samplers_dict.values())[0].average_over_pixels
            if list(samplers_dict.values())[0].average_over_pixels:
                num_pixels = sum([sampler.indep_prod.numel() for sampler in samplers_dict.values()])
                penalty_tensor = penalty_tensor / num_pixels

            penalty = float(penalty_tensor.item())
            training_logs["penalty"].append(penalty)
            if isnan(penalty):
                hooked_model.logger(f"Stop learning because penalty is nan")
                revert_to_valid_checkpoint()
                break

            loss = loss + lamb * penalty_tensor
            training_logs["full_loss"].append(loss.item())

            sampling_optimizer.zero_grad()
            loss.backward()
            sampling_optimizer.step()

            for sampler_name, sampler in samplers_dict.items():
                nan_count = sum(p.isnan().sum().item() for p in samplers_dict[sampler_name].parameters())
                if nan_count > 0:
                    hooked_model.logger(f"num NaN {sampler_name} = {nan_count}")

            del batch, labels, masks, logits, target_logits, result
            if 'loss' in locals():
                del loss
            if 'penalty_tensor' in locals():
                del penalty_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            inputs = []
            current_step += 1
            
            if current_step % log_interval == 0:
                hooked_model.logger(f"step: {current_step}", {k: sum(v)/len(v) for k, v in training_logs.items()} )
                save_training_logs.append({k: sum(v)/len(v) for k, v in training_logs.items()})
                cur_match_acc = sum(training_logs["match_acc"]) / len(training_logs["match_acc"])
                training_logs = defaultdict(list)

                if cur_match_acc < match_acc_threshold:
                    steps_below_match_threshold += 1
                    if steps_below_match_threshold >= match_acc_patience:
                        hooked_model.logger(f"Stop learning: match accuracy below threshold {match_acc_threshold} for {match_acc_patience} consecutive intervals. Current match_acc: {cur_match_acc}")
                        revert_to_valid_checkpoint()
                        break
                else:
                    steps_below_match_threshold = 0
                    valid_checkpoint = None
                    valid_checkpoint = save_valid_checkpoint(current_step)

            if current_step == num_steps:
                break
    
    valid_checkpoint = None
    
    for h in hooks:
        h.remove()
    hooked_model.hooks.append(hooked_model.model.lm_head.register_forward_hook(
        hooked_model.lm_head_hook
    ))
    hooked_model.attention_forward = prev_attn_forward
    if save_dir is not None:
        save_subdir = Path(save_dir) / "learning_logs"
        save_subdir.mkdir(exist_ok=True, parents=True)
        already_saved = list(map(int, [f[:f.find(".json")] for f in os.listdir(str(save_subdir))]))
        if len(already_saved) == 0:
            new_logs = "1"
        else:
            new_logs = str(max(already_saved) + 1)
        save_file = save_subdir / f"{new_logs}.json"
        save_content = {
            "logs": save_training_logs,
            "hyperparams": tuple_of_hyperparams
        }
        with open(save_file, "w") as file:
            json.dump(save_content, file)