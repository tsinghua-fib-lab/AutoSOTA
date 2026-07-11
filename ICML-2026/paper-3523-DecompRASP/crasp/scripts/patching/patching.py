from torch.utils.data import Dataset, DataLoader
from functools import partial
from matplotlib import pyplot as plt
from transformers import GPT2LMHeadModel
import yaml
import click
from functools import partial
import torch
import torch.nn.functional as F
import json
from pathlib import Path
import numpy as np
from copy import deepcopy
from patching_utils import get_logging_function, set_seed, restore_int_keys
from patching_data import EvalDataset, get_tokenizer_and_dataset_for_task
from train_new_models import customCollator, customBCECollator
from collections import defaultdict
from pruning_model import MaskSampler, OptimalAblationVectors, PruningModelWithHooks, MaskSamplerFullPaths, PruningModelWithHooksFullPaths, \
    MaskSamplerForQK, OptimalQueryBiasVectors, PruningModelWithHooksForQK, convert_config_fullpaths_, convert_mask_to_config_, convert_mask_to_config_qk_, \
    convert_full_paths_config_to_prune_inside_kq_config_, capture_LN_var, convert_config_fullpaths_incl_mlp_, remove_other_edges_after_QK_pruning_
from patching_helper_functions import get_full_possible_config_for_pruning

@click.command()
@click.option('--config_path', default="config/patching.yaml", help="Path to config file")
def run(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    torch.set_printoptions(sci_mode=False, precision=5)
    device = torch.device("cuda") if config_dict["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    set_seed(config_dict["seed"])
    output_dir: Path = Path(config_dict["output_dir"]) / config_dict["exp_name"]
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_config_path = output_dir / "args.json"
    with open(output_config_path, "w") as f:
        json.dump(config_dict, f)
    output_dict = {}

    logger = get_logging_function(output_dir)

    logger(config_dict)
    model = GPT2LMHeadModel.from_pretrained(Path(config_dict["path_to_saved_model"]) / config_dict["model_name"]).to(device)
    model.eval()

    orig_model = GPT2LMHeadModel.from_pretrained(Path(config_dict["path_to_saved_model"]) / config_dict["model_name"]).to(device)
    orig_model.eval()

    task_name = config_dict["model_name"].split("-")[0]
    tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task_name, config_dict["length_range"], config_dict["max_test_length"], config_dict)

    logger(tokenizer.vocab)
    if hasattr(iterable_dataset, "BCE"):
        use_BCE = iterable_dataset.BCE
    else:
        use_BCE = False
    if not use_BCE:
        collator = customCollator(tokenizer.pad_token_id)
    else:
        collator = customBCECollator(tokenizer.pad_token_id)


    num_layers = len(model.transformer.h)
    num_heads_per_layer = {layer: model.transformer.h[layer].attn.num_heads for layer in range(num_layers)}
    
    # compatible with old config
    if "start_stage" in config_dict and "end_stage" in config_dict:
        stages = range(config_dict["start_stage"]-1, config_dict["end_stage"])
    else:
        stages = range(1 + int(config_dict["use_full_paths_as_subspaces"] + int(config_dict["prune_components_inside_kq_product"])))
    
    if stages[0] == 0:
        current_config = get_full_possible_config_for_pruning(num_heads_per_layer)
    else:
        oa_vecs = torch.load(Path(config_dict["input_exp_path"]) / "oa_vecs.pt", map_location=device, weights_only=False)
        with open(Path(config_dict["input_exp_path"]) / "output.json") as f:
            output_dict = json.load(f)
            current_config = output_dict[f"result_patching_config_global_iteration_{stages[0]-1}"]
        current_config = restore_int_keys(current_config)
        acc_match = output_dict["acc_match"]
        acc_task = output_dict["acc_task"]
        kl_div = output_dict["kl_div"]
        task_loss = output_dict["task_loss"]

    for global_removal_iteration in stages:
        set_seed(config_dict["seed"])
        logger("\n", "*"*10, "global iteration", global_removal_iteration, "*"*10)
        
        if global_removal_iteration == 0:

            mask_sampler = MaskSampler(current_config).to(device)
            if "init_sample_param" in config_dict:
                mask_sampler.sample_params.data *= config_dict["init_sample_param"]
            linear_LN = config_dict.get("linear_LN", True)
            hooked_model = PruningModelWithHooks(model, current_config, mask_sampler.mapping_to_param_idx, logger, linear_LN)
            logger("mask param names", [n for n, p in mask_sampler.named_parameters()])
            logger("total edge count", sum(p.numel() for p in mask_sampler.parameters()))

            converted_LN_var = torch.zeros(len(mask_sampler.output_vertex))
            if linear_LN:
                LN_var = capture_LN_var(iterable_dataset, orig_model, collator, device, [tokenizer.pad_token_id, tokenizer.eos_token_id])
                for i, output_v in enumerate(mask_sampler.output_vertex):
                    match output_v:
                        case (layer, act, head):
                            converted_LN_var[i] = LN_var[layer*2]
                        case (layer, mlp):
                            converted_LN_var[i] = LN_var[layer*2+1]
                        case (lm_head,):
                            converted_LN_var[i] = LN_var[-1]
                converted_LN_var = converted_LN_var.log()
            else:
                logger(" *** use real LayerNorm *** ")

            oa_vecs = OptimalAblationVectors(mask_sampler.input_vertex, mask_sampler.output_vertex, mask_sampler.output_vertex, None, model.config, converted_LN_var).to(device)
            # init oa_out better
            for layer in range(num_layers):
                for head in range(num_heads_per_layer[layer]):
                    for act in ["q", "k", "v"]:
                        for i in range(layer):
                            oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[(layer, act, head)]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)
                for i in range(layer+1):
                    oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[(layer, "mlp")]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)
            for i in range(num_layers):
                oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[("lm_head",)]] += model.transformer.h[i].attn.resid_dropout(model.transformer.h[i].attn.c_proj.bias)

            # original hyper-param: lr=0.1 for sampler, 0.002 for oa, lamb = 1e-3, gamma = 0.5
            
            lamb = config_dict["sparsity_coef0_for_pruning"]
            num_steps = config_dict["training_steps0_for_pruning"]
            

        elif global_removal_iteration == 1:
            if not config_dict["split_mlps"]:
                convert_config_fullpaths_(current_config, num_heads_per_layer)
            else:
                convert_config_fullpaths_incl_mlp_(current_config, num_heads_per_layer)
            logger("after conversion to full paths", current_config)
            mask_sampler = MaskSamplerFullPaths(current_config, config_dict["split_mlps"]).to(device)
            if "init_sample_param" in config_dict:
                mask_sampler.sample_params.data *= config_dict["init_sample_param"]
            hooked_model = PruningModelWithHooksFullPaths(model, current_config, mask_sampler.mapping_to_param_idx, config_dict["split_mlps"], logger)
            logger("mask param names", [n for n, p in mask_sampler.named_parameters()])
            logger("total edge count", sum(p.numel() for p in mask_sampler.parameters()))

            converted_LN_var = torch.zeros(len(mask_sampler.all_output_vertex))
            for i, output_v in enumerate(mask_sampler.all_output_vertex):
                match output_v:
                    case (layer, v, head, inp_v):
                        converted_LN_var[i] = oa_vecs.LN_var[oa_vecs.to_LN_idx[(layer, v, head)]].item()
                    case (layer, "mlp", inp_v):
                        converted_LN_var[i] = oa_vecs.LN_var[oa_vecs.to_LN_idx[(layer, "mlp")]].item()
                    case _:
                        converted_LN_var[i] = oa_vecs.LN_var[oa_vecs.to_LN_idx[output_v]].item()

            oa_vecs = OptimalAblationVectors(mask_sampler.input_vertex, mask_sampler.output_vertex, mask_sampler.all_output_vertex, mask_sampler.mlp_output_vertex, model.config, converted_LN_var).to(device)
            
            lamb = config_dict["sparsity_coef1_for_pruning"]
            num_steps = config_dict["training_steps1_for_pruning"]


        elif global_removal_iteration == 2:
            convert_full_paths_config_to_prune_inside_kq_config_(current_config, num_heads_per_layer)
            logger("after conversion to qk products", current_config)

            mask_sampler = MaskSamplerForQK(current_config).to(device)
            if "init_sample_param" in config_dict:
                mask_sampler.sample_params.data *= config_dict["init_sample_param"]

            hooked_model = PruningModelWithHooksForQK(model, current_config, mask_sampler.mapping_to_param_idx, hasattr(oa_vecs, "MLPs"), logger)
            logger("mask param names", [n for n, p in mask_sampler.named_parameters()])
            logger("total product count", sum(p.numel() for p in mask_sampler.parameters()))

            oa_vecs = OptimalQueryBiasVectors(mask_sampler.key_names, model.transformer.h[0].attn.head_dim, oa_vecs).to(device)

            lamb = config_dict["sparsity_coef2_for_pruning"]
            num_steps = config_dict["training_steps2_for_pruning"]

        if mask_sampler.sample_params.numel() == 0:
            hooked_model.remove_hooks()
            output_dict[f"result_patching_performance_global_iteration_{global_removal_iteration}"] = {"acc_match": acc_match, "acc_task": acc_task, "task_loss": task_loss, "kl_div": kl_div, "num_edges": 0, "coef": lamb}
            output_dict[f"result_patching_config_global_iteration_{global_removal_iteration}"] = deepcopy(current_config)
            logger("nothing to prune, exit...")
            continue
            
            
        if global_removal_iteration > 0:
            logger("***** pretrain oa vecs for output node *****")
            num_pretrain_steps = 500
            batch_size = 64
            log_interval = 50
            if global_removal_iteration == 1:
                oa_vecs.input_vertex_oa.requires_grad_(False)
                oa_vecs.LN_var.requires_grad_(False)
                param_groups = [
                    {"params": [oa_vecs.output_vertex_oa], "lr": config_dict["lr_oa_for_pruning"]}
                ]
                if hasattr(oa_vecs, "MLPs"):
                    param_groups.append({"params": oa_vecs.MLPs.parameters(), "lr": config_dict["lr_MLP_for_pruning"]}) 
                oa_optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
            elif global_removal_iteration == 2:
                oa_vecs.LN_var.requires_grad_(False)
                if hasattr(oa_vecs, "MLPs"):
                    oa_vecs.MLPs.requires_grad_(False)
                oa_optimizer = torch.optim.AdamW([oa_vecs.output_vertex_oa, oa_vecs.q_bias_term], lr=config_dict["lr_oa_for_pruning"], weight_decay=0)

            dataloader = DataLoader(iterable_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            training_logs = defaultdict(list)
            for current_step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                labels = batch.pop("labels")

                with torch.no_grad():
                    target_logits = orig_model(**batch).logits

                masks = mask_sampler.sample_binary_masks(batch_size)
                logits = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch).logits
                
                if not use_BCE:
                    task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()

                    target_logits = target_logits[:, :-1][labels[:, 1:]!=-100]
                    logits = logits[:, :-1][labels[:, 1:]!=-100]
                    loss = F.kl_div(F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1), log_target=True)
                else:
                    mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)
                    
                    task_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                    task_loss = task_loss[mask].mean().item()
            
                    loss = F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")
                    loss = loss[mask].mean()

                training_logs["kl_div"].append(loss.item())
                training_logs["task_loss"].append(task_loss)

                oa_optimizer.zero_grad()
                loss.backward()
                oa_optimizer.step()

                if (current_step+1) % log_interval == 0:
                    logger( {k: sum(v)/len(v) for k, v in training_logs.items()} )
                    training_logs = defaultdict(list)

                if (current_step+1) == num_pretrain_steps:
                    break
            
            if global_removal_iteration == 1:
                oa_vecs.input_vertex_oa.requires_grad_(True)
                oa_vecs.LN_var.requires_grad_(True)
            elif global_removal_iteration == 2:
                oa_vecs.LN_var.requires_grad_(True)
                if hasattr(oa_vecs, "MLPs"):
                    oa_vecs.MLPs.requires_grad_(True)

            for p in oa_vecs.parameters():
                p.grad = None

        logger("***** start real training *****")

        sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=config_dict["lr_sampler_for_pruning"], weight_decay=0, betas=(0.9, 0.995))
        if global_removal_iteration <= 1:
            param_groups = [
                {"params": [oa_vecs.LN_var], "lr": config_dict["lr_LN_var_for_pruning"]},
                {"params": [oa_vecs.input_vertex_oa], "lr": config_dict["lr_oa_for_pruning"]},
                {"params": [oa_vecs.output_vertex_oa], "lr": 1e-4},
            ]    
        elif global_removal_iteration == 2:
            param_groups = [
                {"params": [oa_vecs.LN_var], "lr": config_dict["lr_LN_var_for_pruning"]},
                {"params": [oa_vecs.q_bias_term], "lr": config_dict["lr_oa_for_pruning"]},
                {"params": [oa_vecs.output_vertex_oa], "lr": 1e-4},
            ]
        if global_removal_iteration > 0 and hasattr(oa_vecs, "MLPs"):
            param_groups.append({"params": oa_vecs.MLPs.parameters(), "lr": config_dict["lr_MLP_for_pruning"]}) 
        oa_optimizer = torch.optim.AdamW(param_groups, weight_decay=0)

        gamma = 0.0
        batch_size = config_dict["batch_size_for_pruning"]
        n_repeat = config_dict["num_repeat_for_pruning"]
        uniq_input_per_batch = batch_size // n_repeat
        assert uniq_input_per_batch * n_repeat == batch_size
        log_interval = 50

        # training
        count_down = num_steps
        patience = 3
        dataloader = DataLoader(iterable_dataset, batch_size=uniq_input_per_batch, shuffle=False, collate_fn=collator)
        training_logs = defaultdict(list)
        for current_step, batch in enumerate(dataloader):
            batch = {k: v.to(device).repeat(n_repeat, *([1]*(v.dim()-1))) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.no_grad():
                target_logits = orig_model(**batch).logits

            masks = mask_sampler.sample_masks(batch_size)
            logits = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch).logits
            
            if not use_BCE:
                task_loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten()).item()

                target_logits = target_logits[:, :-1][labels[:, 1:]!=-100]
                logits = logits[:, :-1][labels[:, 1:]!=-100]
                loss = F.kl_div(F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1), log_target=True)
            else:
                mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)
                
                task_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                task_loss = task_loss[mask].mean().item()
        
                loss = F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")
                loss = loss[mask].mean()

            training_logs["kl_div"].append(loss.item())
            training_logs["task_loss"].append(task_loss)
            penalty, (reg_edge, reg_node) = mask_sampler.get_penalty(gamma)
            training_logs["reg_edge"].append(reg_edge)
            training_logs["reg_node"].append(reg_node)
            loss = loss + lamb * penalty

            sampling_optimizer.zero_grad()
            oa_optimizer.zero_grad()
            loss.backward()
            
            training_logs["oa_grad_norm"].append(torch.nn.utils.clip_grad_norm_(oa_vecs.parameters(), max_norm=float('inf')).item())
            training_logs["sampler_grad_norm"].append(torch.nn.utils.clip_grad_norm_(mask_sampler.parameters(), max_norm=float('inf')).item())
            torch.nn.utils.clip_grad_norm_(mask_sampler.parameters(), 5)
            sampling_optimizer.step()
            oa_optimizer.step()

            nan_count = sum(p.isnan().sum().item() for p in mask_sampler.parameters())
            if nan_count > 0:
                logger("WARNING: num NaN", nan_count)

            if (current_step+1) % log_interval == 0:
                logger( {k: sum(v)/len(v) for k, v in training_logs.items()} )
                all_sample_p = torch.cat([p.data.detach().view(-1) for p in mask_sampler.parameters()], dim=0)
                hist, bin_edges = torch.histogram(all_sample_p.cpu(), bins=5)
                logger("histgram of sampling param", "\nhist", hist, "\nbin edges", bin_edges)

                if all_sample_p.max().item() < -2:
                    logger("All pruned, training is failed. Stop early...")
                    break

                if "baseline_loss" in config_dict and (sum(training_logs["kl_div"]) / len(training_logs["kl_div"])) > config_dict["baseline_loss"]:
                    patience -= 1
                    if patience == 0:
                        logger("Loss stuck at high value, training is failed. Stop early...")
                        break
                else:
                    patience = 3
                
                if ((all_sample_p > -1) & (all_sample_p < 1)).sum().item() > 0:
                    count_down = num_steps + 1

                training_logs = defaultdict(list)

            count_down -= 1
            if count_down == 0:
                break
            if (current_step+1) == 5000:
                break

        logger(f"finish training ({current_step+1} steps)")
        
        # testing
        num_test_step = 200
        num_correct = 0
        num_match = 0
        task_loss = 0
        kl_div = 0
        dataloader = DataLoader(iterable_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        loss_func = torch.nn.CrossEntropyLoss()
        # flag = True
        with torch.no_grad():
            for current_step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                target_logits = orig_model(**batch).logits

                masks = mask_sampler.sample_binary_masks(batch_size)
                logits = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch).logits

                if not use_BCE:
                    shift_target_logits = target_logits[:, :-1]
                    shift_logits = logits[:, :-1]
                    shift_labels = labels[:, 1:]
                    target_predictions = shift_target_logits.argmax(dim=-1)
                    predictions = shift_logits.argmax(dim=-1)

                    match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                    num_match += match.sum().item()

                    correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                    num_correct += correct.sum().item()

                    task_loss += loss_func(shift_logits.flatten(end_dim=1), shift_labels.flatten()).item()
                    kl_div += F.kl_div(F.log_softmax(shift_logits[shift_labels!=-100], dim=-1), F.log_softmax(shift_target_logits[shift_labels!=-100], dim=-1), log_target=True).item()
                
                else:
                    mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                    target_predictions = (target_logits > 0).long()
                    predictions = (logits > 0).long()

                    match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                    num_match += match.sum().item()

                    correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                    num_correct += correct.sum().item()
                
                    task_loss += F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")[~mask].mean().item()
                    kl_div += F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")[~mask].mean().item()

                if current_step+1 == num_test_step:
                    break

        acc_match = num_match / (num_test_step * batch_size)
        acc_task = num_correct / (num_test_step * batch_size)
        task_loss /= num_test_step
        kl_div /= num_test_step

        logger(f"\nfinish testing ({num_test_step * batch_size} samples), Acc match: {acc_match:.3f}  Acc task: {acc_task:.3f} \t Task loss: {task_loss:.4f}  KL div: {kl_div:.6f}")

        masks = mask_sampler.sample_binary_masks(1).squeeze(0)
        num_edges = (masks == 1).sum().item()
        if global_removal_iteration <= 1:
            logger("after pruning edge count", num_edges)
            convert_mask_to_config_(masks, current_config, mask_sampler.mapping_to_param_idx)
        elif global_removal_iteration == 2:
            logger("after pruning product count", num_edges)
            convert_mask_to_config_qk_(masks, current_config, mask_sampler.mapping_to_param_idx)
            remove_other_edges_after_QK_pruning_(current_config, hasattr(oa_vecs, "MLPs"), logger)
        logger("config after pruning\n", current_config)

        output_dict[f"result_patching_performance_global_iteration_{global_removal_iteration}"] = {"acc_match": acc_match, "acc_task": acc_task, "task_loss": task_loss, "kl_div": kl_div, "num_edges": num_edges, "coef": lamb}
        
        logger("End of global iteration", global_removal_iteration)
        output_dict[f"result_patching_config_global_iteration_{global_removal_iteration}"] = deepcopy(current_config)

        hooked_model.remove_hooks()
    
    torch.save(oa_vecs, output_dir / "oa_vecs.pt")
    output_dict["acc_match"] = acc_match
    output_dict["acc_task"] = acc_task
    output_dict["kl_div"] = kl_div
    output_dict["task_loss"] = task_loss

    output_file = output_dir / "output.json"
    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=4)

    logger.cleanup()


if __name__ == '__main__':
    run()