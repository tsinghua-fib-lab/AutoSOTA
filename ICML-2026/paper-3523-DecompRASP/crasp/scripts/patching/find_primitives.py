import torch
import json
from functools import partial
import copy
import re
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any

from try_primitives import try_primitives
from primitives_search import try_replacing_with_primitives
from primitives_classes import AttentionInteraction, LogitsInteraction

def try_find_primitives(hooked_model, original_model, iterable_dataset, batch_size,
                      num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp, model_name, save_configs_path,
                      search_method=None, hyperparameters=None):
    stats_to_save = {
        stat: {
            "before": None,
            "after": None
        }
        for stat in ["acc", "kl", "acc_match", "task_loss"]
    }
    primitives_to_try = {}
    for layer in range(len(hooked_model.model.transformer.h)):
        primitives_to_try[layer] = {}
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            primitives_to_try[layer][head] = {
                AttentionInteraction(
                    activation_name_to_keep_q=item[0],
                    activation_name_to_keep_k=item[1]
                ): None
                for item in hooked_model.config[layer]["qk"][head]
            }
            primitives_to_try[layer][head].update({
                AttentionInteraction(
                    activation_name_to_keep_k=item,
                ): None
                for item in hooked_model.config[layer]["k"][head]
            })
    primitives_to_try["lm_head"] = {
        LogitsInteraction(
            activation_name_to_keep=item,
        ): None
        for item in hooked_model.config["lm_head"] + ["vocab_bias"]
    }
    
    hooked_model.model_name = model_name
    hooked_model.save_matrices = False
    hooked_model.saved_matrices = set()
    hooked_model.save_matrices_path = f"{save_configs_path}/heatmaps"
    acc, kl, acc_match, task_loss = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, batch_size,
                      num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"ACC: {acc}, kl: {kl}; before trying on full dataset")
    stats_to_save["acc"]["before"] = acc
    stats_to_save["kl"]["before"] = kl
    stats_to_save["acc_match"]["before"] = acc_match
    stats_to_save["task_loss"]["before"] = task_loss

    hooked_model.save_matrices = False
    test_num_test_step = 10
    test_batch_size = 120
    acc_original, kl_original, _, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"ACC: {acc_original}, kl: {kl_original}; before trying on small dataset")

    primitives_to_try = try_replacing_with_primitives(search_method, primitives_to_try,
                                hooked_model, original_model, iterable_dataset, test_batch_size,
                                test_num_test_step, collator, mask_sampler, oa_vecs,
                                tokenizer, converted_mlp,
                                hyperparameters, save_configs_path)
    
    acc, kl, _, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"FINAL ACC {acc}; FINAL KL {kl}")

    hooked_model.save_matrices = True
    acc, kl, acc_match, task_loss = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, batch_size,
                      num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.save_matrices = False
    hooked_model.logger(f"FINAL ACC ON FULL DATASET {acc}; KL {kl}")
    stats_to_save["acc"]["after"] = acc
    stats_to_save["kl"]["after"] = kl
    stats_to_save["acc_match"]["after"] = acc_match
    stats_to_save["task_loss"]["after"] = task_loss

    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            new_list_qk_interactions = []
            new_list_k_interactions = []
            for interaction, primitive in primitives_to_try[layer][head].items():
                item = asdict(interaction)
                if primitive is not None:
                    item["primitive"] = primitive.name
                    if primitive.primitive is not None:
                        item["predefined_primitive"] = primitive.primitive.name
                    if primitive.special_primitive is not None:
                        item["predefined_special_primitive"] = primitive.special_primitive.name
                    if primitive.scaling_factor_primitive is not None:
                        item["scaling_factor_primitive"] = primitive.scaling_factor_primitive
                    if primitive.replacement_matrix is not None:
                        item["replacement_matrix"] = primitive.replacement_matrix.get_matrix().tolist()
                else:
                    item["primitive"] = None
                if interaction.activation_name_to_keep_q is None:
                    new_list_k_interactions.append(item)
                else:
                    new_list_qk_interactions.append(item)
            primitives_to_try[layer][head] = {
                "qk_interactions": new_list_qk_interactions,
                "k_interactions": new_list_k_interactions
            }
    new_list = []
    for interaction, primitive in primitives_to_try["lm_head"].items():
        item = asdict(interaction)
        if primitive is not None:
            item["primitive"] = primitive.name
            if primitive.primitive is not None:
                item["predefined_primitive"] = primitive.primitive.name
            if primitive.special_primitive is not None:
                item["predefined_special_primitive"] = primitive.special_primitive.name
            if primitive.scaling_factor_primitive is not None:
                item["scaling_factor_primitive"] = primitive.scaling_factor_primitive
            if primitive.replacement_matrix is not None:
                item["replacement_matrix"] = primitive.replacement_matrix.get_matrix().tolist()
        else:
            item["primitive"] = None
        new_list.append(item)
    primitives_to_try["lm_head"] = new_list

    result_config = {
        "primitives": primitives_to_try,
        "config": hooked_model.config,
        "accuracy": stats_to_save["acc"],
        "kl": stats_to_save["kl"],
        "acc_match": stats_to_save["acc_match"],
        "task_loss": stats_to_save["task_loss"],
    }
    path_to_config_file = f"{save_configs_path}/configs/{hooked_model.model_name}.json"
    Path(path_to_config_file).parent.mkdir(exist_ok=True, parents=True)
    with open(path_to_config_file, "w") as file:
        json.dump(result_config, file, indent=2)

    del hooked_model.model_name
    del hooked_model.save_matrices
    del hooked_model.save_matrices_path
    del hooked_model.saved_matrices

    return result_config