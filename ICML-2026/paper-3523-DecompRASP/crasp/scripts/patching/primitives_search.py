import torch
import json
import copy
from pathlib import Path
from dataclasses import dataclass, asdict

from primitives_for_coefficients import ATTENTION_CONST_ALL_PRIMITIVES, ATTENTION_ALL_PRIMITIVES
from try_primitives import try_primitives
from primitives_classes import AbstractPrimitive, PrimitiveEval
from primitives_for_coefficients import LOGITS_CONST_ALL_PRIMITIVES, LOGITS_ALL_PRIMITIVES
from primitives_helpers import is_token_dim_activation, get_product_for_one_side_for_unembed_ignore_dep_prod, expand_grid, get_product_for_one_side_for_head_ignore_dep_prod
from round_primitive import MatrixRounder, base_learning_loop

def greedy_primitives(primitives_before_search,
                          hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp, hyperparams, save_dir, acc_match_threshold):
    only_default_scalars = hyperparams["only_default_scalars"]
    if only_default_scalars:
        SCALING_FACTORS = [1e4]
    else:
        SCALING_FACTORS = [1e4, 1., 0.01, 0.1, 10, 100]
    hooked_model.logger("Start searching for attention: greedy_attention_primitives")
    primitives_to_try = copy.deepcopy(primitives_before_search)
    acc, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
                    test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            vanilla_interactions = primitives_before_search[layer][head].keys()
            assert all([primitives_before_search[layer][head][interaction] is None
                        for interaction in vanilla_interactions]), vanilla_interactions
            for interaction in vanilla_interactions:
                primitives_parameters = {}
                if interaction.activation_name_to_keep_q is None:
                    k_token_dim = is_token_dim_activation(interaction.activation_name_to_keep_k, converted_mlp)
                    hooked_model.logger(f"{interaction.activation_name_to_keep_k}: {k_token_dim}")
                    if k_token_dim:
                        if only_default_scalars:
                            primitives_parameters["primitive"] = [p for p in ATTENTION_CONST_ALL_PRIMITIVES if p.has_default_scalar]
                        else:
                            primitives_parameters["primitive"] = ATTENTION_CONST_ALL_PRIMITIVES
                    else:
                        if only_default_scalars:
                            primitives_parameters["primitive"] = [p for p in ATTENTION_CONST_ALL_PRIMITIVES if (not p.is_only_token) and p.has_default_scalar]
                        else:
                            primitives_parameters["primitive"] = [p for p in ATTENTION_CONST_ALL_PRIMITIVES if not p.is_only_token]
                    primitives_parameters["special_primitive"] = None
                else:
                    q_token_dim = is_token_dim_activation(interaction.activation_name_to_keep_q, converted_mlp)
                    k_token_dim = is_token_dim_activation(interaction.activation_name_to_keep_k, converted_mlp)
                    hooked_model.logger(f"{interaction.activation_name_to_keep_k}: {k_token_dim}")
                    hooked_model.logger(f"{interaction.activation_name_to_keep_q}: {q_token_dim}")
                    if k_token_dim:
                        if q_token_dim:
                            if only_default_scalars:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar]
                                primitives_parameters["special_primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar]
                            else:
                                primitives_parameters["primitive"] = ATTENTION_ALL_PRIMITIVES
                                primitives_parameters["special_primitive"] = ATTENTION_ALL_PRIMITIVES
                        else:
                            if only_default_scalars:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar]
                            else:
                                primitives_parameters["primitive"] = ATTENTION_ALL_PRIMITIVES
                            primitives_parameters["special_primitive"] = None
                    else:
                        if q_token_dim:
                            if only_default_scalars:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar and not p.is_only_token]
                                primitives_parameters["special_primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar and not p.is_only_token]
                            else:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if not p.is_only_token]
                                primitives_parameters["special_primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if not p.is_only_token]
                        else:
                            if only_default_scalars:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if p.has_default_scalar and not p.is_only_token]
                            else:
                                primitives_parameters["primitive"] = [p for p in ATTENTION_ALL_PRIMITIVES if not p.is_only_token]
                            primitives_parameters["special_primitive"] = None
                sets_of_primitives = expand_grid(primitives_parameters)
                found_for_interaction = False
                for set_of_primitives in sets_of_primitives:
                    if found_for_interaction:
                        break
                    primitive = set_of_primitives["primitive"]
                    special_primitive = set_of_primitives["special_primitive"]
                    if primitive:
                        hooked_model.logger(f"primitive={primitive.name}")
                    if special_primitive:
                        hooked_model.logger(f"special_primitive={special_primitive.name}")
                    for scaling_factor in SCALING_FACTORS:
                        if found_for_interaction:
                            break
                        primitives_to_try[layer][head][interaction] = AbstractPrimitive(
                            name="predefined_primitive",
                            primitive=primitive,
                            special_primitive=special_primitive,
                            scaling_factor_primitive=scaling_factor,
                        )
                        _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
                            test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
                        hooked_model.logger(f"kl: {kl}; acc match {acc_match}")
                        if acc_match < acc_match_threshold:
                            hooked_model.logger(f"Reject primitive; scaling={scaling_factor}; {acc_match} < {acc_match_threshold}")
                            primitives_to_try[layer][head][interaction] = None
                        else:
                            hooked_model.logger(f"Accept primitive; scaling={scaling_factor}")
                            found_for_interaction = True

    hooked_model.logger("Start searching for logits: greedy_logits_primitives")
    vanilla_interactions = primitives_before_search["lm_head"].keys()
    assert all([primitives_before_search["lm_head"][interaction] is None
                for interaction in vanilla_interactions]), vanilla_interactions
    for interaction in vanilla_interactions:
        primitives_parameters = {}
        hooked_model.logger(f"interaction.activation_name_to_keep={interaction.activation_name_to_keep}")
        if interaction.activation_name_to_keep == "vocab_bias":
            primitives_parameters["primitive"] = LOGITS_CONST_ALL_PRIMITIVES
            primitives_parameters["special_primitive"] = None
        else:
            token_dim = is_token_dim_activation(interaction.activation_name_to_keep, converted_mlp)
            if token_dim:
                primitives_parameters["primitive"] = LOGITS_ALL_PRIMITIVES
                primitives_parameters["special_primitive"] = LOGITS_ALL_PRIMITIVES
            else:
                primitives_parameters["primitive"] = LOGITS_ALL_PRIMITIVES
                primitives_parameters["special_primitive"] = None

        sets_of_primitives = expand_grid(primitives_parameters)
        found_for_interaction = False
        for set_of_primitives in sets_of_primitives:
            primitive = set_of_primitives["primitive"]
            special_primitive = set_of_primitives["special_primitive"]
            if primitive:
                hooked_model.logger(f"primitive={primitive.name}")
            if special_primitive:
                hooked_model.logger(f"special_primitive={special_primitive.name}")
            if found_for_interaction:
                break
            for scaling_factor in SCALING_FACTORS:
                if found_for_interaction:
                    break
                primitives_to_try["lm_head"][interaction] = AbstractPrimitive(
                    name="predefined_primitive",
                    primitive=primitive,
                    scaling_factor_primitive=scaling_factor,
                    special_primitive=special_primitive,
                )
                _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
                    test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
                hooked_model.logger(f"acc_match: {acc_match}, kl: {kl}; harden coefs")
                if acc_match < acc_match_threshold:
                    hooked_model.logger(f"Reject, scaling={scaling_factor}; {acc_match} < {acc_match_threshold}")
                    primitives_to_try["lm_head"][interaction] = None
                else:
                    hooked_model.logger(f"Accept, scaling={scaling_factor}")
                    found_for_interaction = True
    num_primitives = 0
    primitives_converted = 0
    for interaction, abstract_primitive in primitives_to_try["lm_head"].items():
        num_primitives += 1
        if abstract_primitive is not None:
            primitives_converted += 1
            new_name = f"{abstract_primitive.scaling_factor_primitive} * predefined_primitive: {abstract_primitive.primitive.name};"
            if abstract_primitive.special_primitive is not None:
                new_name = new_name + f" special_primitive={abstract_primitive.special_primitive.name};"
            primitives_to_try["lm_head"][interaction].name = new_name
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            for interaction, abstract_primitive in primitives_to_try[layer][head].items():
                num_primitives += 1
                if abstract_primitive is not None:
                    primitives_converted += 1
                    new_name = f"{abstract_primitive.scaling_factor_primitive} * predefined_primitive: {abstract_primitive.primitive.name};"
                    if abstract_primitive.special_primitive is not None:
                        new_name = new_name + f" special_primitive={abstract_primitive.special_primitive.name};"
                    primitives_to_try[layer][head][interaction].name = new_name

    if num_primitives == 0:
        primitives_eval = PrimitiveEval(
            zero_parameters=[1],
            total_parameters=[-1],
            is_fully_replaced=[True]
        )
        return primitives_before_search, primitives_eval
    primitives_eval = PrimitiveEval(
        zero_parameters=[primitives_converted],
        total_parameters=[num_primitives],
        is_fully_replaced=[(num_primitives == primitives_converted)]
    )
    return primitives_to_try, primitives_eval

def round_after_greedy_primitives(primitives_to_try_after_greedy,
                      hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparams, save_dir, acc_match_threshold):
    vanilla_interactions = [(p, "lm_head") for p in primitives_to_try_after_greedy["lm_head"] if primitives_to_try_after_greedy["lm_head"][p] is None]
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            vanilla_interactions.extend([(p, {"layer": layer, "head": head}) for p in primitives_to_try_after_greedy[layer][head] if primitives_to_try_after_greedy[layer][head][p] is None])
    assert all([primitives_to_try_after_greedy[interaction_params["layer"]][interaction_params["head"]][interaction] is None
                for interaction, interaction_params in vanilla_interactions if interaction_params != "lm_head"]), vanilla_interactions
    assert all([primitives_to_try_after_greedy["lm_head"][interaction] is None
                for interaction, interaction_params in vanilla_interactions if interaction_params == "lm_head"]), vanilla_interactions
    
    primitives_to_try_after_round, primitives_eval_after_round = round_for_vanilla_primitives(primitives_to_try_after_greedy, vanilla_interactions,
                      hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparams, save_dir, acc_match_threshold)
    
    is_fully_replaced = []
    for interaction in primitives_to_try_after_round["lm_head"].keys():
        if primitives_to_try_after_round["lm_head"][interaction].replacement_matrix is not None:
            if primitives_to_try_after_round["lm_head"][interaction].replacement_matrix.do_round:
                is_fully_replaced.append(True)
            else:
                is_fully_replaced.append(False)
        elif primitives_to_try_after_round["lm_head"][interaction].primitive is not None:
            is_fully_replaced.append(True)
        else:
            is_fully_replaced.append(False)
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            for interaction in primitives_to_try_after_round[layer][head]:
                if primitives_to_try_after_round[layer][head][interaction].replacement_matrix is not None:
                    if primitives_to_try_after_round[layer][head][interaction].replacement_matrix.do_round:
                        is_fully_replaced.append(True)
                    else:
                        is_fully_replaced.append(False)
                elif primitives_to_try_after_round[layer][head][interaction].primitive is not None:
                    is_fully_replaced.append(True)
                else:
                    is_fully_replaced.append(False)
    primitives_eval_after_round.is_fully_replaced = is_fully_replaced

    return primitives_to_try_after_round, primitives_eval_after_round
    
    

def round_primitives(primitives_before_search,
                      hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparams, save_dir, acc_match_threshold):
    vanilla_interactions = [(p, "lm_head") for p in primitives_before_search["lm_head"]]
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            vanilla_interactions.extend([(p, {"layer": layer, "head": head}) for p in primitives_before_search[layer][head]])
    assert all([primitives_before_search[interaction_params["layer"]][interaction_params["head"]][interaction] is None
                for interaction, interaction_params in vanilla_interactions if interaction_params != "lm_head"]), vanilla_interactions
    assert all([primitives_before_search["lm_head"][interaction] is None
                for interaction, interaction_params in vanilla_interactions if interaction_params == "lm_head"]), vanilla_interactions
    
    return round_for_vanilla_primitives(primitives_before_search, vanilla_interactions,
                      hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparams, save_dir, acc_match_threshold)
    

def round_for_vanilla_primitives(primitives_before_search, vanilla_interactions,
                      hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparams, save_dir, acc_match_threshold):
    training_lambda = hyperparams["training_lambda"]
    num_steps = hyperparams["num_steps"]
    lr = hyperparams["lr"]
    two_stages = hyperparams["two_stages"]
    params = {}
    for hyperparam in MatrixRounder.possible_params_and_default_values:
        params[hyperparam] = hyperparams[hyperparam]

    
    rounders = {}
    all_params = []
    for interaction, interaction_params in vanilla_interactions:
        if interaction_params != "lm_head":
            indep_prod_k = get_product_for_one_side_for_head_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, interaction_params["layer"], interaction_params["head"], "k", interaction.activation_name_to_keep_k)
            if interaction.activation_name_to_keep_q is not None:
                indep_prod_q = get_product_for_one_side_for_head_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, interaction_params["layer"], interaction_params["head"], "q", interaction.activation_name_to_keep_q)
                indep_prod = (indep_prod_q @ indep_prod_k.transpose(-1, -2)).detach()
            else:
                alpha = oa_vecs.q_bias_term.data[oa_vecs.to_q_bias[(interaction_params["layer"], interaction_params["head"], interaction.activation_name_to_keep_k)]].unsqueeze(0)
                indep_prod = (alpha @ indep_prod_k.transpose(-1, -2)).squeeze().detach()
        else:
            indep_prod = get_product_for_one_side_for_unembed_ignore_dep_prod(path=interaction.activation_name_to_keep,
                                                                        hooked_model=hooked_model, oa_vecs=oa_vecs, converted_mlp=converted_mlp).detach()
        rounder = MatrixRounder(indep_prod, params=params)
        if interaction_params == "lm_head":
            rounders[interaction.activation_name_to_keep] = rounder
        else:
            rounders[f"{interaction_params['layer']}-{interaction_params['head']}-{interaction.activation_name_to_keep_k}-{interaction.activation_name_to_keep_q}"] = rounder
        all_params.extend(rounder.parameters())

    if len(rounders) == 0:
        primitives_eval = PrimitiveEval(
            zero_parameters=[1],
            total_parameters=[-1],
            is_fully_replaced=[True]
        )
        return primitives_before_search, primitives_eval

    primitives_to_try = copy.deepcopy(primitives_before_search)
    _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
        test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"acc_match before replacement before fixing primitives as replacement matrices: {acc_match}, kl {kl};")

    for interaction in primitives_to_try["lm_head"]:
        if interaction.activation_name_to_keep in rounders:
            primitives_to_try["lm_head"][interaction] = AbstractPrimitive(
                name="replacement_matrix",
                replacement_matrix=rounders[interaction.activation_name_to_keep],
            )
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            for interaction in primitives_to_try[layer][head]:
                if f"{layer}-{head}-{interaction.activation_name_to_keep_k}-{interaction.activation_name_to_keep_q}" in rounders:
                    primitives_to_try[layer][head][interaction] = AbstractPrimitive(
                        name="replacement_matrix",
                        replacement_matrix=rounders[f"{layer}-{head}-{interaction.activation_name_to_keep_k}-{interaction.activation_name_to_keep_q}"],
                    )

    for rounder_name, rounder in rounders.items():
        rounder.disable_rounding()
    _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
        test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"acc_match before replacement: {acc_match}, kl {kl};")

    stages = [1, 2] if two_stages else [None]
    for stage in stages:
        for rounder_name, rounder in rounders.items():
            rounder.set_stage(stage)
        tuple_of_hyperparams = tuple([(k, v) for k, v in hyperparams.items()] + [("stage", stage)])
        base_learning_loop(hooked_model, original_model, collator, iterable_dataset, mask_sampler, oa_vecs,
                  primitives_to_try, tokenizer, converted_mlp,
                  all_params, rounders, lr=lr,
                  lamb = training_lambda, num_steps = num_steps, batch_size = test_batch_size, log_interval = test_num_test_step,
                save_dir=save_dir, tuple_of_hyperparams=tuple_of_hyperparams,
                match_acc_threshold=acc_match_threshold)
        _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
            test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
        hooked_model.logger(f"acc_match after stage {stage}, not rounded: {acc_match}, kl {kl};")

    is_fully_replaced = []
    for rounder_name, rounder in rounders.items():
        rounders[rounder_name].enable_rounding()
        _, kl, acc_match, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
            test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
        hooked_model.logger(f"{rounder_name}: acc_match after replacement, rounded: {acc_match}, kl {kl};")
        if acc_match < acc_match_threshold:
            hooked_model.logger(f"Reject rounding: {acc_match} < {acc_match_threshold}")
            rounders[rounder_name].disable_rounding()
            is_fully_replaced.append(False)
        else:
            is_fully_replaced.append(True)
            hooked_model.logger(f"Accept rounding")

    for rounder_name, rounder in rounders.items():
        matrix = rounder.get_matrix()
        print(f"non-zero params count {rounder_name}: {(matrix != 0.).sum().item()} out of {matrix.numel()}")
        print(f"num not rounded {rounder_name}: {(matrix != torch.round(matrix)).sum()} out of {matrix.numel()}, l1={torch.abs(matrix - torch.round(matrix)).sum()}")

    zero_parameters = []
    total_parameters = []
    for rounder_name, rounder in rounders.items():
        matrix = rounder.get_matrix()
        zero_parameters.append((matrix == 0).sum().item())
        total_parameters.append(matrix.numel())

    for interaction in primitives_to_try["lm_head"].keys():
        if interaction.activation_name_to_keep in rounders:
            replacement_matrix = primitives_to_try["lm_head"][interaction].replacement_matrix
            matrix = replacement_matrix.get_matrix()
            new_name = f"projection[non-zero={(matrix != 0.).sum().item()} out of {matrix.numel()}]"
            if replacement_matrix.do_round:
                new_name = new_name + "[round]"
            primitives_to_try["lm_head"][interaction].name = new_name
    for layer in range(len(hooked_model.model.transformer.h)):
        for head in range(hooked_model.model.transformer.h[layer].attn.num_heads):
            for interaction in primitives_to_try[layer][head]:
                if f"{layer}-{head}-{interaction.activation_name_to_keep_k}-{interaction.activation_name_to_keep_q}" in rounders:
                    replacement_matrix = primitives_to_try[layer][head][interaction].replacement_matrix
                    matrix = replacement_matrix.get_matrix()
                    new_name = f"projection[non-zero={(matrix != 0.).sum().item()} out of {matrix.numel()}]"
                    if replacement_matrix.do_round:
                        new_name = new_name + "[round]"
                    primitives_to_try[layer][head][interaction].name = new_name

    primitives_eval = PrimitiveEval(
        zero_parameters=zero_parameters,
        total_parameters=total_parameters,
        is_fully_replaced=is_fully_replaced
    )

    return primitives_to_try, primitives_eval


LAMBDAS = [0.1, 1e-2, 1e-4]

POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES = {
    "round": {
        "training_lambda": LAMBDAS,
        "num_steps": 2000,
        "lr": [1e-4],
        "threshold_on_acc": [0.9],
    },
    "greedy_search": {
        "threshold_on_acc": [0.95],
        "only_default_scalars": [True],
    },
}
POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES["round"].update(MatrixRounder.possible_params_and_default_values)
    
def try_replacing_with_primitives(replacement_type, primitives_to_try,
                          hooked_model, original_model, iterable_dataset, test_batch_size,
                      test_num_test_step, collator, mask_sampler, oa_vecs,
                      tokenizer, converted_mlp,
                      hyperparameters, save_configs_path):
    if replacement_type == "greedy_search_then_round":
        if hyperparameters is None:
            hyperparameters = [
                POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES["greedy_search"],
                POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES["round"]
            ]
        else:
            for replacement_style_part_i, replacement_style_part in enumerate(["greedy_search", "round"]):
                for param in POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES[replacement_style_part]:
                    if param not in hyperparameters[replacement_style_part_i]:
                        hyperparameters[replacement_style_part_i][param] = POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES[replacement_style_part][param]
    else:
        if hyperparameters is None:
            hyperparameters = POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES[replacement_type]
        else:
            for param in POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES[replacement_type]:
                if param not in hyperparameters:
                    hyperparameters[param] = POSSIBLE_HYPERPARAMETERS_AND_DEFAULT_VALUES[replacement_type][param]

    hooked_model.logger(f"Start searching for primitives with {replacement_type} replacement")
    primitives_before_replacement = copy.deepcopy(primitives_to_try)
    acc_original, kl_original, acc_match_original, _ = try_primitives(hooked_model, original_model, primitives_to_try, iterable_dataset, test_batch_size,
        test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
    hooked_model.logger(f"acc_match before replacement with {replacement_type}: {acc_match_original} ; kl {kl_original}")

    for cur_replacement_type_i, cur_replacement_func in enumerate(REPLACEMENT_TYPE_TO_FUNCTION[replacement_type]):
        accepted_primitives = []
        accs_per_hyperparams_dict = {}
        if replacement_type == "greedy_search_then_round":
            sets_of_hyperparams = expand_grid(hyperparameters[cur_replacement_type_i])
        else:
            sets_of_hyperparams = expand_grid(hyperparameters)
        for set_of_hyperparams in sets_of_hyperparams:
            hooked_model.logger(f"Running for hyperparams: {set_of_hyperparams}")
            acc_match_threshold = acc_match_original * set_of_hyperparams["threshold_on_acc"]
            acc_match_threshold_for_search = acc_match_original * (set_of_hyperparams["threshold_on_acc"] + 0.01)
            primitives_after_replacement, primitives_eval = cur_replacement_func(primitives_to_try,
                                            hooked_model, original_model, iterable_dataset, test_batch_size,
                                            test_num_test_step, collator, mask_sampler, oa_vecs,
                                            tokenizer, converted_mlp,
                                            set_of_hyperparams, save_configs_path, acc_match_threshold_for_search)

            acc, kl, acc_match, task_loss = try_primitives(hooked_model, original_model, primitives_after_replacement, iterable_dataset, test_batch_size,
                test_num_test_step, collator, mask_sampler, oa_vecs, tokenizer, converted_mlp)
            hooked_model.logger(f"acc_match after replacement with {replacement_type}.{cur_replacement_type_i}: {acc_match} ; kl {kl}")
            tuple_of_hyperparams = tuple([(k, v) for k, v in set_of_hyperparams.items()])
            accs_per_hyperparams_dict[str(tuple_of_hyperparams)] = {
                "acc": acc, "kl": kl, "acc_match": acc_match, "task_loss": task_loss, "eval": asdict(primitives_eval),
                "original_acc": acc_original, "original_kl": kl_original, "original_acc_match": acc_match_original,
                "primitives_names": {
                    str((layer, head, interaction.activation_name_to_keep_q, interaction.activation_name_to_keep_k)): (
                        primitives_after_replacement[layer][head][interaction].name if primitives_after_replacement[layer][head][interaction] else None
                    )
                    for layer in range(len(hooked_model.model.transformer.h))
                    for head in range(hooked_model.model.transformer.h[layer].attn.num_heads)
                    for interaction in primitives_after_replacement[layer][head]
                }
            }
            accs_per_hyperparams_dict[str(tuple_of_hyperparams)]["primitives_names"].update({
                interaction.activation_name_to_keep: (
                    primitives_after_replacement["lm_head"][interaction].name if primitives_after_replacement["lm_head"][interaction] else None
                )
                for interaction in primitives_after_replacement["lm_head"]
            })
            if acc_match < acc_match_threshold:
                hooked_model.logger(f"Reject run for hyperparams; {acc_match} < {acc_match_threshold}")
            else:
                hooked_model.logger(f"Accept run for hyperparams; ; {acc_match} >= {acc_match_threshold}")
                avg_params = sum([z / t for z, t in zip(primitives_eval.zero_parameters, primitives_eval.total_parameters)]) / len(primitives_eval.total_parameters)
                avg_replacement = sum(primitives_eval.is_fully_replaced) / len(primitives_eval.is_fully_replaced)
                accepted_primitives.append((avg_params, acc_match, avg_replacement, primitives_after_replacement))
        if len(accepted_primitives) > 0:
            sorted_accepted_primitives = sorted(accepted_primitives, key=lambda t: (-t[0], -t[1], -t[2]))
            primitives_to_try = sorted_accepted_primitives[0][-1]

        path_to_config_file = f"{save_configs_path}/accs_per_hyperparams/{replacement_type}.{cur_replacement_type_i}/{hooked_model.model_name}.json"
        Path(path_to_config_file).parent.mkdir(exist_ok=True, parents=True)
        with open(path_to_config_file, "w") as file:
            json.dump(accs_per_hyperparams_dict, file, indent=2)

    if len(accepted_primitives) > 0:
        sorted_accepted_primitives = sorted(accepted_primitives, key=lambda t: (-t[0], -t[1], -t[2]))
        hooked_model.logger(f"sorted_accepted_primitives: {[p[:-1] for p in sorted_accepted_primitives]}")
        return sorted_accepted_primitives[0][-1]
    return primitives_before_replacement

REPLACEMENT_TYPE_TO_FUNCTION = {
    "round": [round_primitives],
    "greedy_search": [greedy_primitives],
    "greedy_search_then_round": [greedy_primitives, round_after_greedy_primitives],
}