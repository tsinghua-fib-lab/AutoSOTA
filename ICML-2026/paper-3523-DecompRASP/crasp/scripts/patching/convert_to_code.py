import re
from transformers import GPT2LMHeadModel
import sys
import os
# sys.path.append(os.path.abspath("./streamlit_app"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app"))
from my_paths import path_to_saved_model
from pathlib import Path
import string
import torch
import numpy as np
import copy
import json
from patching_data import customTokenizer, get_tokenizer_for_task
from primitives_for_coefficients import ATTENTION_ALL_PRIMITIVES, ATTENTION_CONST_ALL_PRIMITIVES, LOGITS_ALL_PRIMITIVES, LOGITS_CONST_ALL_PRIMITIVES

def remove_unnecessary_primitives(config, primitives):
    def remove_interaction_attn(layer, head, tp, act_q, act_k, config, primitives):
        new_primitives = copy.deepcopy(primitives)
        new_config = copy.deepcopy(config)
        new_primitives[layer][head][tp] =  [p for p in new_primitives[layer][head][tp]
                                            if p["activation_name_to_keep_q"] != act_q or p["activation_name_to_keep_k"] != act_k]
        if tp == "qk_interactions":
            new_config[layer]["qk"][head] = [i for i in new_config[layer]["qk"][head] if i[0] != act_q or i[1] != act_k]
        elif tp == "k_interactions":
            new_config[layer]["k"][head] = [i for i in new_config[layer]["k"][head] if i != act_k]
        else:
            raise NotImplementedError()
        return new_config, new_primitives
    
    def remove_interaction_lm_head(act, config, primitives):
        new_primitives = copy.deepcopy(primitives)
        new_config = copy.deepcopy(config)
        new_primitives["lm_head"] =  [p for p in new_primitives["lm_head"] if p["activation_name_to_keep"] != act]
        new_config["lm_head"] = [i for i in new_config["lm_head"] if i != act]
        return new_config, new_primitives
    
    for layer in primitives:
        if layer == "lm_head":
            continue
        for head in primitives[layer]:
            interactions = primitives[layer][head]["qk_interactions"] + primitives[layer][head]["k_interactions"]
            interactions_tp = ["qk_interactions"] * len(primitives[layer][head]["qk_interactions"]) + ["k_interactions"] * len(primitives[layer][head]["k_interactions"])
            for interaction_tp, interaction in zip(interactions_tp, interactions):
                if (("predefined_primitive" in interaction and interaction["predefined_primitive"] == "uniform_zeros") and (
                    "predefined_special_primitive" not in interaction or interaction["predefined_special_primitive"] == "uniform_zeros"
                )):
                    config, primitives = remove_interaction_attn(layer, head, interaction_tp,
                                                                 interaction["activation_name_to_keep_q"], interaction["activation_name_to_keep_k"],
                                                                 config, primitives)
                if "replacement_matrix" in interaction and (torch.tensor(interaction["replacement_matrix"]) == 0).all():
                    config, primitives = remove_interaction_attn(layer, head, interaction_tp,
                                                                 interaction["activation_name_to_keep_q"], interaction["activation_name_to_keep_k"],
                                                                 config, primitives)
    interactions = primitives["lm_head"]
    for interaction in interactions:
        if (("predefined_primitive" in interaction and interaction["predefined_primitive"] == "uniform_zeros") and (
                    "predefined_special_primitive" not in interaction or interaction["predefined_special_primitive"] == "uniform_zeros"
        )):
            config, primitives = remove_interaction_lm_head(interaction["activation_name_to_keep"], config, primitives)
        if "replacement_matrix" in interaction and (torch.tensor(interaction["replacement_matrix"]) == 0).all():
            config, primitives = remove_interaction_lm_head(interaction["activation_name_to_keep"], config, primitives)

    all_used_paths = set()
    for layer in config:
        if layer == "lm_head":
            for act in config["lm_head"]:
                pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))
            continue
        for head in config[layer]["k"]:
            for act in config[layer]["k"][head]:
                pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))
        for head in config[layer]["qk"]:
            for act_pair in config[layer]["qk"][head]:
                for act in act_pair:
                    pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"
                    matches = re.findall(pattern, act)
                    for m_i in range(len(matches)):
                        all_used_paths.add("-".join(matches[m_i:]))

    has_unsplitted_mlps = False
    for layer in config:
        if layer == "lm_head":
            continue
        if f"mlp-{layer}" in all_used_paths:
            has_unsplitted_mlps = True

    if has_unsplitted_mlps:
        for layer in config:
            if layer == "lm_head":
                continue
            for act in config[layer]["mlp"]:
                pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))

    has_unsplitted_mlps = False
    for layer in config:
        if layer == "lm_head":
            continue
        if f"mlp-{layer}" in all_used_paths:
            has_unsplitted_mlps = True

    if has_unsplitted_mlps:
        for layer in config:
            if layer == "lm_head":
                continue
            for act in config[layer]["mlp"]:
                pattern = r"attn_output-\d+-\d+|mlp-\d+|wte|wpe"
                matches = re.findall(pattern, act)
                for m_i in range(len(matches)):
                    all_used_paths.add("-".join(matches[m_i:]))

    for layer in config:
        if layer == "lm_head":
            continue
        for head in config[layer]["v"]:
            for act in config[layer]["v"][head]:
                if f"attn_output-{layer}-{head}-{act}" not in all_used_paths:
                    config[layer]["v"][head] = [i for i in config[layer]["v"][head] if i != act]
        for act in config[layer]["mlp"]:
            if f"mlp-{layer}-{act}" not in all_used_paths and f"mlp-{layer}" not in all_used_paths:
                config[layer]["mlp"] = [i for i in config[layer]["mlp"] if i != act]

    
    return config, primitives

def convert_config_to_code_one_step_paths(config):
    def convert_list(lis):
        new_lis = list(map(lambda x: var_mapping[x], lis))
        if len(new_lis) == 1:
            return new_lis[0]
        else:
            return new_lis

    code = ""
    var_mapping = {"wpe": "pos", "wte": "token"}    # component name : variable name
    for layer_i in range(len(config)-1):
        layer_i = str(layer_i)
        for head_i in config[layer_i]["q"]:
            q = convert_list(config[layer_i]["q"][head_i])
            k = convert_list(config[layer_i]["k"][head_i])
            v = convert_list(config[layer_i]["v"][head_i])
            new_var = f"a{layer_i}-{head_i}"
            code += f"{new_var} = Attention(Q={q}, K={k}, V={v})\n"
            var_mapping[f"attn_output-{layer_i}-{head_i}"] = new_var

        if config[layer_i]["mlp"]:
            inp = convert_list(config[layer_i]["mlp"])
            new_var = f"m{layer_i}"
            code += f"{new_var} = MLP(inp={inp})\n"
            var_mapping[f"mlp-{layer_i}"] = new_var
    inp = convert_list(config["lm_head"])
    code += f"output = Unembed(inp={inp})"
    return code

def convert_config_to_code_full_paths(config, plot_heatmaps = False, tokenizer_max_test_length = None, model_name = None):
    if plot_heatmaps:
        model = GPT2LMHeadModel.from_pretrained(Path(path_to_saved_model) / model_name)
        task_name, arch_name = model_name.split("-")
        tokenizer = get_tokenizer_for_task(task_name, tokenizer_max_test_length)
        reversed_vocab = {index: letter for letter, index in tokenizer.vocab.items()}
        labels =  {"tokens": [reversed_vocab[i] for i in range(len(reversed_vocab))], "positions": [i for i in range(tokenizer_max_test_length)]}

    def convert_node_to_output(node, keep_v=False):
        # Matches 'attn_output-<l>-<h>', 'mlp-<l>', or the literals 'lm_head', 'wte', 'wpe'
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        splitted_node = re.findall(pattern, node)
        left_most_non_attn, left_most_non_attn_i = [(splitted_node[i], i) for i in range(len(splitted_node)) if "attn_output" not in splitted_node[i]][0]
        converted_attention = rf"[{var_mapping[left_most_non_attn]}_{{0}}...{var_mapping[left_most_non_attn]}_{{i}}]"
        cur_var = rf"{var_mapping[left_most_non_attn]}_{{i}}"
        for attn_op in reversed(splitted_node[:left_most_non_attn_i]):
            layer_str, head_str = re.match(r'attn_output-(\d+)-(\d+)', attn_op).groups()
            cur_var = rf"a^{{{layer_str}}}_{{{head_str}}}.{cur_var}"
            if keep_v:
                s = r'P \times ' + var_mapping['wpe'] if splitted_node[-1] == 'wpe' else r'T \times ' + var_mapping['wte'] if splitted_node[-1] == 'wte' else splitted_node[-1]
                converted_attention = rf"\text{{attn}}({cur_var}, {converted_attention}) \times V^{{{layer_str}}}_{{{head_str}}} \times {s}_{{i}}"
            else:
                converted_attention = rf"\text{{attn}}({cur_var}, {converted_attention})"
        return rf"{converted_attention}"

    code = []
    var_mapping = {"wpe": r"\text{{pos}}", "wte": r"\text{{token}}", "lm_head": r"\text{{lm-head}}"}
    if plot_heatmaps:
        node_to_residual_stream_mapping_matrix = {
            "wpe": model.transformer.wpe.weight[:len(tokenizer.vocab), :],
            "wte": model.transformer.wte.weight,
        }
        node_to_residual_stream_mapping_equation = {
            "wpe": "P",
            "wte": "T",
        }
    for layer in range(len(config) - 1):
        layer_str = str(layer)
        for head_str in config[layer_str]["q"]:
            v_inp = config[layer_str]["v"][head_str]
            q_inp = config[layer_str]["q"][head_str]
            k_inp = config[layer_str]["k"][head_str]
            for v_var, v_node in [(var_mapping[node], node) for node in v_inp]:
                new_var = f"a^{layer_str}_{head_str}.{v_var}"
                var_mapping[f"attn_output-{layer_str}-{head_str}-{v_node}"] = new_var
                if plot_heatmaps:
                    Q, K, V = model.transformer.h[int(layer_str)].attn.c_attn.weight.split(model.transformer.h[int(layer_str)].attn.c_attn.weight.shape[-1] // 3, dim=-1)
                    node_to_residual_stream_mapping_matrix[f"attn_output-{layer_str}-{head_str}-{v_node}"] = (
                        node_to_residual_stream_mapping_matrix[v_node] @ V
                    )
                    node_to_residual_stream_mapping_equation[f"attn_output-{layer_str}-{head_str}-{v_node}"] = (
                        rf"{node_to_residual_stream_mapping_equation[v_node]} \times V^{layer_str}_{head_str}"
                    )
                attention_components = []
                var_latex = rf"$a^{{{layer_str}}}_{{{head_str}}}.{v_var}_{{i}}$"
                for q_var, q_node in [(var_mapping[node], node) for node in q_inp]:
                    for k_var, k_node in [(var_mapping[node], node) for node in k_inp]:
                        attention_components.append({
                            "equation": rf"$\text{{ATTN}}^{{{layer_str}}}_{{{head_str}}}(Q={convert_node_to_output(q_node)}, K={convert_node_to_output(k_node)})$",
                            "left_var": None,
                            "right_var": None,
                            "function": node_to_residual_stream_mapping_matrix[q_node] @ (Q @ K.T) @ node_to_residual_stream_mapping_matrix[k_node].T if plot_heatmaps else None,
                            "function_equation":rf"$({node_to_residual_stream_mapping_equation[q_node]}) \times (Q \times K^T) \times ({node_to_residual_stream_mapping_equation[k_node]})^T$" if plot_heatmaps else None,
                            "labels_left": labels["tokens"] if q_node.endswith("wte") else labels["positions"],
                            "labels_right": labels["tokens"] if k_node.endswith("wte") else labels["positions"]
                        })
                if len(attention_components) > 0:
                    code.append({
                        "var": var_latex,
                        "type": "attention",
                        "components": attention_components
                    })
        mlp_inp = config[layer_str]["mlp"]
        new_var = f"mlp^{layer_str}"
        var_mapping[f"mlp-{layer_str}"] = new_var
        mlp_inp = config[layer_str]["mlp"]
        mlp_components = []
        var_latex = rf"$\text{{mlp}}^{{{layer_str}}}_{{i}}$"
        for mlp_var, mlp_node in [(var_mapping[node], node) for node in mlp_inp]:
            mlp_components.append({
                "equation": rf"${convert_node_to_output(mlp_node, keep_v=True)}$",
                "var": None,
            })
        if len(mlp_components) > 0:
            code.append({
                "var": var_latex,
                "type": "mlp",
                "components": mlp_components
            })
    lm_head_inp = config["lm_head"]
    lm_head_components = []
    var_latex = rf"$\text{{lm-head}}_{{i}}$"
    for lm_head_var, lm_head_node in [(var_mapping[node], node) for node in lm_head_inp]:
        lm_head_components.append({
            "equation": rf"${convert_node_to_output(lm_head_node, keep_v=True)}$",
            "var": None,
        })
    if len(lm_head_components) > 0:
        code.append({
            "var": var_latex,
            "type": "lm_head",
            "components": lm_head_components
        })

    code_printed = []
    for line in code:
        if line["type"] == "attention":
            code_printed.append(rf"{line['var']} = {' + '.join([c['equation'] for c in line['components']])}")
        elif line["type"] == "mlp":
            code_printed.append(rf"{line['var']} = {line['var']}({' + '.join([c['equation'] for c in line['components']])})")
        elif line["type"] == "lm_head":
            code_printed.append(rf"{line['var']} = {line['var']}({' + '.join([c['equation'] for c in line['components']])})")
        else:
            print(line)
    return code_printed, code


def map_primitive_to_names(primitive):
    match primitive:
        case "unknown":
            return "new", "element_wise_op"
        case "no_op":
            return "", ""
        case "erase":
            return "erased", "erase"
        case "harden":
            return "hardened", "harden"
        case ("sharpen", n):
            return "sharpened", "sharpen"
        case ("exists", idx):   # TODO: incorporate tokenizer
            return f"is_{idx}_exists", f"is_{idx}_exists"
        case ("forall", thr):
            return "is_pure", "is_pure"
        case ("01balance", pow, center):
            return "is_01_balance", "is_01_balance"
        case ("ABbalance", pow, center):
            return "is_AB_balance", "is_AB_balance"
        case ("diff", token1, token2):
            return f"diff_{token1}{token2}", f"diff_{token1}{token2}"
        case "combine":
            return "_x_", "Cartesian_product"
        case _:
            raise RuntimeError

def get_code_line_given_predefined_primitives_with_ops(select_name, counter, primitive, list_of_all_primitives, var_mapping_q,
                                              var_mapping_k, var_mapping_inp, layer_idx, head_idx):
    primitive_instance = [p for p in list_of_all_primitives if p.name == primitive["predefined_primitive"]]
    assert len(primitive_instance) == 1
    primitive_instance = primitive_instance[0]
    scaling_factor = primitive['scaling_factor_primitive']
    assert scaling_factor == 1e4, scaling_factor

    if var_mapping_q is None and var_mapping_k is None and var_mapping_inp == "bias":
        code_line = f"project(op=({primitive_instance.operation()})"
    elif var_mapping_q is None and var_mapping_k is None:
        code_line = f"project(inp={var_mapping_inp}, op=({primitive_instance.operation()})"
    elif var_mapping_q == "bias":
        code_line = f"select(k={var_mapping_k}, op=({primitive_instance.operation()})"
    else:
        code_line = f"select(q={var_mapping_q}, k={var_mapping_k}, op=({primitive_instance.operation()})"
        
    if "predefined_special_primitive" in primitive and primitive["predefined_special_primitive"] != primitive["predefined_primitive"]:
        special_primitive_instance = [p for p in list_of_all_primitives if p.name == primitive["predefined_special_primitive"]]
        assert len(special_primitive_instance) == 1
        special_primitive_instance = special_primitive_instance[0]
        assert special_primitive_instance.operation is not None, special_primitive_instance.name
        code_line = code_line + f",\n\t\tspecial_op=({special_primitive_instance.operation()})"
    code_line = code_line + ")"

    if var_mapping_q is not None and var_mapping_k is not None and var_mapping_q != "bias":
        code_line = code_line + f"\t# layer {layer_idx} head {head_idx}"
    return select_name, code_line

def get_code_line_given_replacement_matrices_with_ops(select_name, count_heatmaps, var_mapping_q,
                                              var_mapping_k, var_mapping_inp, layer_idx, head_idx,
                                              q, k, inp, converted_mlp, show_logits_for_unconverted_mlp):
    
    is_after_mlp = False
    op_letter = chr(count_heatmaps + ord('a'))
    if var_mapping_q is None and var_mapping_k is None and var_mapping_inp == "bias":
        code_line = f"project(op=\\circled{{{op_letter}}})"
    elif var_mapping_q is None and var_mapping_k is None:
        if "mlp" in inp and inp[inp.find("mlp"):] not in converted_mlp and not show_logits_for_unconverted_mlp:
            code_line = f"project(inp={var_mapping_inp}, op=(inp==out))"
            is_after_mlp = True
        else:
            code_line = f"project(inp={var_mapping_inp}, op=\\circled{{{op_letter}}})"
    elif var_mapping_q == "bias":
        code_line = f"select(k={var_mapping_k}, op=\\circled{{{op_letter}}})"
    else:
        if ("mlp" in q and q[q.find("mlp"):] not in converted_mlp) and not show_logits_for_unconverted_mlp:
            code_line = f"select(q={var_mapping_q}, k={var_mapping_k}, op=(q==k))"
            is_after_mlp = True
        elif ("mlp" in k and k[k.find("mlp"):] not in converted_mlp) and not show_logits_for_unconverted_mlp:
            code_line = f"select(q={var_mapping_q}, k={var_mapping_k}, op=(q==k))"
            is_after_mlp = True
        else:
            code_line = f"select(q={var_mapping_q}, k={var_mapping_k}, op=\\circled{{{op_letter}}})"
        
    if var_mapping_q is not None and var_mapping_k is not None and var_mapping_q != "bias":
        code_line = code_line + f"\t# layer {layer_idx} head {head_idx}"
    return select_name, code_line, is_after_mlp
        
def convert_config_to_code_qk_pruned(config, split_mlps, converted_mlp, convert_to_primitives = False,
                                     primitives_config = None, return_var_mapping=False, use_code_with_ops=True,
                                     show_logits_for_unconverted_mlp=False):
    assert use_code_with_ops
    if convert_to_primitives:
        config, primitives_config = remove_unnecessary_primitives(config, primitives_config)
    code = []
    selector_to_config = {}
    counter = {"s": 1, "a": 1, "m": 1}
    var_mapping = {"wpe": "pos", "wte": "token"}    # component name : variable name
    count_heatmaps = 0
    for layer_idx in range(len(config)-1):
        layer_idx = str(layer_idx)
        for head_idx in config[layer_idx]["v"]:
            select_names = []
            for prod in config[layer_idx]["qk"][head_idx] + config[layer_idx]["k"][head_idx]:
                select_name = "s" + str(counter["s"])
                if type(prod) == tuple or type(prod) == list:
                    q, k = prod
                    if convert_to_primitives:
                        primitive = [p for p in primitives_config[layer_idx][head_idx]["qk_interactions"] if p["activation_name_to_keep_q"] == q and p["activation_name_to_keep_k"] == k]
                        assert len(primitive) == 1
                        primitive = primitive[0]
                        if "predefined_primitive" in primitive:
                            select_name, code_line = get_code_line_given_predefined_primitives_with_ops(select_name, counter, primitive,
                                                                                               ATTENTION_ALL_PRIMITIVES,
                                                                                               var_mapping[q], var_mapping[k], None,
                                                                                               layer_idx, head_idx)
                            code.append( f"{select_name} = {code_line}" )
                        elif 'primitive' in primitive and primitive['primitive'] is not None and 'projection' in primitive['primitive']:
                                select_name, code_line, is_after_mlp = get_code_line_given_replacement_matrices_with_ops(select_name, count_heatmaps,
                                                                                                               var_mapping[q], var_mapping[k], None,
                                                                                                               layer_idx, head_idx,
                                                                                                               q, k, None, converted_mlp,
                                                                                                               show_logits_for_unconverted_mlp=show_logits_for_unconverted_mlp)
                                count_heatmaps -= int(is_after_mlp)
                                code.append( f"{select_name} = {code_line}" )
                        else:
                            if 'primitive' in primitive:
                                primitive_name = primitive['primitive']
                            else:
                                primitive_name = "select"
                            code.append( f"{select_name} = {primitive_name}(q={var_mapping[q]}, k={var_mapping[k]}) \t  # layer {layer_idx} head {head_idx}" )
                        count_heatmaps += 1
                    else:
                        code.append( f"{select_name} = select(q={var_mapping[q]}, k={var_mapping[k]}) \t  # layer {layer_idx} head {head_idx}" )
                elif type(prod) == str:
                    k = prod
                    if convert_to_primitives:
                        primitive = [p for p in primitives_config[layer_idx][head_idx]["k_interactions"] if p["activation_name_to_keep_k"] == k]
                        assert len(primitive) == 1
                        primitive = primitive[0]
                        if "predefined_primitive" in primitive:
                            select_name, code_line = get_code_line_given_predefined_primitives_with_ops(select_name, counter, primitive,
                                                                                            ATTENTION_CONST_ALL_PRIMITIVES,
                                                                                            "bias", var_mapping[k], None,
                                                                                            layer_idx, head_idx)
                        
                            code.append(f"{select_name} = {code_line}")
                        elif 'primitive' in primitive and primitive['primitive'] is not None and 'projection' in primitive['primitive']:
                            select_name, code_line, is_after_mlp = get_code_line_given_replacement_matrices_with_ops(select_name, count_heatmaps,
                                                                                                                "bias", var_mapping[k], None,
                                                                                                                layer_idx, head_idx,
                                                                                                                "bias", k, None, converted_mlp,
                                                                                                                show_logits_for_unconverted_mlp=show_logits_for_unconverted_mlp)
                            count_heatmaps -= int(is_after_mlp)
                            code.append( f"{select_name} = {code_line}" )
                        else:
                            if 'primitive' in primitive:
                                primitive_name = primitive['primitive']
                            else:
                                primitive_name = "select"
                            code.append( f"{select_name} = {primitive_name}(k={var_mapping[k]}) \t  # layer {layer_idx} head {head_idx}" )
                        count_heatmaps += 1
                    else:
                        code.append( f"{select_name} = select(k={var_mapping[k]}) \t  # layer {layer_idx} head {head_idx}" )
                else:
                    raise RuntimeError(prod)
                selector_to_config[select_name] = (int(layer_idx), int(head_idx), prod)
                select_names.append(select_name)
                counter["s"] += 1

            if len(select_names) == 0:
                select_names = "[]"
            else:
                select_names = "+".join(select_names)
            for v in config[layer_idx]["v"][head_idx]:
                attn_out_name = "a" + str(counter["a"])
                code.append( f"{attn_out_name} = aggregate(s={select_names}, v={var_mapping[v]}) \t  # layer {layer_idx} head {head_idx}" )
                var_mapping[f"attn_output-{layer_idx}-{head_idx}-{v}"] = attn_out_name
                counter["a"] += 1
        
        if config[layer_idx]["mlp"]:
            if not split_mlps:
                path = f"mlp-{layer_idx}"
                if converted_mlp is not None and path in converted_mlp:
                    # primitive for multi source mlp converted
                    primitive = converted_mlp[path][0]
                    if primitive[0] == "keep_one":
                        mlp_inp = config[layer_idx]["mlp"][primitive[1]]
                        var_mapping[f"mlp-{layer_idx}"] = var_mapping[mlp_inp]
                        continue
                    connect_symbol, op_name = map_primitive_to_names(primitive)
                    mlp_inputs = list(map(lambda x: var_mapping[x], config[layer_idx]["mlp"]))
                    code.append( f"{connect_symbol.join(mlp_inputs)} = {op_name}({', '.join(mlp_inputs)}) \t  # layer {layer_idx} mlp" )
                    var_mapping[f"mlp-{layer_idx}"] = f"{connect_symbol.join(mlp_inputs)}"
                else:
                    # old default
                    mlp_out_name = "m" + str(counter["m"])
                    mlp_inputs = "+".join(map(lambda x: var_mapping[x], config[layer_idx]["mlp"]))
                    code.append( f"{mlp_out_name} = mlp({mlp_inputs}) \t  # layer {layer_idx} mlp" )
                    var_mapping[f"mlp-{layer_idx}"] = mlp_out_name
                    counter["m"] += 1
            else:
                for inp in config[layer_idx]["mlp"]:
                    path = f"mlp-{layer_idx}-{inp}"
                    if path in converted_mlp:
                        primitive = converted_mlp[path][0]
                    else:
                        primitive = "unknown"
                    if primitive == "no_op":
                        var_mapping[f"mlp-{layer_idx}-{inp}"] = var_mapping[inp]
                        continue
                    var_prefix, op_name = map_primitive_to_names(primitive)
                    code.append( f"{var_prefix}_{var_mapping[inp]} = {op_name}({var_mapping[inp]}) \t  # layer {layer_idx} mlp" )
                    var_mapping[f"mlp-{layer_idx}-{inp}"] = f"{var_prefix}_{var_mapping[inp]}"
    
    distribution_to_config = {}
    counter["l"] = 1
    i = 0
    for i, inp in enumerate(config["lm_head"], start=1):
        var_name = f"logits{i}"
        if convert_to_primitives:
            primitive = [p for p in primitives_config["lm_head"] if p["activation_name_to_keep"] == inp]
            assert len(primitive) == 1, len(primitive)
            primitive = primitive[0]
            if "predefined_primitive" in primitive:
                var_name, code_line = get_code_line_given_predefined_primitives_with_ops(var_name, counter, primitive,
                                                                            LOGITS_ALL_PRIMITIVES,
                                                                            None, None, var_mapping[inp],
                                                                            None, None)
                code.append( f"{var_name} = {code_line}" )
            elif 'primitive' in primitive and primitive['primitive'] is not None and 'projection' in primitive['primitive']:
                var_name, code_line, is_after_mlp = get_code_line_given_replacement_matrices_with_ops(var_name, count_heatmaps,
                                                                                            None, None, var_mapping[inp],
                                                                                            None, None,
                                                                                            None, None, inp, converted_mlp,
                                                                                            show_logits_for_unconverted_mlp=show_logits_for_unconverted_mlp)
                count_heatmaps -= int(is_after_mlp)
                code.append( f"{var_name} = {code_line}" )
            else:
                if 'primitive' in primitive and primitive['primitive'] is not None:
                    primitive_name = primitive['primitive']
                else:
                    primitive_name = "not_converted_proj_to_vocab"
                code.append( f"{var_name} = {primitive_name}({var_mapping[inp]})" )
            count_heatmaps += 1
        else:
            code.append( f"{var_name} = proj_to_vocab({var_mapping[inp]})" )
        distribution_to_config[var_name] = inp
        counter["l"] += 1
    if convert_to_primitives:
        primitive = [p for p in primitives_config["lm_head"] if p["activation_name_to_keep"] == "vocab_bias"]
        assert len(primitive) <= 1, len(primitive)
        if len(primitive) == 1:
            var_name = f"logits{i+1}"
            primitive = primitive[0]
            if "predefined_primitive" in primitive:
                var_name, code_line = get_code_line_given_predefined_primitives_with_ops(var_name, counter, primitive,
                                                                                    LOGITS_CONST_ALL_PRIMITIVES,
                                                                                    None, None, "bias",
                                                                                    None, None)
                code.append( f"{var_name} = {code_line}" )
            elif 'primitive' in primitive and primitive['primitive'] is not None and 'projection' in primitive['primitive']:
                var_name, code_line, is_after_mlp = get_code_line_given_replacement_matrices_with_ops(var_name, count_heatmaps,
                                                                                        None, None, "bias",
                                                                                        None, None,
                                                                                        None, None, "bias", converted_mlp,
                                                                                        show_logits_for_unconverted_mlp=show_logits_for_unconverted_mlp)
                count_heatmaps -= int(is_after_mlp)
                code.append( f"{var_name} = {code_line}" )
            else:
                if 'primitive' in primitive and primitive['primitive'] is not None:
                    primitive_name = primitive['primitive']
                else:
                    primitive_name = "not_converted_proj_to_vocab"
                code.append( f"{var_name} = {primitive_name}(bias)" )
            distribution_to_config[var_name] = "vocab_bias"
        count_heatmaps += 1
    else:
        var_name = f"logits{i+1}"
        code.append( f"{var_name} = proj_to_vocab(bias)" )
        distribution_to_config[var_name] = "vocab_bias"

    temp = '+\n\t\t\t'.join(list(distribution_to_config.keys()))
    code.append( f"prediction = softmax({temp})" )

    selector_to_config.update(distribution_to_config)
    if return_var_mapping:
        return code, selector_to_config, var_mapping
    return code, selector_to_config

def convert_keys_to_int(d):
    """Recursively convert string keys that are integer-like to int."""
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = int(k)
            except (ValueError, TypeError):
                new_key = k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    else:
        return d
    
if __name__ == "__main__":
    import json
    config = {0: {'k': {0: ['wpe']}, 'q': {0: ['wpe']}, 'v': {0: ['wte']}, 'mlp': ['wte', 'attn_output-0-0-wte']}, 1: {'k': {0: ['wte', 'attn_output-0-0-wte', 'mlp-0']}, 'q': {0: ['wte', 'attn_output-0-0-wte', 'mlp-0']}, 'v': {0: ['attn_output-0-0-wte']}, 'mlp': []}, 'lm_head': ['attn_output-1-0-attn_output-0-0-wte']}
    config = json.loads(json.dumps(config))

    # convert_config_to_code_full_paths(config)
    # convert_config_to_code_full_paths(config, plot_heatmaps=True, tokenizer_max_test_length=150, model_name="unique_reverse-2l1h64d_dropout00")
    config = {0: {'k': {0: ['wpe']}, 'v': {0: ['wte']}, 'mlp': ['wte', 'attn_output-0-0-wte'], 'qk': {0: [('wpe', 'wpe')]}}, 1: {'k': {0: ['attn_output-0-0-wte']}, 'v': {0: ['attn_output-0-0-wte']}, 'mlp': [], 'qk': {0: [('wte', 'wte'), ('attn_output-0-0-wte', 'attn_output-0-0-wte'), ('mlp-0', 'attn_output-0-0-wte'), ('mlp-0', 'mlp-0')]}}, 'lm_head': ['attn_output-1-0-attn_output-0-0-wte']}
    config = json.loads(json.dumps(config))
    code, mapping = convert_config_to_code_qk_pruned(config)
    print("\n".join(code))
    print(mapping)