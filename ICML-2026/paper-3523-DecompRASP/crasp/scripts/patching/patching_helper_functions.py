def get_full_possible_config_one_step_paths(num_layers, num_heads):
    full_possible_config = {
        layer: {
            "k": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer)]
                for head in range(num_heads[layer])
            },
            "q": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer)]
                for head in range(num_heads[layer])
            },
            "v": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer)]
                for head in range(num_heads[layer])
            },
        }
        for layer in range(num_layers)
    }
    for layer in range(num_layers):
        full_possible_config[layer].update({
            "mlp": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer + 1) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer + 1)]
        })
        full_possible_config[layer].update({
            "ln_1": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer)]
        })
        full_possible_config[layer].update({
            "ln_2": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer + 1) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(layer)] + [f"attention_bias-{l}" for l in range(layer + 1)]
        })
    full_possible_config.update({
        "lm_head": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(num_layers) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(num_layers)] + [f"attention_bias-{l}" for l in range(num_layers)],
        "ln_f": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(num_layers) for h in range(num_heads[l])] + [f"mlp-{l}" for l in range(num_layers)] + [f"attention_bias-{l}" for l in range(num_layers)]
    })
    return full_possible_config

def get_full_possible_config_for_pruning(num_heads_per_layer):
    full_config = {
        layer: {
            "k": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "q": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "v": {
                head: ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer) for h in range(num_heads_per_layer[l])] + [f"mlp-{l}" for l in range(layer)]
                for head in range(num_heads_per_layer[layer])
            },
            "mlp": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(layer + 1) for h in range(num_heads_per_layer[l])] + [f"mlp-{l}" for l in range(layer)] 
        }
        for layer in range(len(num_heads_per_layer))
    }
    full_config.update({
        "lm_head": ["wte", "wpe"] + [f"attn_output-{l}-{h}" for l in range(len(num_heads_per_layer)) for h in range(num_heads_per_layer[l])] + [f"mlp-{l}" for l in range(len(num_heads_per_layer))]
    })
    return full_config


def convert_one_step_config_to_full_paths_config(current_config, num_layers, num_heads):
    for layer in range(num_layers):
        for head in range(num_heads[layer]):
            for attn_act in ["k", "q", "v"]:
                current_config[layer][attn_act][head] = [
                    act for act in ["wte", "wpe"] if act in current_config[layer][attn_act][head]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(layer)
                    for h in range(num_heads[l])
                    for prev_path in current_config[l]["v"][h]
                if f"attn_output-{l}-{h}" in current_config[layer][attn_act][head]] + \
                [f"attention_bias-{l}" for l in range(layer) if f"attention_bias-{l}" in current_config[layer][attn_act][head]] + \
                [f"mlp-{l}"
                    for l in range(layer)
                    if f"mlp-{l}" in current_config[layer][attn_act][head]]
        for layer_level_act in ["mlp", "ln_1", "ln_2"]:
            current_config[layer][layer_level_act] = [
                    act for act in ["wte", "wpe"] if act in current_config[layer][layer_level_act]
                    ] + \
                    [f"attn_output-{l}-{h}-{prev_path}"
                        for l in range(layer + 1)
                        for h in range(num_heads[l])
                        for prev_path in current_config[l]["v"][h]
                        if f"attn_output-{l}-{h}" in current_config[layer][layer_level_act]] + \
                    [f"attention_bias-{l}" for l in range(layer + 1) if f"attention_bias-{l}" in current_config[layer][layer_level_act]] + \
                    [f"mlp-{l}"
                    for l in range(layer)
                    if f"mlp-{l}" in current_config[layer][layer_level_act]]
    for model_level_act in ["lm_head", "ln_f"]:
        current_config[model_level_act] = [
                    act for act in ["wte", "wpe"] if act in current_config[model_level_act]
                    ] + \
                    [f"attn_output-{l}-{h}-{prev_path}"
                        for l in range(num_layers)
                        for h in range(num_heads[l])
                        for prev_path in current_config[l]["v"][h]
                        if f"attn_output-{l}-{h}" in current_config[model_level_act]] + \
                    [f"attention_bias-{l}" for l in range(num_layers) if f"attention_bias-{l}" in current_config[model_level_act]] + \
                    [f"mlp-{l}"
                    for l in range(num_layers)
                    if f"mlp-{l}" in current_config[model_level_act]]
    return current_config

def convert_full_paths_config_to_prune_inside_kq_config(current_config, num_layers, num_heads):
    for layer in range(num_layers):
        current_config[layer]["qk"] = {}
        for head in range(num_heads[layer]):
            current_config[layer]["qk"][head] = [
                (path_in_q, path_in_k)
                for path_in_q in current_config[layer]["q"][head]
                for path_in_k in current_config[layer]["k"][head]
            ]
        del current_config[layer]["q"]
        del current_config[layer]["k"]
    return current_config
