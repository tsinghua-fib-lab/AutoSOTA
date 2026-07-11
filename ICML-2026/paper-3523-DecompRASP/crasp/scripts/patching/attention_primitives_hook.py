import torch
from itertools import product

from primitives_helpers import plot_and_save_primitives_matrices, is_token_dim_activation, is_cartesian_act
from show_heatmap import get_product_for_one_side_for_head, get_product_for_one_side, get_product_for_one_side_multi_source
from plot_func_for_main_paper import plot_and_save_primitives_matrices_for_main_paper

def primitives_attention_forward(self, module, layer=None, primitives=None, tokenizer=None, converted_mlp=None, oa_vecs=None):
    assert self.current_layer == layer
    assert primitives is not None
    assert hasattr(self, "activation_name_to_keep") and self.activation_name_to_keep is not None

    num_heads = module.num_heads
    head_dim = module.head_dim
    split_size = module.split_size
    batch_size, seq_len, d_model = self.activations["wte"].size()
    device = self.activations["wte"].device

    # to compute v
    input_activations = []
    for head in range(num_heads):
        if self.activation_name_to_keep[head] is not None:
            activation_name = self.activation_name_to_keep[head]
            input_act = self.activations[activation_name]

            LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "v", head, activation_name)]].exp()
            input_act = self.linear_layer_norm(self.ln_1[layer], input_act, LN_var, bias=False)    # avoid adding bias term multiple times, and leave this to output oa vec
        else:
            input_act = torch.zeros_like(self.activations["wte"])
        input_activations.append(input_act)

    input_activations = torch.stack(input_activations)

    output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
                            module.c_attn.weight[:, split_size*2:].view(d_model, num_heads, head_dim).transpose(0, 1)
    value_states = output_activations.transpose(0, 1).contiguous().view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
    if self.save_matrices and ("attn", "pos-tok") not in self.saved_matrices:
        save_path = f"{self.save_matrices_path}/{self.model_name}"
        ticks_x_example = tokenizer.convert_ids_to_tokens(self.wte_inputs[0].tolist())
        self.saved_matrices.add(("attn", "pos-tok"))
        toks = torch.nn.functional.one_hot(self.wte_inputs[0], num_classes=self.model.config.vocab_size).float()
        poss = torch.nn.functional.one_hot(self.wpe_inputs[0], num_classes=self.model.config.max_position_embeddings).float()
        plot_and_save_primitives_matrices(toks.T,
                f"{save_path}/pos-tok/tok.pdf",
                ticks_x=ticks_x_example, ticks_y=[t for t in tokenizer.vocab])
        plot_and_save_primitives_matrices(poss.T,
                f"{save_path}/pos-tok/pos.pdf",
                ticks_x=ticks_x_example, ticks_y=None)
        plot_and_save_primitives_matrices(toks.T,
                f"{save_path}/pos-tok/tok.png",
                ticks_x=ticks_x_example, ticks_y=[t for t in tokenizer.vocab])
        plot_and_save_primitives_matrices(poss.T,
                f"{save_path}/pos-tok/pos.png",
                ticks_x=ticks_x_example, ticks_y=None)
        if hasattr(self, "plot_for_main_paper") and self.plot_for_main_paper:
            plot_and_save_primitives_matrices_for_main_paper(poss.T,
                f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-pos.pdf",
                ticks_x=ticks_x_example, ticks_y=None)
            plot_and_save_primitives_matrices_for_main_paper(poss.T,
                f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-pos.png",
                ticks_x=ticks_x_example, ticks_y=None)
            plot_and_save_primitives_matrices_for_main_paper(toks.T,
                f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-tok.pdf",
                ticks_x=ticks_x_example, ticks_y=[t for t in tokenizer.vocab])
            plot_and_save_primitives_matrices_for_main_paper(toks.T,
                f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-tok.png",
                ticks_x=ticks_x_example, ticks_y=[t for t in tokenizer.vocab])
    for head in range(num_heads):
        for interaction, interaction_primitive in primitives[layer][head].items():
            activation_name_to_keep  = {}
            activation_name_to_keep["q"] = interaction.activation_name_to_keep_q
            activation_name_to_keep["k"] = interaction.activation_name_to_keep_k
            save_path = f"{self.save_matrices_path}/{self.model_name}/{layer}-{head}"

            ticks_x_example = tokenizer.convert_ids_to_tokens(self.wte_inputs[0].tolist())
            ticks_y_example = tokenizer.convert_ids_to_tokens(self.wte_inputs[0].tolist())
            if activation_name_to_keep["q"] is None:
                ticks_y_example = None

            if activation_name_to_keep["q"] is not None:
                if is_token_dim_activation(activation_name_to_keep["q"], converted_mlp):
                    ticks_y = [t for t in tokenizer.vocab]
                elif is_cartesian_act(activation_name_to_keep["q"], converted_mlp, self.config)[0]:
                    repeats = is_cartesian_act(activation_name_to_keep["q"], converted_mlp, self.config)[1]
                    ticks_y = ["-".join(t) for t in product(tokenizer.vocab, repeat=repeats)]
                else:
                    ticks_y = None
            else:
                ticks_y = None
            if is_token_dim_activation(activation_name_to_keep["k"], converted_mlp):
                ticks_x = [t for t in tokenizer.vocab]
            elif is_cartesian_act(activation_name_to_keep["k"], converted_mlp, self.config)[0]:
                repeats = is_cartesian_act(activation_name_to_keep["k"], converted_mlp, self.config)[1]
                ticks_x = ["-".join(t) for t in product(tokenizer.vocab, repeat=repeats)]
            else:
                ticks_x = None

            if activation_name_to_keep["q"] is not None:
                activation_name_to_save = f"{activation_name_to_keep['q']}-{activation_name_to_keep['k']}"
            else:
                activation_name_to_save = f"bias-{activation_name_to_keep['k']}"
            if interaction_primitive is None or (interaction_primitive.primitive is None and interaction_primitive.replacement_matrix is None): 

                products = {}
                if activation_name_to_keep["q"] is not None:
                    qk_types = ["q", "k"]
                else:
                    qk_types = ["k"]
                for qk_type in qk_types:
                    products[qk_type] = get_product_for_one_side_for_head(self, oa_vecs, converted_mlp,
                                                                            layer, head, qk_type, activation_name_to_keep[qk_type], self.wte_inputs.squeeze(0), self.wpe_inputs.squeeze(0))
                act_soft = {}
                if activation_name_to_keep["q"] is not None:
                    act_soft["q"] = products["q"][0]
                    indep_prod = products["q"][1] @ products["k"][1].transpose(-1, -2)
                else:
                    alpha = oa_vecs.q_bias_term.data[oa_vecs.to_q_bias[(layer, head, activation_name_to_keep["k"])]].unsqueeze(0)
                    indep_prod = alpha @ products["k"][1].transpose(-1, -2)
                act_soft["k"] = products["k"][0]
                if "q" not in act_soft:
                    first_term = (indep_prod.squeeze().unsqueeze(0) @ act_soft["k"].transpose(-1, -2))
                else:
                    first_term = act_soft["q"] @ indep_prod @ act_soft["k"].transpose(-1, -2)

                if self.save_matrices and ("attn", "original-example", layer, head, activation_name_to_save) not in self.saved_matrices:
                    self.saved_matrices.add(("attn", "original-example", layer, head, activation_name_to_save))
                    plot_and_save_primitives_matrices(first_term[0],
                            f"{save_path}/original-example/{activation_name_to_save}.pdf",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example)
                    plot_and_save_primitives_matrices(first_term[0],
                            f"{save_path}/original-example/{activation_name_to_save}.png",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example)
                attn_weights[:, head, :, :] += first_term
            else:
                products = {}
                if activation_name_to_keep["q"] is not None:
                    qk_types = ["q", "k"]
                else:
                    qk_types = ["k"]
                for qk_type in qk_types:
                    products[qk_type] = get_product_for_one_side_for_head(self, oa_vecs, converted_mlp,
                                                                            layer, head, qk_type, activation_name_to_keep[qk_type], self.wte_inputs.squeeze(0), self.wpe_inputs.squeeze(0))
                act_soft = {}
                if activation_name_to_keep["q"] is not None:
                    act_soft["q"] = products["q"][0]
                    indep_prod = products["q"][1] @ products["k"][1].transpose(-1, -2)
                else:
                    alpha = oa_vecs.q_bias_term.data[oa_vecs.to_q_bias[(layer, head, activation_name_to_keep["k"])]].unsqueeze(0)
                    indep_prod = alpha @ products["k"][1].transpose(-1, -2)
                if self.save_matrices and ("attn", "original", layer, head, activation_name_to_save) not in self.saved_matrices:
                    self.saved_matrices.add(("attn", "original", layer, head, activation_name_to_save))
                    plot_and_save_primitives_matrices(indep_prod,
                            f"{save_path}/original-matrices/{activation_name_to_save}.pdf", ticks_x=ticks_x, ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(indep_prod,
                            f"{save_path}/original-matrices/{activation_name_to_save}.png", ticks_x=ticks_x, ticks_y=ticks_y)
                act_soft["k"] = products["k"][0]
                if interaction_primitive.primitive is not None:
                    special_tokens = [
                        tokenizer.sep_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
                    ]
                    if activation_name_to_keep["q"] is not None:
                        primitive_matrix = interaction_primitive.primitive.contruct_matrix(indep_prod.shape[0], indep_prod.shape[1], tokenizer).to(attn_weights.dtype).to(attn_weights.device)
                        if interaction_primitive.special_primitive is not None:
                            special_primitive_matrix = interaction_primitive.special_primitive.contruct_matrix(indep_prod.shape[0], indep_prod.shape[1], tokenizer).to(attn_weights.dtype).to(attn_weights.device)
                            primitive_matrix[special_tokens, :] = special_primitive_matrix[special_tokens, :]
                    else:
                        primitive_matrix = interaction_primitive.primitive.contruct_matrix(indep_prod.shape[1], tokenizer).to(attn_weights.dtype).to(attn_weights.device)
                        assert interaction_primitive.special_primitive is None
                    if type(interaction_primitive.scaling_factor_primitive) == str and interaction_primitive.scaling_factor_primitive == "inf":
                        mask_value = torch.finfo(primitive_matrix.dtype).max
                        primitive_matrix = torch.where(primitive_matrix > 0, mask_value, 0.)
                    else:
                        primitive_matrix = interaction_primitive.scaling_factor_primitive * primitive_matrix

                elif interaction_primitive.replacement_matrix is not None:
                    primitive_matrix = interaction_primitive.replacement_matrix.get_matrix()

                if self.save_matrices and ("attn", "primitives", layer, head, activation_name_to_save) not in self.saved_matrices:
                    self.saved_matrices.add(("attn", "primitives", layer, head, activation_name_to_save))
                    plot_and_save_primitives_matrices(primitive_matrix,
                            f"{save_path}/primitives-matrices/{activation_name_to_save}.pdf", ticks_x=ticks_x, ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(primitive_matrix,
                            f"{save_path}/primitives-matrices/{activation_name_to_save}.png", ticks_x=ticks_x, ticks_y=ticks_y)
                    if hasattr(self, "plot_for_main_paper") and self.plot_for_main_paper:
                        plot_and_save_primitives_matrices_for_main_paper(primitive_matrix,
                            f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/primitive-{layer}-{head}-{activation_name_to_save}.pdf",
                            ticks_x=ticks_x, ticks_y=ticks_y)
                        plot_and_save_primitives_matrices_for_main_paper(primitive_matrix,
                            f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/primitive-{layer}-{head}-{activation_name_to_save}.png",
                            ticks_x=ticks_x, ticks_y=ticks_y)
                    
                if activation_name_to_keep["q"] is not None:
                    prod = act_soft["q"] @ primitive_matrix @ act_soft["k"].transpose(-1, -2)
                else:
                    prod = (primitive_matrix.squeeze().unsqueeze(0) @ act_soft["k"].transpose(-1, -2))
                    
                if len(prod.shape) == 2:
                    prod = prod.unsqueeze(0)

                if self.save_matrices and ("attn", "primitives-example", layer, head, activation_name_to_save) not in self.saved_matrices:
                    self.saved_matrices.add(("attn", "primitives-example", layer, head, activation_name_to_save))
                    add_causal_mask = True if "bias" not in activation_name_to_save else False
                    plot_and_save_primitives_matrices(prod[0],
                            f"{save_path}/primitives-example/{activation_name_to_save}.pdf",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example, add_causal_mask=add_causal_mask)
                    plot_and_save_primitives_matrices(prod[0],
                            f"{save_path}/primitives-example/{activation_name_to_save}.png",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example, add_causal_mask=add_causal_mask)
                    if hasattr(self, "plot_for_main_paper") and self.plot_for_main_paper:
                        plot_and_save_primitives_matrices_for_main_paper(prod[0],
                            f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-{layer}-{head}-{activation_name_to_save}.pdf",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example, add_causal_mask=add_causal_mask)
                        plot_and_save_primitives_matrices_for_main_paper(prod[0],
                            f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-{layer}-{head}-{activation_name_to_save}.png",
                            ticks_x=ticks_x_example, ticks_y=ticks_y_example, add_causal_mask=add_causal_mask)
                attn_weights[:, head, :, :] += prod

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value_states.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # if only "normal" attention layer implements causal mask
    query_length, key_length = seq_len, seq_len # query_states.size(-2), key_states.size(-2)
    causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    # attn_weights = torch.where(mask_for_pruning, attn_weights.to(attn_weights.dtype), mask_value)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value_states.dtype)
    attn_weights = module.attn_dropout(attn_weights)
    self.activations[f"attn_weights-{layer}"] = attn_weights.detach()


    for head in range(num_heads):
        if self.activation_name_to_keep[head] is not None:
            activation_name = self.activation_name_to_keep[head]
            get_prod_func = get_product_for_one_side if hasattr(oa_vecs, "MLPs") else get_product_for_one_side_multi_source
            dep_v, _ = get_prod_func(self, oa_vecs, converted_mlp,
                                                activation_name, self.input_ids, self.position_ids)
            
            if dep_v.dim() < 3:
                dep_v = dep_v.unsqueeze(0)

            
            assert attn_weights.dim() == 4 and dep_v.dim() == 3, (attn_weights.dim(), dep_v.dim())
            
            agg_var = torch.matmul(attn_weights[0, head, :, :], dep_v[0, :, :])

            ticks_example = tokenizer.convert_ids_to_tokens(self.wte_inputs[0].tolist())
            ticks_activation = None
            if is_token_dim_activation(activation_name, converted_mlp):
                ticks_activation = [t for t in tokenizer.vocab]
            elif is_cartesian_act(activation_name, converted_mlp, self.config)[0]:
                ticks_activation = is_cartesian_act(activation_name, converted_mlp, self.config)[1]
                ticks_activation = ["-".join(t) for t in product(tokenizer.vocab, repeat=repeats)]
            else:
                ticks_activation = None

            save_path = f"{self.save_matrices_path}/{self.model_name}/{layer}-{head}"
            activation_name = f"attn_output-{layer}-{head}-" + activation_name
            if self.save_matrices and ("attn", "primitives-example", layer, head, f"aggregation-{activation_name}") not in self.saved_matrices:
                self.saved_matrices.add(("attn", "primitives-example", layer, head, f"aggregation-{activation_name}"))
                plot_and_save_primitives_matrices(agg_var.T,
                        f"{save_path}/primitives-example/aggregation-{activation_name}.pdf",
                        ticks_x=ticks_example, ticks_y=ticks_activation)
                plot_and_save_primitives_matrices(agg_var.T,
                        f"{save_path}/primitives-example/aggregation-{activation_name}.png",
                        ticks_x=ticks_example, ticks_y=ticks_activation)
                if hasattr(self, "plot_for_main_paper") and self.plot_for_main_paper:
                    plot_and_save_primitives_matrices_for_main_paper(agg_var.T,
                        f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-{layer}-{head}-aggregation-{activation_name}.pdf",
                        ticks_x=ticks_example, ticks_y=ticks_activation)
                    plot_and_save_primitives_matrices_for_main_paper(agg_var.T,
                        f"{self.save_matrices_path}/plots_for_main_paper/{self.model_name}/example-{layer}-{head}-aggregation-{activation_name}.png",
                        ticks_x=ticks_example, ticks_y=ticks_activation)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)

    attention_by_head = attn_output.contiguous().view(batch_size * seq_len, num_heads, head_dim)
    attention_by_head_output = attention_by_head.transpose(0, 1) @ module.c_proj.weight.view(num_heads, head_dim, d_model)  # num_head, bz*seq_len, d_model
    attention_by_head_output = module.resid_dropout(attention_by_head_output)
    attention_by_head_output = attention_by_head_output.view(num_heads, batch_size, seq_len, d_model).unbind(dim=0)

    for head, attn in enumerate(attention_by_head_output):
        if self.activation_name_to_keep[head] is not None:
            self.activations[f"attn_output-{layer}-{head}-{self.activation_name_to_keep[head]}"] = attn

    return None