import torch
from itertools import product

from primitives_helpers import plot_and_save_primitives_matrices, is_token_dim_activation, is_cartesian_act
from plot_func_for_main_paper import plot_and_save_primitives_matrices_for_main_paper

from show_heatmap import get_product_for_one_side_for_unembed

def lm_head_hook(module, input, output, hooked_model, primitives=None, tokenizer=None, oa_vecs=None, converted_mlp=None):
    output = None
    for interaction, primitive in primitives["lm_head"].items():
        activation_name_to_keep = interaction.activation_name_to_keep

        ticks_example = tokenizer.convert_ids_to_tokens(hooked_model.wte_inputs[0].tolist())
        if activation_name_to_keep == "vocab_bias":
            ticks_example = None

        if activation_name_to_keep == "vocab_bias":
            ticks_y = None
        else:
            if is_token_dim_activation(activation_name_to_keep, converted_mlp):
                ticks_y = [t for t in tokenizer.vocab]
            elif is_cartesian_act(activation_name_to_keep, converted_mlp, hooked_model.config)[0]:
                repeats = is_cartesian_act(activation_name_to_keep, converted_mlp, hooked_model.config)[1]
                ticks_y = ["-".join(t) for t in product(tokenizer.vocab, repeat=repeats)]
            else:
                ticks_y = None

        if primitive is None or (primitive.primitive is None and primitive.replacement_matrix is None):
            dep_prod, indep_prod = get_product_for_one_side_for_unembed(hooked_model, oa_vecs, converted_mlp,
                                                                        activation_name_to_keep,
                                                                        hooked_model.input_ids.to(hooked_model.device), hooked_model.position_ids.to(hooked_model.device))
            
            if activation_name_to_keep == "vocab_bias":
                prod = indep_prod
            else:
                prod = dep_prod @ indep_prod

            if len(prod.shape) == 2:
                prod = prod.unsqueeze(0)

            if hooked_model.save_matrices and ("logits", "original", activation_name_to_keep) not in hooked_model.saved_matrices:
                hooked_model.saved_matrices.add(("logits", "original", activation_name_to_keep))
                if activation_name_to_keep == "vocab_bias":
                    plot_and_save_primitives_matrices(prod.squeeze(),
                        f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-matrices/bias.pdf",
                        ticks_x=[t for t in tokenizer.vocab],
                        ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-example/bias.pdf",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_example)
                    plot_and_save_primitives_matrices(prod.squeeze(),
                        f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-matrices/bias.png",
                        ticks_x=[t for t in tokenizer.vocab],
                        ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-example/bias.png",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_example)
                else:
                    _, indep_prod = get_product_for_one_side_for_unembed(hooked_model, oa_vecs, converted_mlp,
                                                                                activation_name_to_keep,
                                                                                hooked_model.input_ids.to(hooked_model.device), hooked_model.position_ids.to(hooked_model.device))
                    plot_and_save_primitives_matrices(indep_prod,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-matrices/{activation_name_to_keep}.pdf",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod[0].T,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-example/{activation_name_to_keep}.pdf",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    plot_and_save_primitives_matrices(indep_prod,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-matrices/{activation_name_to_keep}.png",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod[0].T,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/original-example/{activation_name_to_keep}.png",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
        else:
            if activation_name_to_keep == "vocab_bias":
                if primitive.replacement_matrix is not None:
                    prod = primitive.replacement_matrix.get_matrix()
                elif primitive.primitive is not None:
                    primitive_matrix = primitive.primitive.contruct_matrix(module.weight.shape[0], tokenizer).to(module.weight.dtype).to(module.weight.device)
                    assert primitive.special_primitive is None
                    if type(primitive.scaling_factor_primitive) == str and primitive.scaling_factor_primitive == "inf":
                        mask_value = torch.finfo(primitive_matrix.dtype).max
                        primitive_matrix = torch.where(primitive_matrix > 0, mask_value, 0.)
                    else:
                        primitive_matrix = primitive.scaling_factor_primitive * primitive_matrix
                    prod = primitive_matrix

                if hooked_model.save_matrices and ("logits", "primitives", activation_name_to_keep) not in hooked_model.saved_matrices:
                    hooked_model.saved_matrices.add(("logits", "primitives", activation_name_to_keep))
                    plot_and_save_primitives_matrices(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-matrices/bias.pdf",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod.squeeze().reshape(-1, 1).repeat(1, len(hooked_model.wte_inputs[0].tolist())),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-example/bias.pdf",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    plot_and_save_primitives_matrices(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-matrices/bias.png",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod.squeeze().reshape(-1, 1).repeat(1, len(hooked_model.wte_inputs[0].tolist())),
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-example/bias.png",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    if hasattr(hooked_model, "plot_for_main_paper") and hooked_model.plot_for_main_paper:
                        plot_and_save_primitives_matrices_for_main_paper(prod.squeeze().reshape(-1, 1).repeat(1, len(hooked_model.wte_inputs[0].tolist())),
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/example-lm_head-bias.pdf",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                        plot_and_save_primitives_matrices_for_main_paper(prod.squeeze().reshape(-1, 1).repeat(1, len(hooked_model.wte_inputs[0].tolist())),
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/example-lm_head-bias.png",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                        plot_and_save_primitives_matrices_for_main_paper(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/primitive-lm_head-bias.pdf",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_example)
                        plot_and_save_primitives_matrices_for_main_paper(prod.squeeze(),
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/primitive-lm_head-bias.png",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_example)
                prod = prod.unsqueeze(0)
            else:
                dep_prod, indep_prod = get_product_for_one_side_for_unembed(hooked_model, oa_vecs, converted_mlp,
                                                                            activation_name_to_keep,
                                                                            hooked_model.input_ids.to(hooked_model.device), hooked_model.position_ids.to(hooked_model.device))
                if primitive.primitive is not None:
                    primitive_matrix = primitive.primitive.contruct_matrix(indep_prod.shape[0], indep_prod.shape[1], tokenizer).to(indep_prod.dtype).to(indep_prod.device)
                    special_tokens = [
                        tokenizer.sep_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
                    ]
                    if primitive.special_primitive is not None:
                        special_primitive_matrix = primitive.special_primitive.contruct_matrix(indep_prod.shape[0], indep_prod.shape[1], tokenizer).to(primitive_matrix.dtype).to(primitive_matrix.device)
                        primitive_matrix[special_tokens, :] = special_primitive_matrix[special_tokens, :]
                    if type(primitive.scaling_factor_primitive) == str and primitive.scaling_factor_primitive == "inf":
                        mask_value = torch.finfo(primitive_matrix.dtype).max
                        primitive_matrix = torch.where(primitive_matrix > 0, mask_value, 0.)
                    else:
                        primitive_matrix = primitive.scaling_factor_primitive * primitive_matrix
                elif primitive.replacement_matrix is not None:
                    primitive_matrix = primitive.replacement_matrix.get_matrix()

                prod = dep_prod @ primitive_matrix
                
                if len(prod.shape) == 2:
                    prod = prod.unsqueeze(0)

                if hooked_model.save_matrices and ("logits", "primitives", activation_name_to_keep) not in hooked_model.saved_matrices:
                    hooked_model.saved_matrices.add(("logits", "primitives", activation_name_to_keep))
                    plot_and_save_primitives_matrices(primitive_matrix,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-matrices/{activation_name_to_keep}.pdf",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod[0].T,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-example/{activation_name_to_keep}.pdf",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    plot_and_save_primitives_matrices(primitive_matrix,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-matrices/{activation_name_to_keep}.png",
                            ticks_x=[t for t in tokenizer.vocab],
                            ticks_y=ticks_y)
                    plot_and_save_primitives_matrices(prod[0].T,
                            f"{hooked_model.save_matrices_path}/{hooked_model.model_name}/lm_head/primitives-example/{activation_name_to_keep}.png",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    if hasattr(hooked_model, "plot_for_main_paper") and hooked_model.plot_for_main_paper:
                        plot_and_save_primitives_matrices_for_main_paper(primitive_matrix,
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/primitive-lm_head-{activation_name_to_keep}.pdf",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_y)
                        plot_and_save_primitives_matrices_for_main_paper(prod[0].T,
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/example-lm_head-{activation_name_to_keep}.pdf",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                        plot_and_save_primitives_matrices_for_main_paper(primitive_matrix,
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/primitive-lm_head-{activation_name_to_keep}.png",
                            ticks_x=[t for t in tokenizer.vocab], ticks_y=ticks_y)
                        plot_and_save_primitives_matrices_for_main_paper(prod[0].T,
                            f"{hooked_model.save_matrices_path}/plots_for_main_paper/{hooked_model.model_name}/example-lm_head-{activation_name_to_keep}.png",
                            ticks_x=ticks_example, ticks_y=[t for t in tokenizer.vocab])
                    
        if output is None:
            output = prod
        else:
            output += prod
    return output