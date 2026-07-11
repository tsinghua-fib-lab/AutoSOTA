from my_paths import SUBMIT_HOST, CONDA_PATH, CONDA_ENV, REMOTE_SCRIPT_PATH, output_dir, HISTORY_FILE, RUNNING_EXPERIMENTS_FILE, path_to_saved_model, CACHE_DIR
import json
import os
import yaml
import subprocess
from pathlib import Path
import math
import streamlit as st
from datetime import datetime, timedelta
import re
import random
import string
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import sys
import glob
from transformers import GPT2LMHeadModel
import pandas as pd
sys.path.append(os.path.abspath(".."))
from convert_to_code import convert_config_to_code_full_paths, convert_config_to_code_one_step_paths, convert_config_to_code_qk_pruned, convert_keys_to_int
from patching_data import get_tokenizer_and_dataset_for_task
from draw_curve_utils import get_pareto_frontier, get_num_element
from show_heatmap import *
from patching_helper_functions import get_full_possible_config_for_pruning
from pruning_model import PruningModelWithHooksForQK, convert_config_fullpaths_, convert_config_fullpaths_incl_mlp_, convert_full_paths_config_to_prune_inside_kq_config_
from collections import defaultdict
import matplotlib.image as mpimg
torch.classes.__path__ = []


st.set_page_config(layout="wide")

def convert_row_to_fig(row: pd.Series, color=None):
    labels = row.index.tolist()
    fig, ax = plt.subplots(figsize=(len(labels), len(labels)/20))  # small, compact
    ax.bar(range(len(labels)), row.values, color=color)

    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)

    # remove top/right spines, make minimal look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(left=False, bottom=False)
    return fig

def save_heatmap_to_cache(matrix, filename):
    # Define a folder to cache heatmaps
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CACHE_DIR / filename
    np.save(filepath, matrix)
    return filepath

def load_heatmap_from_cache(filename):
    # Define a folder to cache heatmaps
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CACHE_DIR / filename
    if filepath.exists():
        return np.load(filepath)
    return None

# Function to plot and save heatmap
def plot_and_cache_heatmap(matrix, labels_right, labels_left, filename):
    cached_matrix = load_heatmap_from_cache(filename)
    if cached_matrix is not None:
        return cached_matrix

    # Plot heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(20, 16))  # Adjust size as needed
    cax = ax.imshow(matrix, cmap='viridis', aspect='auto')
    # Add tick labels to the heatmap axes
    ax.set_xticks(np.arange(len(labels_right)))
    ax.set_xticklabels(labels_right, rotation=90)
    ax.set_yticks(np.arange(len(labels_left)))
    ax.set_yticklabels(labels_left)

    fig.tight_layout()
    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.title("Attention Heatmap")
    plt.savefig(CACHE_DIR / f"{filename}.png")
    plt.close(fig)

    # Save matrix to cache
    save_heatmap_to_cache(matrix, filename)
    return matrix

def temp_func_return_0():   # to make it saveable for pickle
    return 0

@st.cache_resource
def load_model_tokenizer_dataset(exp_config, result_config):
    if exp_config["find_graph_method"] == "run_saved_model":
        path_to_model_outputs = exp_config["path_to_saved_pruned_model"]
        with open(Path(path_to_model_outputs) / "config.yaml") as f:
            exp_config = yaml.safe_load(f)
    else:
        # path_to_model_outputs = exp_config["output_dir"]
        path_to_model_outputs = Path(output_dir) / exp_config["exp_name"] # important do not change
    model = GPT2LMHeadModel.from_pretrained(Path(path_to_saved_model) / exp_config["model_name"])
    model.eval()
    task_name = exp_config["model_name"].split("-")[0]
    tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task_name, exp_config["length_range"], exp_config["max_test_length"], exp_config)
    oa_vecs = torch.load(Path(path_to_model_outputs) / "oa_vecs.pt", map_location="cpu", weights_only=False)
    result_config = convert_keys_to_int(result_config)
    hooked_model = PruningModelWithHooksForQK(model, result_config, defaultdict(temp_func_return_0), exp_config.get("split_mlps", False), print)
    if (Path(path_to_model_outputs) / "converted_mlp.pt").exists():
        converted_mlp = torch.load(Path(path_to_model_outputs) / "converted_mlp.pt", map_location="cpu", weights_only=False)
    else:
        converted_mlp = None
    if (Path(path_to_model_outputs) / "mlp_input_output.pt").exists():
        mlp_input_output = torch.load(Path(path_to_model_outputs) / "mlp_input_output.pt", map_location="cpu", weights_only=False)
    else:
        mlp_input_output = None
    return oa_vecs, hooked_model, tokenizer, iterable_dataset, converted_mlp, mlp_input_output

@st.cache_resource
def load_model_tokenizer_dataset2(exp_config):
    if exp_config["find_graph_method"] == "run_saved_model":
        path_to_model_outputs = exp_config["path_to_saved_pruned_model"]
        with open(Path(path_to_model_outputs) / "config.yaml") as f:
            exp_config = yaml.safe_load(f)
    else:
        # path_to_model_outputs = exp_config["output_dir"]
        path_to_model_outputs = Path(output_dir) / exp_config["exp_name"]
    model = GPT2LMHeadModel.from_pretrained(Path(path_to_saved_model) / exp_config["model_name"], attn_implementation="eager")
    model.eval()
    task_name = exp_config["model_name"].split("-")[0]
    tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task_name, exp_config["length_range"], exp_config["max_test_length"], exp_config)
    return model, tokenizer, iterable_dataset

@torch.no_grad()
def run_model_on_one_input(hooked_model, oa_vecs, iterable_dataset, exp_name):
    input_ids, pos_ids, labels = next(iter(iterable_dataset))

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    pos_ids = torch.tensor([pos_ids], dtype=torch.long)
    if hasattr(iterable_dataset, "BCE"):
        use_BCE = iterable_dataset.BCE
    else:
        use_BCE = False
    labels = torch.tensor([labels], dtype=torch.long)
    if not use_BCE:
        labels[labels == tokenizer.pad_token_id] = -100
    logits = hooked_model(masks=torch.ones((1, 1)), oa_vecs=oa_vecs, input_ids=input_ids, position_ids=pos_ids).logits

    if not use_BCE:
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        predictions = shift_logits.argmax(dim=-1)
        correct = ((predictions == shift_labels) | (shift_labels == -100)).all().item()
    else:
        mask = (input_ids == tokenizer.pad_token_id) | (input_ids == tokenizer.eos_token_id)
        predictions = (logits > 0).long()
        correct = ((predictions == labels).all(dim=-1) | mask).all().item()

    st.session_state[f"last_run-{exp_name}"] = (input_ids, pos_ids, hooked_model.activations, correct)

@torch.no_grad()
def run_model_on_one_input2(model, iterable_dataset, exp_name):
    input_ids, pos_ids, labels = next(iter(iterable_dataset))

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    pos_ids = torch.tensor([pos_ids], dtype=torch.long)
    if hasattr(iterable_dataset, "BCE"):
        use_BCE = iterable_dataset.BCE
    else:
        use_BCE = False
    labels = torch.tensor([labels], dtype=torch.long)
    if not use_BCE:
        labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(input_ids=input_ids, position_ids=pos_ids, output_attentions=True)
    logits = outputs.logits

    if not use_BCE:
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        predictions = shift_logits.argmax(dim=-1)
        correct = ((predictions == shift_labels) | (shift_labels == -100)).all().item()
    else:
        mask = (input_ids == tokenizer.pad_token_id) | (input_ids == tokenizer.eos_token_id)
        predictions = (logits > 0).long()
        correct = ((predictions == labels).all(dim=-1) | mask).all().item()

    st.session_state[f"last_run-{exp_name}"] = (input_ids, pos_ids, outputs.attentions, correct)


st.header("Program")


paths = sorted(glob.glob("newprune-*", root_dir=output_dir) + glob.glob("range50-*", root_dir=output_dir))
path = st.selectbox(label="select exp series", options=paths, index=None)
if path is None:
    st.stop()
model_name = "-".join(path.split("-")[1:-1])
original_results = Path(path_to_saved_model) / model_name / "acc.txt" #f"{model_name.split('-')[0]}-average.txt"
if original_results.exists():
    with open(original_results) as f:
        lines = f.read()
    # original_results = list(filter(lambda x: x, [line.rstrip() for line in lines]))[-1]
    st.write(lines)
series_path = Path(output_dir) / path
selected_stage = st.radio("select stage", options=[0, 1, 2, 3], index=3, format_func=lambda i: f"stage {i}", horizontal=True, label_visibility="collapsed")
data_points = []
for exp_name in glob.glob("exp*", root_dir=series_path):
    if (series_path / exp_name / "output.json").exists():
        with open(series_path / exp_name / "output.json") as f:
            output_dict = json.load(f)
        stages = []
        for key in output_dict:
            if match := re.search(r"result_patching_config_global_iteration_(\d+)", key):
                stages.append(int(match.group(1)))
        if max(stages) == 0:
            with open(series_path / exp_name / "config.yaml") as f:
                linear_LN = yaml.safe_load(f).get("linear_LN", True)
            if linear_LN:
                exp_stage = 1
            else:
                exp_stage = 0
        else:
            exp_stage = max(stages)+1
        if exp_stage != selected_stage:
            continue
    else:
        continue

    final_config = output_dict[f"result_patching_config_global_iteration_{max(stages)}"]
    num_edges = get_num_element(final_config)
    acc_match = output_dict["acc_match"]
    data_points.append((acc_match, num_edges, exp_name))

if len(data_points) == 0:
    st.write("No valid point")
    st.stop()


frontier = get_pareto_frontier(data_points)

num_heads_per_layer = {layer_idx: len(final_config[str(layer_idx)]["k"]) for layer_idx in range(len(final_config)-1)}
with open(series_path / exp_name / "config.yaml") as f:
    split_mlps = yaml.safe_load(f)["split_mlps"]
full_config = get_full_possible_config_for_pruning(num_heads_per_layer)
if selected_stage >= 2:
    if split_mlps:
        convert_config_fullpaths_incl_mlp_(full_config, num_heads_per_layer)
    else:
        convert_config_fullpaths_(full_config, num_heads_per_layer)
if selected_stage >= 3:
    convert_full_paths_config_to_prune_inside_kq_config_(full_config, num_heads_per_layer)
orig_num_edges = get_num_element(full_config)

accs, num_edges, exp_names = zip(*data_points)
frontier_accs, frontier_num_edges, _ = zip(*frontier)

fig = go.Figure()

# Yellow staircase line
fig.add_trace(go.Scatter(
    x=frontier_num_edges, y=frontier_accs,
    mode="lines",
    line=dict(color="orange", width=3, shape="hv"),  # staircase: horizontal→vertical
    name="Pareto Frontier",
    hoverinfo="skip"
))

# Red "x" markers
fig.add_trace(go.Scatter(
    x=num_edges, y=accs,
    mode="markers",
    marker=dict(symbol="x", size=10, color="red"),
    name="Pruned Models"
))

fig.add_trace(go.Scatter(
    x=[orig_num_edges], y=[1.0],
    mode="markers",
    name="Original Model",
    marker=dict(
        symbol="star",
        size=10,
        color="gold",
        line=dict(color="black", width=1.5)
    )
))

if orig_num_edges > max(num_edges)*10:
    logscale = st.toggle("log scale x-axis", value=True)
else:
    logscale = False
if logscale:
    xaxis = dict(range=[-1, math.log10(orig_num_edges)*1.05], autorange=False, type="log")
else:
    xaxis = dict(range=[0, math.ceil(orig_num_edges*1.05)], autorange=False)

fig.update_layout(
    # title=model_name,
    xaxis_title="Number of Edges",
    yaxis_title="Match Accuracy",
    template="simple_white",
    hovermode="closest",
    yaxis=dict(range=[0, 1.05], autorange=False),
    xaxis=xaxis,
)

event = st.plotly_chart(
    fig,
    key="pareto",
    use_container_width=False,
    on_select="rerun",
    selection_mode=("points",)
)

selected_exp_name = None
if event and event.get("selection", {}).get("points"):
    pt = event["selection"]["points"][0]
    if pt["curve_number"] == 1:
        selected_exp_name = exp_names[pt["point_index"]]

if selected_exp_name is None:
    st.write("select a point to see program")
    st.stop()

selected_exp_data = {}
with open(series_path / selected_exp_name / "config.yaml") as f:
    selected_exp_data["config"] = yaml.safe_load(f)
    selected_exp_data["config"]["exp_name"] = Path(path) / selected_exp_data["config"]["exp_name"]

with st.expander("Check configuration"):
    st.code(yaml.dump(selected_exp_data['config']), language="yaml")

# Load output.json for the selected experiment
output_path = Path(output_dir) / selected_exp_data['config']['exp_name'] / "output.json"
if not output_path.exists():
    st.error("Output file not found for the selected experiment.")
    st.stop()

with open(output_path, 'r') as f:
    output_data = json.load(f)

hooked_model = None
if "result_patching_config_global_iteration_2" in output_data and selected_exp_data['config']["find_graph_method"] == "pruning" or selected_exp_data['config']["find_graph_method"] == "run_saved_model":
    config = output_data["result_patching_config_global_iteration_2"]
    oa_vecs, hooked_model, tokenizer, iterable_dataset, converted_mlp, mlp_input_output = load_model_tokenizer_dataset(selected_exp_data['config'], output_data["result_patching_config_global_iteration_2"])

    if selected_exp_data['config']["find_graph_method"] == "run_saved_model":
        with open(Path(selected_exp_data["config"]["path_to_saved_pruned_model"]) / "config.yaml") as f:
            split_mlps = yaml.safe_load(f).get("split_mlps", False)
    elif selected_exp_data['config']["find_graph_method"] == "pruning":
        split_mlps = selected_exp_data["config"].get("split_mlps", False)
    
    primitives_path_data = Path(output_dir) / selected_exp_data['config']['exp_name'] / "attention_primitives" / "configs" / f"{selected_exp_data['config']['model_name'].split('-')[0]}.json"
    replace_attn_with_primitives, attn_primitives = False, None
    if primitives_path_data.exists():
        with open(primitives_path_data, 'r') as f:
            primitives_data = json.load(f)
        st.text(f"Replace attention with primitives, acc: {primitives_data['accuracy']}; match acc: {primitives_data['acc_match']}")
        replace_attn_with_primitives = True
        attn_primitives = primitives_data["primitives"]
    code, heatmap_to_config = convert_config_to_code_qk_pruned(config, split_mlps, converted_mlp,
                                                                replace_attn_with_primitives, attn_primitives)

    st.text(output_data["result_patching_config_global_iteration_2"])
    st.subheader("Generated Code")
    st.code("\n".join(code))
    selected_heatmap = st.pills("select a heatmap", options=list(heatmap_to_config.keys()), selection_mode="single", label_visibility="hidden") 

    
    st.button("random example", on_click=run_model_on_one_input, args=(hooked_model, oa_vecs, iterable_dataset, selected_exp_data['config']['exp_name']))
    if f"last_run-{selected_exp_data['config']['exp_name']}" not in st.session_state:
        run_model_on_one_input(hooked_model, oa_vecs, iterable_dataset, selected_exp_data['config']['exp_name'])
    
    input_ids, pos_ids, hooked_model.activations, correct = st.session_state[f"last_run-{selected_exp_data['config']['exp_name']}"]
    
    if selected_heatmap is not None:
        # attn_layer_idx, _, attn_head_idx, (q_act, k_act) = 2, "qk", 1, ('mlp-0', 'mlp-0')
        # attn_layer_idx, _, attn_head_idx, (q_act, k_act) = 1, "qk", 0, ('wte', 'wte')
        if type(heatmap_to_config[selected_heatmap]) == tuple:
            attn_layer_idx, attn_head_idx, prod = heatmap_to_config[selected_heatmap]
            if type(prod) == tuple or type(prod) == list:
                q_act, k_act = prod

                products = {}
                for qk_type, path in zip(["q", "k"], [q_act, k_act]):
                    products[qk_type] = get_product_for_one_side_for_head(hooked_model, oa_vecs, converted_mlp, attn_layer_idx, attn_head_idx, qk_type, path, input_ids.squeeze(0), pos_ids.squeeze(0))
                
                final_left = products["q"][0]
                final_middle = products["q"][1] @ products["k"][1].T
                final_right = products["k"][0].T

                fig, axes = plt.subplots(1, 3 + int(replace_attn_with_primitives), figsize=(18, 6))

                im0 = axes[0].imshow(final_left.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[0].set_title('query input dependent')
                plt.colorbar(im0, ax=axes[0], shrink=0.5)

                im1 = axes[1].imshow(final_middle.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[1].set_title('query-key input independent')
                plt.colorbar(im1, ax=axes[1], shrink=0.5)

                im2 = axes[2].imshow(final_right.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[2].set_title('key input dependent')
                plt.colorbar(im2, ax=axes[2], shrink=0.5)

                if replace_attn_with_primitives:
                    primitive_heatmap_path = Path(output_dir) / selected_exp_data['config']['exp_name'] / "attention_primitives" / "heatmaps" / f"{selected_exp_data['config']['model_name'].split('-')[0]}" / f"{attn_layer_idx}-{attn_head_idx}" / "primitives-matrices" / f"{q_act}-{k_act}.png"
                    if primitive_heatmap_path.exists():
                        img = mpimg.imread(primitive_heatmap_path)
                        axes[-1].imshow(img)
                        axes[-1].axis("off")  # Hide axes for image
                        axes[-1].set_title("Primitive")

                fig.suptitle(f"q:{q_act}            k:{k_act}", fontsize=20)
                plt.tight_layout()
                st.pyplot(fig)


            elif type(prod) == str:
                k_act = prod

                dependent, independent = get_product_for_one_side_for_head(hooked_model, oa_vecs, converted_mlp, attn_layer_idx, attn_head_idx, "k", k_act, input_ids.squeeze(0), pos_ids.squeeze(0))
                alpha = oa_vecs.q_bias_term.data[oa_vecs.to_q_bias[(attn_layer_idx, attn_head_idx, k_act)]].unsqueeze(0) # d_head

                final_middle = alpha @ independent.T
                final_right = dependent.T

                fig, axes = plt.subplots(1, 2 + int(replace_attn_with_primitives), figsize=(18, 6))

                im1 = axes[0].imshow(final_middle.cpu().detach().numpy(), aspect=3, cmap='viridis', interpolation='none')
                axes[0].set_title('query-key input independent')
                plt.colorbar(im1, ax=axes[0])

                im2 = axes[1].imshow(final_right.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[1].set_title('key input dependent')
                plt.colorbar(im2, ax=axes[1])

                if replace_attn_with_primitives:
                    primitive_heatmap_path = Path(output_dir) / selected_exp_data['config']['exp_name'] / "attention_primitives" / "heatmaps" / f"{selected_exp_data['config']['model_name'].split('-')[0]}" / f"{attn_layer_idx}-{attn_head_idx}" / "primitives-matrices" / f"bias-{k_act}.png"
                    if primitive_heatmap_path.exists():
                        img = mpimg.imread(primitive_heatmap_path)
                        axes[-1].imshow(img)
                        axes[-1].axis("off")  # Hide axes for image
                        axes[-1].set_title("Primitive")

                fig.suptitle(f"q:None            k:{k_act}", fontsize=20)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                raise RuntimeError(prod)
        
        elif type(heatmap_to_config[selected_heatmap]) == str:
            lm_head_inp = heatmap_to_config[selected_heatmap]
            dependent, independent = get_product_for_one_side_for_unembed(hooked_model, oa_vecs, converted_mlp, lm_head_inp, input_ids.squeeze(0), pos_ids.squeeze(0))

            if lm_head_inp != "vocab_bias":
                fig, axes = plt.subplots(1, 2 + int(replace_attn_with_primitives),
                                            figsize=(15, 6))

                im0 = axes[0].imshow(dependent.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[0].set_title('input dependent')
                plt.colorbar(im0, ax=axes[0], shrink=0.5)

                im1 = axes[1].imshow(independent.cpu().detach().numpy(), cmap='viridis', interpolation='none')
                axes[1].set_title('input independent')
                plt.colorbar(im1, ax=axes[1], shrink=0.5)

                if replace_attn_with_primitives:
                    primitive_heatmap_path = Path(output_dir) / selected_exp_data['config']['exp_name'] / "attention_primitives" / "heatmaps" / f"{selected_exp_data['config']['model_name'].split('-')[0]}" / "lm_head" / "primitives-matrices" / f"{lm_head_inp}.png"
                    if primitive_heatmap_path.exists():
                        img = mpimg.imread(primitive_heatmap_path)
                        axes[2].imshow(img)
                        axes[2].axis("off")  # Hide axes for image
                        axes[2].set_title("Primitive")

                fig.suptitle(f"{lm_head_inp}", fontsize=20)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                fig, ax = plt.subplots(1, 1 + int(replace_attn_with_primitives), figsize=(8, 2))

                if replace_attn_with_primitives:
                    im0 = ax[0].imshow(independent.cpu().detach().numpy(), aspect=3, cmap='viridis', interpolation='none')
                    ax[0].set_title('input independent')
                    plt.colorbar(im0, ax=ax)

                    primitive_heatmap_path = Path(output_dir) / selected_exp_data['config']['exp_name'] / "attention_primitives" / "heatmaps" / f"{selected_exp_data['config']['model_name'].split('-')[0]}" / "lm_head" / "primitives-matrices" / f"bias.png"
                    if primitive_heatmap_path.exists():
                        img = mpimg.imread(primitive_heatmap_path)
                        ax[1].imshow(img)
                        ax[1].axis("off")  # Hide axes for image
                        ax[1].set_title("Primitive")
                else:
                    im0 = ax.imshow(independent.cpu().detach().numpy(), aspect=3, cmap='viridis', interpolation='none')
                    ax.set_title('input independent')
                    plt.colorbar(im0, ax=ax)

                fig.suptitle(f"{lm_head_inp}", fontsize=20)
                plt.tight_layout()
                st.pyplot(fig)
                
        else:
            raise RuntimeError(selected_heatmap)
    
    # show mlp inp-out if exists
    if mlp_input_output is not None and len(mlp_input_output) > 0:
        st.subheader("Inspect MLP via input and output")
        def format_func(option):
            if type(option) == str:
                return option
            else:
                q, k, l, h = option
                return f"Attn Head {l}.{h}: &nbsp;&nbsp; q={q} &nbsp;&nbsp; k={k}"
        selected_path = st.pills("select", options=list(mlp_input_output.keys()), selection_mode="single", label_visibility="hidden", format_func=format_func)
        show_vis = st.toggle("show visualization")
        if selected_path is not None:
            if type(selected_path) == str:    # end point is unembed
                # k, vocab, in_vocab
                inp_cache, out_cache = mlp_input_output[selected_path]
                tokens = [tokenizer.vocab_inv[i] for i in range(len(tokenizer))]
                sel_token = st.pills("select output token", options=tokens, selection_mode="single")
                if sel_token is not None:
                    sel_token_id = tokenizer.vocab[sel_token]
                    show_out = st.toggle("show out distribution")
                    if isinstance(inp_cache, list):
                        for bin_idx, (bin_inp, bin_out) in enumerate(zip(inp_cache, out_cache)):
                            st.text(f"Bin {bin_idx}")
                            in_labels = tokens if bin_inp.size(2) == len(tokens) else list(range(bin_inp.size(2)))
                            # in_labels.append(f"logit_{sel_token}")
                            # temp = torch.cat([bin_inp[:, sel_token_id], bin_out[:, sel_token_id, sel_token_id].unsqueeze(1)], dim=1)
                            if not show_out:
                                df = pd.DataFrame(bin_inp[:, sel_token_id].numpy(force=True), columns=in_labels)
                                if show_vis:
                                    for i in range(len(df)):
                                        row = df.iloc[i]
                                        fig = convert_row_to_fig(row)

                                        col1, col2, col3 = st.columns([6, 1, 2])
                                        col1.pyplot(fig)
                                        col2.markdown("### →")
                                        col3.markdown(f"**{bin_out[i, sel_token_id, sel_token_id].item():.2f}**")
                                        plt.close(fig)
                                else:
                                    df.insert(loc=len(in_labels), column="", value="-->")
                                    df.insert(loc=len(df.columns), column=f"logit:{sel_token}", value=bin_out[:, sel_token_id, sel_token_id].numpy(force=True))
                                    df = df.round(2)
                                    st.dataframe(df, use_container_width=False, hide_index=True)
                            else:
                                if show_vis:
                                    df1 = pd.DataFrame(bin_inp[:, sel_token_id].numpy(force=True), columns=in_labels)
                                    df2 = pd.DataFrame(bin_out[:, sel_token_id].numpy(force=True), columns=in_labels)

                                    for i in range(len(df1)):
                                        row1 = df1.iloc[i]
                                        fig1 = convert_row_to_fig(row1)
                                        row2 = df2.iloc[i]
                                        fig2 = convert_row_to_fig(row2, np.where(row2.values < 0, 'red', 'green'))

                                        col1, col2, col3 = st.columns([6, 1, 6])
                                        col1.pyplot(fig1)
                                        col2.markdown("### →")
                                        col3.pyplot(fig2)
                                        plt.close(fig1)
                                        plt.close(fig2)
                                else:
                                    labels = [("Input Variable", l) for l in in_labels] + [("Output Logits", t) for t in tokens]
                                    columns = pd.MultiIndex.from_tuples(labels)
                                    df = pd.DataFrame(torch.cat([bin_inp[:, sel_token_id], bin_out[:, sel_token_id]], dim=1).numpy(force=True), columns=columns).round(2)
                                    st.dataframe(df, use_container_width=False, hide_index=True)
                    else:
                        
                        # TODO: merge code for the three cases, remove redundancy
                        for bin_idx in range(len(out_cache)):
                            bin_inp = [(k, v[bin_idx]) for k, v in inp_cache.items()]
                            st.text(f"Bin {bin_idx}")
                            all_in_labels = [(k, tokens if bin_inp_item.size(2) == len(tokens) else list(range(bin_inp_item.size(2)))) for k, bin_inp_item in bin_inp]
                            if show_vis:
                                st.warning("visualized bar plot takes too much space, not implemented")

                            labels = []
                            for k, in_labels in all_in_labels:
                                labels.extend([(k, l) for l in in_labels])
                            bin_out = out_cache[bin_idx]
                            
                            if not show_out:
                                labels.append(("Output Logits", sel_token))
                                columns = pd.MultiIndex.from_tuples(labels)
                                temp = torch.cat([v[:, sel_token_id] for k, v in bin_inp], dim=1)
                                temp = torch.cat([temp, bin_out[:, sel_token_id, sel_token_id].unsqueeze(1)], dim=1)
                                df = pd.DataFrame(temp.numpy(force=True), columns=columns).round(2)
                            else:
                                labels.extend([("Output Logits", t) for t in tokens])
                                columns = pd.MultiIndex.from_tuples(labels)
                                df = pd.DataFrame(torch.cat([v[:, sel_token_id] for k, v in bin_inp] + [bin_out[:, sel_token_id]], dim=1).numpy(force=True), columns=columns).round(2)
                                
                            st.dataframe(df, use_container_width=False, hide_index=True)

            else:
                q_inp_cache, k_inp_cache, out_cache = mlp_input_output[selected_path]
                # k, num_q_cluster, in_vocab
                num_q_cluster = q_inp_cache[0].size(1)
                tokens = [tokenizer.vocab_inv[i] for i in range(len(tokenizer))]
                clusters = ["All"] + [f"cluster_{i}" for i in range(num_q_cluster-1)]
                sel_cluster = st.pills("select output token", options=clusters, selection_mode="single")
                if sel_cluster is not None:
                    sel_cluster_id = clusters.index(sel_cluster)
                    for bin_idx, (bin_q_inp, bin_k_inp, bin_out) in enumerate(zip(q_inp_cache, k_inp_cache, out_cache)):
                        st.text(f"Bin {bin_idx}")
                        q_in_labels = tokens if bin_q_inp.size(2) == len(tokens) else list(range(bin_q_inp.size(2)))
                        k_in_labels = tokens if bin_k_inp.size(2) == len(tokens) else list(range(bin_k_inp.size(2)))

                        if show_vis:
                            df1 = pd.DataFrame(bin_q_inp[:, sel_cluster_id].numpy(force=True), columns=q_in_labels)
                            df2 = pd.DataFrame(bin_k_inp[:, sel_cluster_id].numpy(force=True), columns=k_in_labels)
                            
                            for i in range(len(df1)):
                                row1 = df1.iloc[i]
                                fig1 = convert_row_to_fig(row1)
                                
                                row2 = df2.iloc[i]
                                fig2 = convert_row_to_fig(row2)

                                col1, col2, col3, col4, col5 = st.columns([6, 1, 6, 2, 1])
                                col1.pyplot(fig1)
                                col2.markdown("### &")
                                col3.pyplot(fig2)
                                col4.markdown("### →")
                                col5.markdown(f"**{bin_out[i, sel_cluster_id].item():.2f}**")
                                plt.close(fig1)
                                plt.close(fig2)
                        else:
                            labels = [("Q", l) for l in q_in_labels] + [("K", l) for l in k_in_labels] + [("Product", "Logits")]
                            columns = pd.MultiIndex.from_tuples(labels)

                            df = pd.DataFrame(
                                torch.cat([bin_q_inp[:, sel_cluster_id], bin_k_inp[:, sel_cluster_id], bin_out[:, sel_cluster_id].unsqueeze(1)], dim=1).numpy(force=True),
                                columns=columns
                            ).round(2)
                            st.dataframe(df, use_container_width=False, hide_index=True)



    # show attn
    st.subheader("Attention Weights")
    effective_heads = []
    for layer_idx in range(len(config)-1):
        layer_idx = str(layer_idx)
        for head_idx in config[layer_idx]["v"]:
            if config[layer_idx]["v"][head_idx]:
                effective_heads.append((layer_idx, head_idx))
    selected_head = st.pills("select a head", options=effective_heads, selection_mode="single", label_visibility="hidden") 

    if selected_head is not None:
        layer_idx, head_idx = selected_head
        A = get_attn_weights_for_head(hooked_model, int(layer_idx), int(head_idx)).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        fig, ax = plt.subplots(figsize=(10,10))

        im1 = ax.imshow(A.cpu().numpy(), cmap='viridis', interpolation='none')
        plt.xticks(np.arange(A.size(1)), tokens, rotation=90, fontsize=10 * (50 / A.size(1)))
        plt.yticks(np.arange(A.size(0)), tokens, fontsize=10 * (50 / A.size(1)))
        plt.colorbar(im1, ax=ax, shrink=0.2)
        plt.tight_layout()
        st.pyplot(fig)

else:
    # show attn in original model
    num_layers, num_heads = re.search("(\d+)l(\d+)h\d+d", selected_exp_data["config"]["model_name"]).groups()
    num_layers, num_heads = int(num_layers), int(num_heads)
    effective_heads = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            effective_heads.append((layer_idx, head_idx))
    selected_head = st.pills("select a head", options=effective_heads, selection_mode="single", label_visibility="hidden") 

    model, tokenizer, iterable_dataset = load_model_tokenizer_dataset2(selected_exp_data['config'])
    st.button("random example", on_click=run_model_on_one_input2, args=(model, iterable_dataset, selected_exp_data['config']['exp_name']))
    if f"last_run-{selected_exp_data['config']['exp_name']}" not in st.session_state:
        run_model_on_one_input2(model, iterable_dataset, selected_exp_data['config']['exp_name'])
    input_ids, pos_ids, attentions, correct = st.session_state[f"last_run-{selected_exp_data['config']['exp_name']}"]

    if selected_head is not None:
        layer_idx, head_idx = selected_head
        A = attentions[layer_idx][0, head_idx]

        st.write("correct:", correct)
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        fig, ax = plt.subplots(figsize=(10,10))

        im1 = ax.imshow(A.cpu().numpy(), cmap='viridis', interpolation='none')
        plt.xticks(np.arange(A.size(1)), tokens, rotation=90, fontsize=10 * (50 / A.size(1)))
        plt.yticks(np.arange(A.size(0)), tokens, fontsize=10 * (50 / A.size(1)))
        plt.colorbar(im1, ax=ax, shrink=0.2)
        plt.tight_layout()
        st.pyplot(fig)