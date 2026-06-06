from dotenv import load_dotenv
load_dotenv() # Load environment variables (HF_CACHE) from .env file

import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from src.activations import get_seq_acts_and_out
from src.data import load_data
import os
import plotly.express as px
from sklearn.manifold import TSNE
from dictionary_learning.utils import load_dictionary
from nnsight import LanguageModel
import argparse


def load_activations_tsne(model, submodule, sae, dataset, chunk=None, **data_kwargs):
    """
    Load activations and labels for sequences, so multiple pieces of data per text sample
    """
    data, domains = load_data(dataset, **data_kwargs)

    if dataset == "wikipedia":
        N = 940
    elif dataset == "finefineweb":
        N = 20000
    elif dataset == "mmlu":
        N = 6000

    features = []
    sequence_labels = []
    part_of_speech_labels = []
    domain_labels = []
    labels = []

    for i in tqdm(range(0, N//30)):
        if dataset == "mmlu":
            prompt = data[i]["question"]
        else:
            prompt = data[i]["text"]
        model_latent, model_in, pos = get_seq_acts_and_out(model, submodule, prompt, num_tokens=30) 
        num_activations = model_latent.shape[0]
        if sae is not None:
            sae_feats = sae.encode(model_latent)
            if chunk == 0:
                features.append(sae_feats[:, :sae.group_sizes[0]])
            elif chunk == 1:
                features.append(sae_feats[:, sae.group_sizes[0]:])
            else:
                features.append(sae_feats)
        else:
            features.append(model_latent)

        for tok in model_in:
            labels.append(model.tokenizer.decode(tok))

        sequence_labels.extend([i]*num_activations)
        part_of_speech_labels.extend(pos)
        domain_labels.extend([domains[i]]*num_activations)
        print(num_activations, domains[i])

    features = torch.cat(features).cpu().numpy()
    sequence_labels = np.array(sequence_labels)
    part_of_speech_labels = np.array(part_of_speech_labels)
    domain_labels = np.array(domain_labels)
    return features, labels, sequence_labels, part_of_speech_labels, domain_labels

def interactive_tsne(features, labels, seq_colors, pos_colors, domain_colors, args):

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    df_tsne = pd.DataFrame(tsne_results, columns=['tsne-one', 'tsne-two'])
    df_tsne['label'] = labels
    df_tsne['pos_colors'] = pos_colors
    df_tsne['domain_colors'] = domain_colors
    df_tsne['seq_colors'] = seq_colors


    # Create the interactive scatter plot using Plotly
    fig = px.scatter(
        df_tsne, x='tsne-one', y='tsne-two', color='pos_colors',
        hover_name='label', # hover_data={'tsne-one': False, 'tsne-two': False, 'species': True},
        # title="t-SNE Visualization of High-Dimensional Data",
        color_continuous_scale=px.colors.sequential.Rainbow
    )
    fig.update_traces(marker=dict(size=8))
    

    fig.update_layout(
        # xaxis_title="t-SNE 1",
        # yaxis_title="t-SNE 2", # legend_title="Species",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        showlegend=False,
        xaxis_title=None, yaxis_title=None
    )
    path = "tsne_plot_pythia_" + args.run_name + "_mmlu_pos_chunk" + str(args.chunk) + "_ffw.html"
    fig.write_html(path) 
    # fig.write_image("tsne_plot_" + args.run_name + "_mmlu_pos_chunk" + str(args.chunk) + ".pdf")


    fig = px.scatter(
        df_tsne, x='tsne-one', y='tsne-two', color='domain_colors',
        hover_name='label', # hover_data={'tsne-one': False, 'tsne-two': False, 'species': True},
        # title="t-SNE Visualization of High-Dimensional Data",
        color_continuous_scale=px.colors.sequential.Rainbow
    )
    fig.update_traces(marker=dict(size=8))
    
    fig.update_layout(
        # xaxis_title="t-SNE 1",
        # yaxis_title="t-SNE 2", # legend_title="Species",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        showlegend=False,
        xaxis_title=None, yaxis_title=None
    )
    path = "tsne_plot_pythia_" + args.run_name + "_mmlu_domain_chunk" + str(args.chunk) + "_ffw.html"
    fig.write_html(path) 
    # fig.write_image("tsne_plot_" + args.run_name + "_mmlu_domain_chunk" + str(args.chunk) + ".pdf")



    # Create the interactive scatter plot using Plotly
    fig = px.scatter(
        df_tsne, x='tsne-one', y='tsne-two', color='seq_colors',
        hover_name='label', # hover_data={'tsne-one': False, 'tsne-two': False, 'species': True},
        # title="t-SNE Visualization of High-Dimensional Data",
        color_continuous_scale=px.colors.cyclical.HSV
    )
    fig.update_traces(marker=dict(size=8))
    
    fig.update_layout(
        # xaxis_title="t-SNE 1",
        # yaxis_title="t-SNE 2", # legend_title="Species",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        ),
        showlegend=False,
        xaxis_title=None, yaxis_title=None
    )
    path = "tsne_plot_pythia_" + args.run_name + "_mmlu_seq_chunk" + str(args.chunk) + "_ffw.html"
    fig.write_html(path) 
    # fig.write_image("tsne_plot_" + args.run_name + "_mmlu_seq_chunk" + str(args.chunk) + ".pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_path", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--chunk", type=int, default=2)
    args = parser.parse_args()
    print(f"Loading SAEs from {args.checkpoints_path}")


    device = "cuda:0"
    model_name =  args.model # can be any Huggingface model

    model = LanguageModel(
        model_name,
        device_map=device,
        dispatch=True,
    )

    if hasattr(model, "model"):
        submodule = model.model.layers[args.layer] # gemma
    elif hasattr(model, "gpt_neox"):
        submodule = model.gpt_neox.layers[args.layer] 
    else:
        print("model doesnt have .model or .gpt_neox attr, check how to access layers")
        exit(1)

    if args.run_name == "baseline_model":
        sae=None
    else:
        sae, trainer_cfg = load_dictionary(os.path.join(args.checkpoints_path, f"{args.run_name}/trainer_0"), device, checkpoint_path=args.checkpoint)
        sae.temporal = False
        sae.eval()
        for param in sae.parameters():
            param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    if args.dataset == "finefineweb":
        domains = ["christianity", "law", "literature", "economics", "food", "drama_and_film", "health", "mathematics", "medical", "history"]
        features, labels, sequence_labels, part_of_speech_labels, domain_labels = load_activations_tsne(model, submodule, sae, args.dataset, chunk=args.chunk, domains=domains)

    elif args.dataset == "mmlu":
        domains = ["high_school_mathematics", "formal_logic", "professional_medicine", "high_school_european_history", "high_school_chemistry", "high_school_statistics", "college_computer_science", "world_religions", "high_school_psychology", "college_physics"]
        features, labels, sequence_labels, part_of_speech_labels, domain_labels = load_activations_tsne(model, submodule, sae, args.dataset, chunk=args.chunk, domains=domains)

    elif args.dataset == "wikipedia":
        features, labels, sequence_labels, part_of_speech_labels, domain_labels = load_activations_tsne(model, submodule, sae, args.dataset, chunk=args.chunk)

    
    # features, labels, seq_colors = generate_labeled_ffw_sequence_data(model, submodule, sae, domains, dataset, args)
    interactive_tsne(features, labels, sequence_labels, part_of_speech_labels, domain_labels, args)