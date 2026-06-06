from dotenv import load_dotenv
load_dotenv()

import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from .activations import get_seq_acts_and_out
import os
import random
 # Load environment variables (HF_CACHE) from .env file

def balance_dataset(data, labelkey, minority_label):
    unique_labels = set(data[labelkey])

    min_count = len(data.filter(lambda x: x[labelkey] == minority_label))
    class_datasets = []
    for label in unique_labels:
        class_data = data.filter(lambda x: x[labelkey] == label)
        
        # If class has more samples than needed, sample randomly
        if len(class_data) > min_count:
            # Use train_test_split to get random sample
            sampled_data, _ = class_data.train_test_split(
                train_size=min_count, 
                seed=42
            ).values()
            class_datasets.append(sampled_data)
        else:
            class_datasets.append(class_data)

    # Concatenate and shuffle
    balanced_dataset = concatenate_datasets(class_datasets).shuffle(seed=42)
    return balanced_dataset

def load_data(dataset, **data_kwargs):
    if dataset == "finefineweb":
        data = load_dataset("m-a-p/FineFineWeb-test", split="train")
        if data_kwargs.get("domains") is not None: ## load domains if specified
            domains = data_kwargs.get("domains")
            data = data.filter(lambda x: x["domain"] in domains)
        data = data.shuffle(seed=42)
        domains = [x["domain"] for x in data]
    elif dataset == "wikipedia":
        data = load_dataset("vietgpt/wikipedia_en", split="train")
        current_path = os.path.dirname(os.path.realpath(__file__))
        domains = pd.read_csv(os.path.join(current_path, "wikipedia_categories.csv"), header=0)["Category"].tolist()
        data = data.shuffle(seed=42)
        domains = [x["domain"] for x in data]
    elif dataset == "mmlu":
        print("LOADING MMLU")
        data = load_dataset("cais/MMLU", "all")["test"]
        if data_kwargs.get("domains") is not None: ## load domains if specified
            domains = data_kwargs.get("domains")
            data = data.filter(lambda x: x["subject"] in domains)
            data = data.filter(lambda x: len(x["question"]) > 200)
        # print(data['subject'].value_counts())
        data = data.shuffle(seed=42)
        domains = [x["subject"] for x in data]
        print(len(domains))
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return data, domains
    
def load_activations_and_labels(model, submodule, sae, dataset, chunk=None, **data_kwargs):
    data, domains = load_data(dataset, **data_kwargs)

    if sae is not None:
        if chunk == 0:
            features = torch.zeros((4000, sae.group_sizes[0]))
        elif chunk == 1:
            features = torch.zeros((4000, sae.group_sizes[1])) # 32768
        else:
            features = torch.zeros((4000, sae.dict_size))
    else: 
        features = torch.zeros((4000, model.config.hidden_size))
    

    part_of_speech_labels = []
    domain_labels = []
    for i in tqdm(range(0, 4000)):
        prompt = data[i]["text"]
        model_latent, model_in, pos = get_seq_acts_and_out(model, submodule, prompt, num_tokens=1) ## model activations for full sequence (50th token onwards)

        if sae is not None:
            sae_feats = sae.encode(model_latent)
            if chunk == 0:
                features[i] = sae_feats[:, :sae.group_sizes[0]]
            elif chunk == 1:
                features[i] = sae_feats[:, sae.group_sizes[0]:]
            else:
                features[i] = sae_feats
        else:
            features[i] = model_latent

        part_of_speech_labels.append(pos[-1])
        domain_labels.append(domains[i])

    # features = torch.cat(features).cpu().numpy()
    part_of_speech_labels = np.array(part_of_speech_labels)
    domain_labels = np.array(domain_labels)
    return features.cpu().numpy(), part_of_speech_labels, domain_labels

def load_activations_and_labels_sequences(model, submodule, sae, dataset, chunk=None, **data_kwargs):
    """
    Load activations and labels for sequences, so multiple pieces of data per text sample
    """
    data, _ = load_data(dataset, **data_kwargs)

    if sae is not None:
        if chunk == 0:
            features = torch.zeros((4000, sae.group_sizes[0]))
        elif chunk == 1:
            features = torch.zeros((4000, sae.group_sizes[1])) # 32768
        else:
            features = torch.zeros((4000, sae.dict_size))
    else: 
        features = torch.zeros((4000, model.config.hidden_size))

    features = []
    sequence_labels = []
    for i in tqdm(range(0, 4000//25)):
        prompt = data[i]["text"]
        model_latent, model_in, pos = get_seq_acts_and_out(model, submodule, prompt, num_tokens=25) ## model activations for full sequence (50th token onwards)
        # model_latent, model_in, pos = get_acts_word_and_pos(model, submodule, prompt) ## model activations for full sequence (50th token onwards)
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
        

        sequence_labels.extend([i]*num_activations)

    features = torch.cat(features, dim=0).reshape(-1, features[0].shape[-1]).cpu().numpy()
    sequence_labels = np.array(sequence_labels)
    return features, sequence_labels

def load_one_sequence(model, submodules, sae, dataset, chunk=None, index=0, **data_kwargs):
    """
    Load activations and labels for 1 full sequence
    """
    data, _ = load_data(dataset, **data_kwargs)
    prompt = data[index]["text"]
    model_latent, model_in, pos = get_seq_acts_and_out(model, submodules, prompt, num_tokens=-1) ## model activations for full sequence (50th token onwards)
    if isinstance(submodules, list):
        features = []
        for layer in range(len(submodules)):
            if sae is not None:
                sae_feats = sae.encode(model_latent[layer])
                if chunk == 0:
                    features.append(sae_feats[:, :sae.group_sizes[0]])
                elif chunk == 1:
                    features.append(sae_feats[:, sae.group_sizes[0]:])
                else:
                    features.append(sae_feats)
            else:
                features.append(model_latent[layer])
        return features
    else:
        if sae is not None:
            sae_feats = sae.encode(model_latent)
            if chunk == 0:
                return sae_feats[:, :sae.group_sizes[0]]
            elif chunk == 1:
                return sae_feats[:, sae.group_sizes[0]:]
            else:
                return sae_feats
        else:
            return model_latent