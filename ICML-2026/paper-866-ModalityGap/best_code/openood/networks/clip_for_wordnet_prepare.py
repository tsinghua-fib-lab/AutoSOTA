# Calculate the cosine similarity matrix between all WordNet candidate words
# (adjectives and nouns) and the ID class names (e.g. ImageNet-1K classes).
# The resulting similarity matrix is saved as a .pth file and used by
# clip_fixed_ood_prompt.py to select the most semantically dissimilar words
# as OOD negative label prototypes.
import os
import glob
import random

import torch
import time, os
import torch.nn as nn
from openood.networks.clip import clip
# import pdb
import math
# from src.clip import clip as clip
# from openood.networks.clip import clip
# import openood.networks.clip as clip
from PIL import Image
# import ipdb


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        x = self.token_embedding(tokenized_prompts).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CLIP_scoring(nn.Module):
    def __init__(self, clip_model, train_preprocess, val_preprocess, tokenized_prompts):
        super().__init__()
        # self.prompt_learner = PromptLearner(args, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.tokenized_prompts = tokenized_prompts

    def prepare_id(self, tokenized_prompts_id):
        with torch.no_grad():
            text_features = self.text_encoder(tokenized_prompts_id)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        cos_sim = text_features @ self.text_features.t()
        return cos_sim

    def forward(self, tokenized_prompts_wordnet):
        with torch.no_grad():
            text_features_wordnet = self.text_encoder(tokenized_prompts_wordnet)

            text_features_wordnet = text_features_wordnet / text_features_wordnet.norm(dim=-1, keepdim=True)

            cos_sim = text_features_wordnet @ self.text_features.t()

        return cos_sim

def Deduplication(listone, listtwo):
    new_list = []
    for item in listone:
        if item in listtwo:
            pass
        else:
            new_list.append(item)
    return new_list

def generate_cossim_idname_wordnet_dedup(classnames, save_path, text_root=None):
    print('starting generating cosine similarity with ID classnames and wordnet names..........')

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected. WordNet preprocessing will be VERY slow on CPU!")
        print("This may take several hours. Consider running on a machine with GPU.")

    path = text_root or os.path.dirname(save_path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"WordNet text directory not found: {path}")
    prompts = ["The nice " + name + "." for name in classnames]
    # prompts = [name for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    text_files = os.listdir(path)
    adj_list = []
    noun_list = []

    for text_file in text_files:
        if text_file[:3] == 'adj' and text_file[-3:] == 'txt':
            file_path = os.path.join(path, text_file)
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
            except FileNotFoundError:
                print(f"The file {file_path} was not found. Please make sure the file path is correct.")
            adj_list += text.split('\n')[:-1]
        elif text_file[:4] == 'noun' and text_file[-3:] == 'txt':
            file_path = os.path.join(path, text_file)
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
            except FileNotFoundError:
                print(f"The file {file_path} was not found. Please make sure the file path is correct.")
            noun_list += text.split('\n')[:-1]
        else:
            print('skip a not text file', text_file)
    adj_list = list(set(adj_list))
    noun_list = list(set(noun_list))
    print('length of adj words', len(adj_list))
    print('length of noun words', len(noun_list))

    adj_list = Deduplication(adj_list, classnames)
    noun_list = Deduplication(noun_list, classnames)
    print('length of adj words after image dedup', len(adj_list))
    print('length of noun words after image dedup', len(noun_list))
    adj_list = Deduplication(adj_list, noun_list)  ## when a word is both a noun and a adj, remain the noun only.   make no influence, but cost much time.
    print('length of adj words after noun dedup', len(adj_list))

    # all_list = adj_list + noun_list
    # prompts_noun = ["this is a " + name + " photo." for name in noun_list]
    # prompts_noun = [name for name in noun_list]
    prompts_noun = ["The nice " + name + "." for name in noun_list]
    tokenized_prompts_noun = torch.cat([clip.tokenize(p) for p in prompts_noun]).to(device)

    prompts_adj = ["This is a " + name + " photo." for name in adj_list]
    # prompts_adj = [name for name in adj_list]
    tokenized_prompts_adj = torch.cat([clip.tokenize(p) for p in prompts_adj]).to(device)

    # prompts_wordnet = [name for name in all_list]
    # tokenized_prompts_wordnet = torch.cat([clip.tokenize(p) for p in prompts_wordnet]).to(device)

    # clip.available_models()
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    model, val_preprocess = clip.load('ViT-B/16', device, jit=False)  ## original use ViT-L/14

    # class_num=1000
    # img_path=path
    clip_selector = CLIP_scoring(model, val_preprocess, val_preprocess, tokenized_prompts)
    cos_sim_id = clip_selector.prepare_id(tokenized_prompts) ### cosine similarity between ID class names.
    batch_size = 100


    total_iter = math.ceil(len(adj_list) / batch_size)
    can_list_adj = []
    can_cos_adj = []
    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)
        if i == total_iter - 1:
            current_list = adj_list[i*batch_size:]
            current_token = tokenized_prompts_adj[i*batch_size:, :]
        else:
            current_list = adj_list[i*batch_size:(i+1)*batch_size]
            current_token = tokenized_prompts_adj[i*batch_size:(i+1)*batch_size]
        cos_sim = clip_selector(current_token) ## batch * 1000
        can_list_adj += current_list
        can_cos_adj.append(cos_sim)
    can_cos_adj = torch.cat(can_cos_adj, dim=0)

    total_iter = math.ceil(len(noun_list) / batch_size)
    can_list_noun = []
    can_cos_noun = []
    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)
        if i == total_iter - 1:
            current_list = noun_list[i*batch_size:]
            current_token = tokenized_prompts_noun[i*batch_size:, :]
        else:
            current_list = noun_list[i*batch_size:(i+1)*batch_size]
            current_token = tokenized_prompts_noun[i*batch_size:(i+1)*batch_size]
        cos_sim = clip_selector(current_token) ## batch * 1000
        can_list_noun += current_list
        can_cos_noun.append(cos_sim)
    can_cos_noun = torch.cat(can_cos_noun, dim=0)

    save_dict = {}
    save_dict['text_list_adj'] =  can_list_adj
    save_dict['cos_sim_adj'] = can_cos_adj.cpu()

    save_dict['text_list_noun'] =  can_list_noun
    save_dict['cos_sim_noun'] = can_cos_noun.cpu()

    save_dict['cos_sim_id'] = cos_sim_id.cpu()
    # ipdb.set_trace()

    torch.save(save_dict, save_path)
