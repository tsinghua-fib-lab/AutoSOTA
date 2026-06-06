#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
from PIL import Image
from nltk.tokenize import word_tokenize
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='GLPN-LLM')
    parser.add_argument('--dataset_name', type=str, default='weibo', help='dataset name (weibo, twitter, pheme)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--threshold', type=float, default=0.95, help='similarity threshold')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--label_rate', type=float, default=0.65, help='label rate for training')
    parser.add_argument('--psesudo_mask_rate', type=float, default=0.05, help='pseudo mask rate for test set')
    parser.add_argument('--llm_label_rate', type=float, default=0.35, help='rate for using LLM labels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels for UniMP')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers for UniMP')
    parser.add_argument('--heads', type=int, default=2, help='number of heads for UniMP')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--save_csv', action='store_true', help='save processed data to CSV')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    return parser.parse_args()

args = get_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)

device = torch.device(args.device)

def preprocess_text(text):
    text = text.strip()
    text = re.sub(re.compile('<.*?>'), ' ', text)
    text = word_tokenize(text)
    text = ' '.join(word for word in text if word.isalpha() or word.isnumeric() or word.isalnum())
    return text

def preprocess_event(text):
    text = text.split('_')[0]
    return text
if args.dataset_name == 'weibo':
    df = pd.read_csv('dataset/weibo/weibo_train.csv')
    df_test = pd.read_csv('dataset/weibo/weibo_test.csv')
    df['event'] = df['image_id']
    df_test['event'] = df_test['image_id']
    for i in  range(0, len(df['event'])):
        df['event'][i] = 1 
    for i in  range(0, len(df_test['event'])):
        df_test['event'][i] = 1 
    IMG_ROOT_train = "dataset/weibo/images"
    IMG_ROOT_test = "dataset/weibo/images"
if args.dataset_name == 'twitter':
    df= pd.read_csv('dataset/twitter/train_posts_clean.csv')
    df_test = pd.read_csv('dataset/twitter/test_posts.csv')
    df['event'] = df['image_id']
    for i in  range(0, len(df['label'])):
        df['label'][i] = 1 if df['label'][i] == 'real' else 0
    df.event = np.array([preprocess_event(text) for text in df.event])
    df_test['event'] = df_test['image_id']
    for i in  range(0, len(df_test['label'])):
        df_test['label'][i] = 1 if df_test['label'][i] == 'real' else 0
    df_test.event = np.array([preprocess_event(text) for text in df_test.event])
    IMG_ROOT_train = "dataset/twitter/twitter_cleaned/images_train"
    IMG_ROOT_test = "dataset/twitter/twitter_cleaned/images_test"
if args.dataset_name == 'pheme':
    df = pd.read_csv('dataset/pheme/pheme_train.csv')
    df_test = pd.read_csv('dataset/pheme/pheme_test.csv')
    IMG_ROOT_train = "dataset/pheme/pheme_image/images"
    IMG_ROOT_test = "dataset/pheme/pheme_image/images" 
    
df.rename(columns={'post_text': 'text'}, inplace=True)
df.text = np.array([preprocess_text(text) for text in df.text])
df_test.rename(columns={'post_text': 'text'}, inplace=True)
df_test.text = np.array([preprocess_text(text) for text in df_test.text])
if args.save_csv and args.dataset_name == 'weibo':
    df.to_csv('dataset/weibo/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/weibo/dataforGCN_test.csv',index=False)
if args.save_csv and args.dataset_name == 'twitter':
    df.to_csv('dataset/twitter/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/twitter/dataforGCN_test.csv',index=False)
if args.save_csv and args.dataset_name == 'pheme':
    df.to_csv('dataset/pheme/dataforGCN_train.csv',index=False)
    df_test.to_csv('dataset/pheme/dataforGCN_test.csv',index=False)


class image_caption_dataset(Dataset):
    def __init__(self, df, IMG_ROOT,  name="twitter"):
        self.dataset_name = name
        self.img_root = IMG_ROOT
        if name == "twitter":
            self.images = df["image_id"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()
        elif name == "weibo":
            self.images = df["image_id"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()
        elif name == "pheme":
            self.images = df["imgnum"].tolist()
            self.caption = df["text"].tolist()
            self.label = df["label"].tolist()
            self.event = df["event"].tolist()

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        if self.dataset_name == "twitter":
            img_path = self.img_root+'/'+self.images[idx]+'.jpg'
            try:
                images = preprocess(Image.open(img_path))
            except FileNotFoundError:
                images = preprocess(Image.new("RGB", (224, 224), color=(128, 128, 128)))
        elif self.dataset_name == "weibo":
            images = preprocess(Image.open(self.img_root+'/'+self.images[idx])) 
        elif self.dataset_name == "pheme":
            images = preprocess(Image.open(self.img_root+'/'+str(self.images[idx])+'.jpg')) 
        caption = self.caption[idx]
        label = self.label[idx]
        event = self.event[idx]
        return images, caption, label, event, idx
    
dataset = image_caption_dataset(df, IMG_ROOT_train, args.dataset_name)
dataset_test = image_caption_dataset(df_test, IMG_ROOT_test, args.dataset_name)

clip_model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
data_dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
pbar = tqdm(data_dataloader, leave=False)
ALLimage_embeds = []
ALLtext_embeds = []
ALLlabels = []
ALLids = []
ALLevents = []
for batch in pbar:
        images, texts, labels, events, idxs = batch
        images = images.to(device)
        texts = clip.tokenize(texts,truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            text_features = clip_model.encode_text(texts).float()
        ALLimage_embeds.append(image_features)
        ALLtext_embeds.append(text_features)
        ALLlabels.extend(list(labels))
        ALLevents.extend(list(events))
        ALLids.append(idxs)
ALLimage_embeds_train = torch.cat(ALLimage_embeds, dim=0)
ALLtext_embeds_train = torch.cat(ALLtext_embeds, dim=0)
ALLids = torch.cat(ALLids, dim=0)

data_dataloader = DataLoader(dataset_test, args.batch_size, shuffle=False)
pbar = tqdm(data_dataloader, leave=False)
ALLimage_embeds = []
ALLtext_embeds = []
ALLlabels = []
ALLids = []
ALLevents = []
for batch in pbar:
        images, texts, labels, events, idxs = batch
        images = images.to(device)
        texts = clip.tokenize(texts,truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images).float()
            text_features = clip_model.encode_text(texts).float()
        ALLimage_embeds.append(image_features)
        ALLtext_embeds.append(text_features)
        ALLlabels.extend(list(labels))
        ALLevents.extend(list(events))
        ALLids.append(idxs)
ALLimage_embeds_test = torch.cat(ALLimage_embeds, dim=0)
ALLtext_embeds_test = torch.cat(ALLtext_embeds, dim=0)
ALLids = torch.cat(ALLids, dim=0)

def calculate_cosine_similarity_matrix(h_emb, eps=1e-5):
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix
ALLCAT_embeds_train = torch.cat((ALLimage_embeds_train, ALLtext_embeds_train), 1)
ALLCAT_embeds_test = torch.cat((ALLimage_embeds_test, ALLtext_embeds_test), 1)
ALLCAT_embeds = torch.cat((ALLCAT_embeds_train, ALLCAT_embeds_test), 0)

ALLTEXT_embeds = torch.cat((ALLtext_embeds_train, ALLtext_embeds_test), 0)

ALLIMAGE_embeds = torch.cat((ALLimage_embeds_train, ALLimage_embeds_test), 0)

ALLimage_embeds = torch.cat((ALLimage_embeds_train, ALLimage_embeds_test), 0)
ALLtext_embeds = torch.cat((ALLtext_embeds_train, ALLtext_embeds_test), 0)

ALLCAT_embeds /= ALLCAT_embeds.norm(dim=-1, keepdim=True)
ALLimage_embeds /= ALLimage_embeds.norm(dim=-1, keepdim=True)
ALLtext_embeds /= ALLtext_embeds.norm(dim=-1, keepdim=True)

ALLCAT_similarity = (ALLCAT_embeds @ ALLCAT_embeds.T)
i2t_similarity = (ALLimage_embeds @ ALLtext_embeds.T)
t2i_similarity = (ALLtext_embeds @ ALLimage_embeds.T)
i2i_similarity = (ALLimage_embeds @ ALLimage_embeds.T)
t2t_similarity = (ALLtext_embeds @ ALLtext_embeds.T)

result_tensor = torch.zeros_like(ALLCAT_similarity)
result_tensor[(i2t_similarity > args.threshold) | (t2t_similarity > args.threshold)] = 2
result_tensor[(t2i_similarity > args.threshold) | (i2i_similarity > args.threshold)] = 3
result_tensor[ALLCAT_similarity > args.threshold] = 1
edge = result_tensor
edge_sparse = edge.to_sparse()

torch.save(edge_sparse, 'dataset/'+args.dataset_name+'/TweetGraph.pt')
torch.save(ALLCAT_embeds, 'dataset/'+args.dataset_name+'/TweetEmbeds.pt')
torch.save(ALLTEXT_embeds, 'dataset/'+args.dataset_name+'/TweetTextEmbeds.pt')
torch.save(ALLIMAGE_embeds, 'dataset/'+args.dataset_name+'/TweetImageEmbeds.pt')