import torch
import argparse
parser = argparse.ArgumentParser()
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import os
import torch.nn.functional as F
import json
import numpy as np
from utils import *
from datasets import load_dataset
from collections import defaultdict
import time
import pickle
start_time = time.time()
# parameters of experimental setting
parser.add_argument('--N1',    default=250,    help = 'Size of each sample')
parser.add_argument('--TN1',    default=20000,    help = 'Size of training sample')
parser.add_argument('--rs',    default=283,    help = 'Random seed')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')

parser.add_argument('--perturb_steps',  default=20,           help='perturb_steps')
parser.add_argument('--adv_levels',  default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],           help='perturb_steps')
parser.add_argument('--loss_fn',  default="cent",           help='loss_fn')
parser.add_argument('--category',  default="Madry",           help='category')

args = parser.parse_args()

num_classes = 1000

resnet50 = models.resnet50(pretrained=True).cuda()
model_without_fc = nn.Sequential(*list(resnet50.children())[:-1])
model_without_fc.eval()

transform = transforms.Compose([
    transforms.Resize(256),  # Resize the shortest side to 256 pixels
    transforms.CenterCrop(224),  # Crop the center to get a 224x224 image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
])

class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if self.transform:
            try:
                image = self.transform(image)
            except:
                return self.__getitem__((idx + 1) % len(self.dataset))
                
        return image, label

class LabelBatchSampler(Sampler):
    def __init__(self, label_to_indices, batch_size):
        self.label_to_indices = label_to_indices
        self.batch_size = batch_size
        self.labels = list(label_to_indices.keys())

    def __iter__(self):
        for label in self.labels:
            indices = self.label_to_indices[label]
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.label_to_indices.values())

dataset = datasets.ImageFolder(root='/data/imagenetv2', transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
margin_imagenetv2 = []
acc_imagenetv2 = []
imagenetv2_Fea=torch.tensor([]).to(args.device)
for X, labels in dataloader:
    X, labels = X.detach().to(args.device), labels.detach().to(args.device)
    pred = resnet50(X)
    softmax_pred = F.softmax(pred, dim=1)
    margin_imagenetv2.append(sum([1-softmax_pred[i][labels[i]] for i in range(10)]).item())
    acc_imagenetv2.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(10)]).item())
    one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    ww = one_hot @ resnet50.fc.weight.detach()
    imagenetv2_Fea = torch.cat([imagenetv2_Fea, (model_without_fc(X).detach().reshape(10, -1) * ww.detach())], dim=0)
with open('margin_imagenetv2.json', 'w') as f:
    json.dump(margin_imagenetv2, f)
with open('acc_imagenetv2.json', 'w') as f:
    json.dump(acc_imagenetv2, f)
torch.save(imagenetv2_Fea, 'imagenetv2_Fea.pt')
del imagenetv2_Fea, margin_imagenetv2, acc_imagenetv2

dataset = datasets.ImageFolder(root='/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)
margin_imagenet = []
acc_imagenet = []
imagenet_Fea=torch.tensor([]).to(args.device)
for X, labels in dataloader:
    X, labels = X.detach().to(args.device), labels.detach().to(args.device)
    pred = resnet50(X)
    softmax_pred = F.softmax(pred, dim=1)
    margin_imagenet.append(sum([1-softmax_pred[i][labels[i]] for i in range(50)]).item())
    acc_imagenet.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(50)]).item())
    one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    ww = one_hot @ resnet50.fc.weight.detach()
    imagenet_Fea = torch.cat([imagenet_Fea, (model_without_fc(X).detach().reshape(50, -1) * ww.detach())], dim=0)
with open('margin_imagenet.json', 'w') as f:
    json.dump(margin_imagenet, f)
with open('acc_imagenet.json', 'w') as f:
    json.dump(acc_imagenet, f)
torch.save(imagenet_Fea, 'imagenet_Fea.pt')
del imagenet_Fea, margin_imagenet, acc_imagenet

imageneta = load_dataset("barkermrl/imagenet-a", split='train', cache_dir='/data')
imageneta = ImageNetDataset(imageneta, transform=transform)
label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(imageneta):
    label_to_indices[label].append(idx)
with open("imageneta_label_to_indices.pkl", "wb") as f:
    pickle.dump(label_to_indices, f)
with open("imageneta_label_to_indices.pkl", "rb") as f:
    label_to_indices = pickle.load(f)
sampler = LabelBatchSampler(label_to_indices, 1000)
dataloader = DataLoader(imageneta, batch_sampler=sampler, shuffle=False, num_workers=0)
margin_imageneta = []
acc_imageneta = []
imageneta_Fea=torch.tensor([]).to(args.device)
for X, labels in dataloader:
    X, labels = X.detach().to(args.device), labels.detach().to(args.device)
    pred = resnet50(X)
    softmax_pred = F.softmax(pred, dim=1)
    margin_imageneta.append(sum([1-softmax_pred[i][labels[i]] for i in range(len(labels))]).item())
    acc_imageneta.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(len(labels))]).item())
    one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    ww = one_hot @ resnet50.fc.weight.detach()
    imageneta_Fea = torch.cat([imageneta_Fea, (model_without_fc(X).detach().reshape(len(labels), -1) * ww.detach())], dim=0)
    print(labels[0])
with open('margin_imageneta.json', 'w') as f:
    json.dump(margin_imageneta, f)
with open('acc_imageneta.json', 'w') as f:
    json.dump(acc_imageneta, f)
torch.save(imageneta_Fea, 'imageneta_Fea.pt')
del imageneta_Fea, margin_imageneta, acc_imageneta
print("imageneta done!")

imagenetsk = load_dataset("imagenet_sketch",  split='train', cache_dir='/data')
imagenetsk = ImageNetDataset(imagenetsk, transform=transform)
label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(imagenetsk):
    label_to_indices[label].append(idx)
with open("imagenetsk_label_to_indices.pkl", "wb") as f:
    pickle.dump(label_to_indices, f)
with open("imagenetsk_label_to_indices.pkl", "rb") as f:
    label_to_indices = pickle.load(f)
sampler = LabelBatchSampler(label_to_indices, 1000)
dataloader = DataLoader(imagenetsk, batch_sampler=sampler, shuffle=False, num_workers=0)
margin_imagenetsk = []
acc_imagenetsk = []
imagenetsk_Fea=torch.tensor([]).to(args.device)
for X, labels in dataloader:
    X, labels = X.detach().to(args.device), labels.detach().to(args.device)
    pred = resnet50(X)
    softmax_pred = F.softmax(pred, dim=1)
    margin_imagenetsk.append(sum([1-softmax_pred[i][labels[i]] for i in range(len(labels))]).item())
    acc_imagenetsk.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(len(labels))]).item())
    one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    ww = one_hot @ resnet50.fc.weight.detach()
    imagenetsk_Fea = torch.cat([imagenetsk_Fea, (model_without_fc(X).detach().reshape(len(labels), -1) * ww.detach())], dim=0)
    print(labels[0])
with open('margin_imagenetsk.json', 'w') as f:
    json.dump(margin_imagenetsk, f)
with open('acc_imagenetsk.json', 'w') as f:
    json.dump(acc_imagenetsk, f)
torch.save(imagenetsk_Fea, 'imagenetsk_Fea.pt')
del imagenetsk_Fea, margin_imagenetsk, acc_imagenetsk
print("imagenetsk done!")

imagenetr = load_dataset("axiong/imagenet-r",  split='test', cache_dir='/data')
with open('imagenet_class_index.json', 'r') as f:
    imagenet_class_index = json.load(f)
wnid_label = defaultdict(int)
for label in list(imagenet_class_index.keys()):
    wnid_label[imagenet_class_index[label][0]] = int(label)
def add_label(example):
    example["label"] = wnid_label.get(example["wnid"], "Unknown")
    return example
imagenetr = imagenetr.map(add_label)
imagenetr = imagenetr.sort('label')
imagenetr = ImageNetDataset(imagenetr, transform=transform)
label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(imagenetr):
    label_to_indices[label].append(idx)
with open("imagenetr_label_to_indices.pkl", "wb") as f:
    pickle.dump(label_to_indices, f)
with open("imagenetr_label_to_indices.pkl", "rb") as f:
    label_to_indices = pickle.load(f)
sampler = LabelBatchSampler(label_to_indices, 1000)
dataloader = DataLoader(imagenetr, batch_sampler=sampler, shuffle=False, num_workers=0)
margin_imagenetr = []
acc_imagenetr = []
imagenetr_Fea=torch.tensor([]).to(args.device)
for X, labels in dataloader:
    X, labels = X.detach().to(args.device), labels.detach().to(args.device)
    pred = resnet50(X)
    softmax_pred = F.softmax(pred, dim=1)
    margin_imagenetr.append(sum([1-softmax_pred[i][labels[i]] for i in range(len(labels))]).item())
    acc_imagenetr.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(len(labels))]).item())
    one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    ww = one_hot @ resnet50.fc.weight.detach()
    imagenetr_Fea = torch.cat([imagenetr_Fea, (model_without_fc(X).detach().reshape(len(labels), -1) * ww.detach())], dim=0)
    print(labels[0])
with open('margin_imagenetr.json', 'w') as f:
    json.dump(margin_imagenetr, f)
with open('acc_imagenetr.json', 'w') as f:
    json.dump(acc_imagenetr, f)
torch.save(imagenetr_Fea, 'imagenetr_Fea.pt')
del imagenetr_Fea, margin_imagenetr, acc_imagenetr
print("imagenetr done!")

# dataset = datasets.ImageFolder(root='/Imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
# dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)
# noise_levels = np.arange(1, 21, step=1.0)
# for noise in noise_levels:
#     noise = round(noise, 1)
#     margin_imagenet = []
#     acc_imagenet = []
#     imagenet_Fea=torch.tensor([]).to(args.device)
#     for X, labels in dataloader:
#         X, labels = X.detach().to(args.device), labels.detach().to(args.device)
#         # noise_X = torch.randn(X.shape,device=args.device) + noise
#         noise_X = torch.randn(X.shape,device=args.device) * noise
#         pred = resnet50(X+noise)
#         softmax_pred = F.softmax(pred, dim=1)
#         margin_imagenet.append(sum([1-softmax_pred[i][labels[i]] for i in range(50)]).item())
#         acc_imagenet.append(sum([torch.argmax(softmax_pred[i]) == labels[i] for i in range(50)]).item())
#         one_hot = torch.zeros(labels.size(0), num_classes, device=args.device)
#         one_hot.scatter_(1, labels.unsqueeze(1), 1)
#         ww = one_hot @ resnet50.fc.weight.detach()
#         imagenet_Fea = torch.cat([imagenet_Fea, (model_without_fc(X+noise).detach().reshape(50, -1) * ww.detach())], dim=0)
#     with open('margin_imagenet_'+str(noise)+'.json', 'w') as f:
#         json.dump(margin_imagenet, f)
#     with open('acc_imagenet_'+str(noise)+'.json', 'w') as f:
#         json.dump(acc_imagenet, f)
#     torch.save(imagenet_Fea, 'imagenet_Fea_'+str(noise)+'.pt')
#     del imagenet_Fea, margin_imagenet, acc_imagenet
    
# 结束计时
end_time = time.time()
# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time} 秒")