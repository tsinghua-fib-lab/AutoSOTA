import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224
from timm.models.mlp_mixer import mixer_b16_224, mixer_l16_224


import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from collections import defaultdict

def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model
    
def load_blip_to_cpu(backbone_name, prec):
    assert backbone_name.startswith("BLIP-"), f"Backbone name must start with 'BLIP-', got {backbone_name}"
    model_type = backbone_name[len("BLIP-"):].lower()

    try:
        from lavis.models import load_model_and_preprocess
    except ImportError:
        raise ImportError("please install salesforce-lavis: pip install salesforce-lavis")

    if model_type == "base":
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip_feature_extractor",
            model_type="base",
            is_eval=True,
            device="cpu"
        )
    elif model_type == "large":
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip_feature_extractor",
            model_type="large",
            is_eval=True,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown BLIP variant: {model_type}")

    if prec == "fp16":
        model = model.half()
    else:
        model = model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-Mixer-B/16":
        model = mixer_b16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-Mixer-L/16":
        model = mixer_l16_224(pretrained=True).eval()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        # 进行线性变换
        return F.linear(x, weight)
class Trainer:
    def __init__(self, cfg):

        self.text_features = None
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(resolution),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])
        transform_train = [transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
            transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])]

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_samples = train_dataset.num_samples
        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.clean_cls_num_list = train_dataset.clean_cls_num_list
        self.classnames = train_dataset.classnames
        self.total_labels = train_dataset.targets

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        # self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        # self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        # self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]
        self.many_idxs = train_dataset.many_idxs
        self.med_idxs = train_dataset.med_idxs
        self.few_idxs = train_dataset.few_idxs

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP-BLIP"):
            print(f"Loading CLIP-BLIP (backbone: {cfg.backbone})")
            self.clip_model = load_clip_to_cpu(cfg.backbone1, cfg.prec)
            cfg.backbone = cfg.backbone1
            self.model = PeftModelFromCLIP(cfg, self.clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

            self.blip_model = load_blip_to_cpu(cfg.backbone2, cfg.prec)
            cfg.backbone = cfg.backbone2    # 切换到 BLIP 分支
            self.model2 = PeftModelFromCLIP(cfg, self.blip_model, num_classes)
            self.model2.to(self.device)
            self.tuner2 = self.model2.tuner
            self.head2 = self.model2.head

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            self.clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, self.clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("BLIP"):
            print(f"Loading BLIP (backbone: {cfg.backbone})")
            self.blip_model = load_blip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, self.blip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT") or cfg.backbone.startswith("IN21K-Mixer"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            # self.customlinear = CustomLinear(512, 768, self.model.image_encoder.dtype).to(self.device)
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)
        # print("Turning on gradients in the model")
        # for name, param in self.model.named_parameters():
        #     param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        param_groups = [{"params": self.head.parameters()},
                        {"params": self.tuner.parameters()}]
        if hasattr(self, 'head2') and self.head2 is not None:
            param_groups.append({"params": self.head2.parameters()})
        if hasattr(self, 'tuner2') and self.tuner2 is not None:
            param_groups.append({"params": self.tuner2.parameters()})
        self.optim = torch.optim.SGD(param_groups,
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "smoothing":
            self.criterion = LabelSmoothingLoss()
        elif cfg.loss_type == "GCL":
            self.criterion = GCLLoss()

        self.needs_cls_num = (cfg.loss_type == "LA")
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    def get_tokenized_prompts_blip(self, classnames, template):
        # 1. 拼接文本并替换下划线为空格
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        
        # 2. 调用 BLIP 自带的 BERT 分词器
        tokenized_output = self.blip_model.tokenizer(
            prompts,
            padding="max_length",  # 补齐长度，防止不同长度的句子无法组成 batch
            truncation=True,       # 遇到超长文本自动截断
            max_length=77,         # ⭐️ 关键：保持与 CLIP 一致的 77 长度，完美骗过框架
            return_tensors="pt"    # 直接输出 PyTorch 的 tensor
        )
        
        # 3. 提取 input_ids (即 Token ID 张量) 并移动到显卡上
        prompts_ids = tokenized_output.input_ids.to(self.device)
        
        return prompts_ids

    # def get_tokenized_prompts(self, classnames, template):
    #     prompts = [template.format(c.replace("_", " ")) for c in classnames]
    #     return prompts  # 返回 list[str]

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            if cfg.backbone.startswith("BLIP"):
                prompts = self.get_tokenized_prompts_blip(classnames, template)
            else:
                prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.text_features = text_features.to(self.device)
        self.head.apply_weight(self.text_features)
        # self.head2.apply_weight(self.text_featuresb)
        # self.text_features = text_features.to(self.device)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)
        self.head2.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def compute_candidate_correction(self, epoch_idx, output_softmax, output2_softmax, label, clean_label, indices, candidate_count, total_candi, cls_num_list_epoch):
        """Update candidate_count based on CLIP, zero-shot."""
        candidate_count[indices, label] += 1

        diff_zeroshot = 0
        for i in range(len(indices)):
            if epoch_idx == 0:
                class_id = label[i].item()
            else:
                class_id = total_candi[indices[i]].item()

            k = int(cls_num_list_epoch[class_id] ** 0.25)
            if k > self.num_classes / 2:
                k = self.num_classes // 4

            prob, pred = output_softmax[i].topk(self.num_classes, 0, True, True)
            prob1, pred1 = output2_softmax[i].topk(self.num_classes, 0, True, True)
            w = prob[:k].sum()
            w1 = prob1[:k].sum()

            if pred1[0] != clean_label[i]:
                diff_zeroshot += 1

            for j in range(k):
                if label[i] == pred1[j] and label[i] == pred[j]:
                    candidate_count[indices[i], pred[j]] += (w.item() + w1.item())
                elif label[i] == pred1[j] and label[i] != pred[j]:
                    candidate_count[indices[i], pred1[j]] += w1.item()
                elif label[i] != pred1[j] and label[i] == pred[j]:
                    candidate_count[indices[i], pred[j]] += w.item()
                else:
                    candidate_count[indices[i], pred[j]] += prob[j].item()
                    candidate_count[indices[i], pred1[j]] += prob1[j].item()

        _, y_candi = torch.max(candidate_count[indices], dim=-1)
        return y_candi, diff_zeroshot

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model = self.clip_model
        model2 = ZeroShotCLIP(clip_model)
        model2.to(self.device)
        template = "a photo of a {}."
        classnames = self.classnames
        prompts = self.get_tokenized_prompts(classnames, template)
        model2.init_text_features(prompts)

        # print(f"Loading BLIP (backbone: {cfg.backbone})")
        # blip_model = self.blip_model
        # model2 = ZeroShotBLIP(blip_model, self.device)
        # model2.to(self.device)
        # template = "a photo of a {}."
        # classnames = self.classnames
        # prompts = self.get_tokenized_prompts_blip(classnames, template)
        # model2.init_text_features(prompts)

        candidate_count = torch.zeros(self.num_samples, self.num_classes).cuda()
        total_candi = torch.zeros(self.num_samples).cuda()
        cls_num_list_epoch = self.cls_num_list
        num_epochs = cfg.num_epochs
        best_acc = 0
        best_epoch_idx = -1
        best_result = {}
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            if self.head is not None:
                self.head.train()
            end = time.time()
            candidate_count_epoch = candidate_count

            diff_zeroshot = 0
            obs_cle_diff = np.zeros(self.num_classes, dtype=int).tolist()
            can_cle_diff = np.zeros(self.num_classes, dtype=int).tolist()
            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                image2 = batch[1].squeeze(1)
                label = batch[2]
                indices = batch[3]
                clean_label = batch[4]
                image = image.to(self.device)
                image2 = image2.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with (((autocast()))):
                        output = self.model(image)
                        output_softmax = output.softmax(dim=-1)

                        with torch.no_grad():
                            output2 = model2(image2).to(self.device)
                            output2_softmax = output2.softmax(dim=-1)

                        y_candi, diff_zeroshot = self.compute_candidate_correction(
                            epoch_idx,
                            output_softmax,
                            output2_softmax,
                            label,
                            clean_label,
                            indices,
                            candidate_count,
                            total_candi,
                            cls_num_list_epoch,
                        )

                        for i in range(len(indices)):
                            true_label = clean_label[i]
                            noisy_label = label[i]
                            pred_label = y_candi[i]
                            if true_label != pred_label:
                                can_cle_diff[pred_label.item()] += 1
                            if true_label != noisy_label:
                                obs_cle_diff[noisy_label.item()] += 1

                        # 更新类别计数
                        total_candi = torch.argmax(candidate_count_epoch, dim=-1)
                        cls_num_list = torch.bincount(total_candi, minlength=self.num_classes).float()
                        cls_num_list[cls_num_list == 0] = self.cls_num_list[self.num_classes - 1]
                        cls_num_list_epoch = cls_num_list.tolist()


                        if self.needs_cls_num:
                            loss = self.criterion(output, y_candi, cls_num_list.cuda())
                        else:
                            loss = self.criterion(output, y_candi)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()

                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()

            print("origin noisy rate:", [round(a / b, 2) for a, b in zip(obs_cle_diff, self.cls_num_list)])
            print("repair noisy rate", [round(a / b, 2) for a, b in zip(can_cle_diff, cls_num_list_epoch)])
            print("sum_noisy_samples_origin:", sum(obs_cle_diff))
            print(
                f"* many: {100*sum([obs_cle_diff[i] for i in self.many_idxs])/sum([self.cls_num_list[i] for i in self.many_idxs]):.1f}%  ")
            if self.med_idxs.size != 0:
                print( f"med: {100*sum([obs_cle_diff[i] for i in self.med_idxs])/sum([self.cls_num_list[i] for i in self.med_idxs]):.1f}%  ")
            if self.few_idxs.size != 0:
                print(f"few: {100*sum([obs_cle_diff[i] for i in self.few_idxs])/sum([self.cls_num_list[i] for i in self.few_idxs]):.1f}%")
            print("sum_noisy_samples_repair:", sum(can_cle_diff))
            print(
                f"* many: {100 * sum([can_cle_diff[i] for i in self.many_idxs]) / sum([cls_num_list_epoch[i] for i in self.many_idxs]):.1f}%  ")
            if self.med_idxs.size != 0:
                print(f"med: {100 * sum([can_cle_diff[i] for i in self.med_idxs]) / sum([cls_num_list_epoch[i] for i in self.med_idxs]):.1f}%  ")
            if self.few_idxs.size != 0:
                print(f"few: {100 * sum([can_cle_diff[i] for i in self.few_idxs]) / sum([cls_num_list_epoch[i] for i in self.few_idxs]):.1f}%")
            print("cls_num_list_clean:", self.clean_cls_num_list)
            print("cls_num_list_origin:", self.cls_num_list)
            print("cls_num_list_repair:", cls_num_list_epoch)
            print("cls_noise_num_list_origin:", obs_cle_diff)
            print("cls_clean_num_list_origin:", [a - b for a, b in zip(self.cls_num_list, obs_cle_diff)])
            print("cls_noise_num_list_repair:", can_cle_diff)
            print("cls_clean_num_list_repair:", [a - b for a, b in zip(cls_num_list_epoch, can_cle_diff)])
            print("zeroshot_accuracy:", 1-round(diff_zeroshot/self.num_samples, 2))

            self.test()

            if self.results["mean_acc"] > best_acc:
                best_acc = self.results["mean_acc"]
                best_result = self.results
                self.save_model(cfg.output_dir)
                best_epoch_idx = epoch_idx

            print("*" * 50)
            print("best_epoch_idx:", best_epoch_idx)
            print(f"* best average: {best_result['mean_acc']:.1f}%")
            print(
                f"* many: {best_result['many_acc']:.1f}%  med: {best_result['med_acc']:.1f}%  few: {best_result['few_acc']:.1f}%")


        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        # self.save_model(cfg.output_dir)

        # self.test()
        print("*"*50)
        print("best result:")
        print(f"* average: {best_result['mean_acc']:.1f}%")
        print(f"* many: {best_result['many_acc']:.1f}%  med: {best_result['med_acc']:.1f}%  few: {best_result['few_acc']:.1f}%")

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader


        # total_feature = []
        # all_label = []
        # tsne = TSNE(n_components=2, random_state=0, perplexity=100, n_iter=1000, early_exaggeration=100)
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)

                # x = self.model.image_encoder.feature.cpu()
                # total_feature.append(x)
                # all_label.append(label.cpu())
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        # x = np.vstack(total_feature)
        # y = torch.cat(all_label, dim=0)
        # x_tsne = tsne.fit_transform(x)



        # plt.style.use("seaborn-v0_8-muted")
        # # 6, 4
        # plt.figure(figsize=(7, 4))
        # plt.yticks([])
        # plt.xticks([])
        # plt.yticks([])
        # for label in range(10):
        #     indices = [i for i in range(len(y)) if y[i] == label]
        #     plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], s=8, color=plt.cm.Set3(label), label=str(label),
        #                 alpha=1)
        # plt.legend(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), loc='upper right', framealpha=0.2, fontsize=15,
        #            markerscale=2.5)
        #
        # plt.tight_layout()
        # plt.savefig("figure_ir10_nr60.pdf", format='pdf')
        # plt.show()

        self.results = self.evaluator.evaluate()

        for k, v in self.results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(self.results.values())[0]

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        # model_dict = self.model.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            # "model": model_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)
