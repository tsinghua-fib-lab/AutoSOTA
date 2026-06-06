import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torchvision.transforms.functional import resize
import mate3d.model.clip as clip
from mate3d.model.networks import TextEncoder, PromptLearner, HyperNet, TargetNet


def parse_args():
    args = argparse.Namespace()
    args.n_ctx = 12
    args.ctx_init = ''
    args.class_token_position = 'front'
    args.csc = True
    args.prec = "fp32"
    args.subsample_classes = "all"
    return args


class HyperScore(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.args = parse_args()

        self.clip_model, _ = clip.load("ViT-B/16")
        self.clip_model = self.clip_model.to(self.device, dtype=self.dtype)
        self.prompt_learner = PromptLearner(
            self.device,
            self.args,
            ['alignment quality', 'geometry quality', 'texture quality', 'overall quality'],
            self.clip_model,
        ).to(self.device, dtype=self.dtype)
        self.tokenized_prompt = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)
        self.fc_quality = nn.Linear(512, 224).to(self.device, dtype=self.dtype)
        self.fc_condition = nn.Linear(512, 5488).to(self.device, dtype=self.dtype)
        self.hypernet = HyperNet().to(self.device, dtype=self.dtype)
        self.relu = nn.ReLU()

        state_dict = torch.load("mate3d/checkpoint/model.pth", map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def image_preprocess(self, images):
        images = resize(images, size=[224, 224])
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images

    @torch.no_grad()
    def __call__(self, images, prompts, num_views):
        batch_size = images.shape[0] // num_views
        images = self.image_preprocess(images.to(self.device, dtype=self.dtype))
        prompts = clip.tokenize(prompts).to(self.device)
        prompt_embedding = self.clip_model.token_embedding(prompts).to(self.device)

        feature_texture = self.clip_model.encode_image_allpatch(images)
        num_patches = feature_texture.shape[1]
        feature_texture = feature_texture.reshape(batch_size, num_views * num_patches, -1)
        feature_prompt_eot, feature_prompt = self.text_encoder(prompt_embedding, prompts)

        prompt_learner = self.prompt_learner()
        feature_condition, _ = self.text_encoder(prompt_learner, self.tokenized_prompt)
        feature_condition_expand = feature_condition.repeat(batch_size, 1, 1)

        sim_texture = einsum(
            'b i d, b j d -> b j i',
            F.normalize(feature_prompt, dim=2),
            F.normalize(feature_texture, dim=2),
        )
        sim_cond = einsum(
            'b i d, b j d -> b j i',
            F.normalize(feature_prompt, dim=2),
            F.normalize(feature_condition_expand, dim=2)
        )
        patch_weight = einsum('b i d, b j d -> b j i', sim_cond, sim_texture)
        patch_weight = F.softmax(patch_weight, dim=1)

        feature_condition = self.fc_condition(feature_condition)
        feature_condition = feature_condition.reshape(-1, 112, 7, 7)

        score_list = []
        for i in range(0, feature_condition.shape[0]):
            feature_texture_fused = torch.sum(feature_texture * patch_weight[:, :, i].unsqueeze(2), dim=1)
            feature_fused = torch.mul(feature_texture_fused, feature_prompt_eot)
            feature_quality = self.fc_quality(feature_fused).unsqueeze(2).unsqueeze(3)
            param = self.hypernet(feature_condition[i])

            targetnet = TargetNet(param)
            score = targetnet(feature_quality)
            score_list.append(score.flatten())

        return score_list
