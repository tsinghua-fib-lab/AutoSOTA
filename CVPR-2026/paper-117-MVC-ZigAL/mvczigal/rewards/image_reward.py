import os
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from ImageReward import ImageReward_download


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class ImageRewardScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        download_root = f"{os.path.expanduser('~')}/.cache/ImageReward"
        config_path = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", download_root)
        model_path = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt", download_root)
        # config_path = os.path.join(download_root, "med_config.json")
        # model_path = os.path.join(download_root, "ImageReward.pt")

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=config_path).to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        self.eval()

    @torch.no_grad()
    def image_preprocess(self, images):
        images = resize(images, size=[224, 224], interpolation=InterpolationMode.BICUBIC)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images

    @torch.no_grad()
    def __call__(self, images, prompts):
        images = self.image_preprocess(images.to(self.device, dtype=self.dtype))
        image_embeds = self.blip.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_input = self.blip.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt"
        ).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].to(dtype=self.dtype)
        scores = self.mlp(txt_features).squeeze(1)
        scores = (scores - self.mean) / self.std

        return scores
