import os
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, CLIPProcessor


class PickScore(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

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
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image_embeds = self.model.get_image_features(images)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores
