import os
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPSv2Scorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision=self.dtype,
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
        # force download of model via score
        hpsv2.score([], "")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

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
        text = self.tokenizer(prompts).to(self.device)
        outputs = self.model(images, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        scores = torch.diagonal(logits_per_image)

        return scores
