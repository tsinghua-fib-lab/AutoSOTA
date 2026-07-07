from typing import List, Optional

import torch
import torch.nn as nn

import clip
from prompter import Prompter


class ClipPrompts(nn.Module):
    def __init__(
        self,
        viz_backbone: str,
        img_size: int,
        train_class_names: List,
        test_class_names: List,
        per_class_prompts: bool,
        prompt_position: str,
        number_of_prompts: int,
        use_conditioning: bool,
        softmax_temp: Optional[float] = None,
        **_,
    ):
        super().__init__()

        assert viz_backbone in clip.available_models()

        clip_model, _ = clip.load(viz_backbone)

        img_size_clip = clip_model.visual.input_resolution
        assert (
            clip_model.visual.input_resolution == img_size
        ), f"Image size must be {img_size_clip}"

        # Get CLIP componenets
        self.clip_dtype = clip_model.dtype

        self.text_encoder = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.final_layer_norm = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.softmax_temp = softmax_temp

        self.use_conditioning = use_conditioning

        self.prompter = Prompter(
            train_class_names=train_class_names,
            test_class_names=test_class_names,
            clip_model=clip_model,
            per_class_prompts=per_class_prompts,
            prompt_position=prompt_position,
            number_of_prompts=number_of_prompts,
            use_conditioning=use_conditioning,
        )

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        self.image_encoder.eval()

    @property
    def trainable_parameters(self):
        return self.prompter.trainable_parameters

    def encode_image(self, image):
        orig_type = image.dtype
        image_features = self.image_encoder(image.to(self.clip_dtype))
        image_features = image_features.to(orig_type)
        return image_features

    def encode_text(self, embeddings):
        input_ndim = embeddings.ndim

        if input_ndim == 4:
            # img conditioning, we have a set of prompts per img
            batch_size, num_classes, context, emb_dim = embeddings.shape
            embeddings = embeddings.reshape(-1, context, emb_dim)

        embeddings = embeddings + self.positional_embedding

        embeddings = embeddings.permute(1, 0, 2)  # NLD -> LND
        orig_type = embeddings.dtype
        embeddings = self.text_encoder(embeddings.to(self.clip_dtype))
        embeddings = embeddings.to(orig_type)

        embeddings = embeddings.permute(1, 0, 2)  # LND -> NLD
        embeddings = self.final_layer_norm(embeddings)

        if input_ndim == 4:
            eot_token_pos = self.prompter.eot_token_pos.repeat(batch_size)
            embeddings = embeddings[torch.arange(
                embeddings.shape[0]), eot_token_pos]
            # resahpe back
            embeddings = embeddings.reshape(batch_size, num_classes, emb_dim)
        else:
            embeddings = embeddings[
                torch.arange(embeddings.shape[0]), self.prompter.eot_token_pos
            ]

        embeddings = embeddings @ self.text_projection
        return embeddings

    def forward(self, image):
        image_features = self.encode_image(image)

        input_embeddings = self.prompter(
            image_features=image_features if self.use_conditioning else None
        )

        text_features = self.encode_text(input_embeddings)

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        if text_features.ndim == 3:
            logits = torch.einsum(
                "bd, bcd -> bc", image_features, text_features)
        else:
            logits = image_features @ text_features.t()

        logit_scale = (
            self.softmax_temp if self.softmax_temp else 1.0 / self.logit_scale.exp()
        )
        logits = logits / logit_scale

        return logits


class VideoClipPrompts(ClipPrompts):
    def __init__(self, num_frames: int, frame_aggregation: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_frames = num_frames
        self.img_emb_size = self.image_encoder.output_dim
        self.frame_aggregation = frame_aggregation

        aggregation_type = frame_aggregation.split("_")[0]
        assert aggregation_type in ["mean", "max", "transformer"]

        if aggregation_type == "transformer":
            num_layers = int(frame_aggregation.split("_")[1])

            self.temporal_trsf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.img_emb_size,
                    nhead=8,
                    dim_feedforward=self.img_emb_size * 4,
                    dropout=0.1,
                    activation="gelu",
                ),
                num_layers=num_layers,
                norm=nn.LayerNorm(self.img_emb_size),
            )

            self.temporal_emb = nn.Embedding(
                self.num_frames, self.img_emb_size).weight

            # init from N(O, O.O1)
            nn.init.normal_(self.temporal_emb, std=0.01)
            self.temporal_trsf.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    @property
    def trainable_parameters(self):
        if "transformer" not in self.frame_aggregation:
            return super().trainable_parameters
        return super().trainable_parameters + ["temporal_trsf", "temporal_emb"]

    def encode_image(self, video):
        assert video.ndim == 5, "Video must be of shape (B, C, T, H, W)"
        assert (
            video.shape[2] == self.num_frames
        ), f"Video ({video.shape}) must have {self.num_frames} frames"

        B, C, T, H, W = video.size()
        video = video.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCWH
        video = video.reshape(-1, C, H, W)

        orig_type = video.dtype
        video_features = self.image_encoder(video.to(self.clip_dtype))
        video_features = video_features.to(orig_type)

        video_features = video_features.reshape(B, T, self.img_emb_size)

        if self.frame_aggregation == "mean":
            return video_features.mean(dim=1)
        if self.frame_aggregation == "max":
            return video_features.max(dim=1)[0]

        video_features = video_features + self.temporal_emb[None]
        video_features = video_features.permute(1, 0, 2)  # NLD -> LND
        video_features = self.temporal_trsf(video_features)
        video_features = video_features.permute(1, 0, 2)  # LND -> NLD

        return video_features.mean(dim=1)
