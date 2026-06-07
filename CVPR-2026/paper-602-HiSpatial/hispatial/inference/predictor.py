"""HiSpatial inference predictor -- loads a trained checkpoint and answers spatial queries."""

import os
import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 448

DEFAULT_REPO_ID = "lhzzzzzy/HiSpatial-3B"


def _is_repo_id(path):
    """Check if a string looks like a HF repo id (e.g. 'user/model')."""
    return "/" in path and not os.path.exists(path)


class HiSpatialPredictor:
    """Load a trained HiSpatialVLM checkpoint and run spatial VQA inference.

    Args:
        model_load_path: Path to a local ``weights.pt`` file, a local
            directory containing ``weights.pt`` + ``config.json``, **or** a
            Hugging Face repo id (e.g. ``"lhzzzzzy/HiSpatial-3B"``).
            Defaults to ``"lhzzzzzy/HiSpatial-3B"`` which downloads the
            checkpoint automatically.
        gpu_rank: CUDA device index.
    """

    def __init__(self, model_load_path=None, gpu_rank=0, **kwargs):
        self.device = torch.device(f"cuda:{gpu_rank}")
        self.img_size = IMAGE_SIZE

        if model_load_path is None:
            model_load_path = DEFAULT_REPO_ID
        if isinstance(model_load_path, list):
            model_load_path = model_load_path[0]

        from transformers import PaliGemmaProcessor, PaliGemmaConfig
        from hispatial.model import HiSpatialVLM

        if _is_repo_id(model_load_path):
            # --- Load from Hugging Face Hub ---
            from huggingface_hub import snapshot_download

            repo_id = model_load_path
            local_dir = snapshot_download(repo_id)

            config = PaliGemmaConfig.from_pretrained(local_dir)
            self.model = HiSpatialVLM(config).to(dtype=torch.float32)

            weights_path = os.path.join(local_dir, "weights.pt")
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            result = self.model.load_state_dict(checkpoint, strict=False)
            if result.missing_keys:
                print(f"Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"Unexpected keys: {result.unexpected_keys}")

            self.processor = PaliGemmaProcessor.from_pretrained(local_dir)
        else:
            # --- Load from local path ---
            if os.path.isdir(model_load_path):
                weights_path = os.path.join(model_load_path, "weights.pt")
                config_dir = model_load_path
            else:
                weights_path = model_load_path
                config_dir = os.path.dirname(model_load_path)

            config_file = os.path.join(config_dir, "config.json")
            if os.path.exists(config_file):
                # Local directory has config — load directly without backbone
                config = PaliGemmaConfig.from_pretrained(config_dir)
                self.model = HiSpatialVLM(config).to(dtype=torch.float32)

                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                result = self.model.load_state_dict(checkpoint, strict=False)
                if result.missing_keys:
                    print(f"Missing keys: {result.missing_keys}")
                if result.unexpected_keys:
                    print(f"Unexpected keys: {result.unexpected_keys}")

                self.processor = PaliGemmaProcessor.from_pretrained(config_dir)
            else:
                # Legacy: bare weights.pt without config, need backbone
                backbone = kwargs.get("backbone_path", "google/paligemma2-3b-pt-448")

                self.model = HiSpatialVLM.from_pretrained(
                    pretrained_model_name_or_path=backbone,
                    attn_implementation="eager",
                )

                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
                result = self.model.load_state_dict(checkpoint, strict=False)
                if result.missing_keys:
                    print(f"Missing keys: {result.missing_keys}")
                if result.unexpected_keys:
                    print(f"Unexpected keys: {result.unexpected_keys}")

                self.processor = PaliGemmaProcessor.from_pretrained(backbone)

        self.model.eval()
        self.model.to(self.device)

    def query(self, image, prompt, xyz_dict=None, xyz_values=None, max_new_tokens=200, do_sample=False, temperature=0.5, top_p=0.9):
        """Run a spatial VQA query on an image.

        Args:
            image: Input image (file path, PIL Image, or numpy array).
            prompt: Text prompt for the model.
            xyz_dict: Optional dict with ``mask`` and ``points`` tensors from MoGe.
            xyz_values: Optional precomputed [4, H, W] tensor.
            max_new_tokens: Max tokens to generate (default 200).
            do_sample: Enable sampling (default False for greedy).
            temperature: Sampling temperature (default 0.5).
            top_p: Nucleus sampling top-p (default 0.9).

        Returns:
            Decoded text response from the model.
        """
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        I2 = cv2.resize(image, (self.img_size, self.img_size))

        if xyz_values is None and xyz_dict is not None:
            mask = xyz_dict["mask"].cpu().numpy().astype(np.int8)
            xyz = xyz_dict["points"].cpu().numpy()

            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            xyz = cv2.resize(xyz, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask = torch.from_numpy(mask)
            xyz = torch.from_numpy(xyz)

            xyz[xyz > 250] = 250
            xyz[torch.isinf(xyz)] = 250
            xyz[torch.isnan(xyz)] = 250

            xyz_values = xyz.permute(2, 0, 1)
            xyz_values = torch.cat((xyz_values, mask.unsqueeze(0)), dim=0)

        if "<image>" not in prompt:
            prompt = "<image>" + prompt

        inputs = self.processor(text=prompt, images=I2, return_tensors="pt")
        inputs["xyz_values"] = xyz_values.unsqueeze(0)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            inputs = inputs.to(self.device)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            generation = self.model.generate(**inputs, **gen_kwargs)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=False)

        return decoded
