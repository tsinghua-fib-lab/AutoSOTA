import base64
import json
from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO

from openai.types.chat import ChatCompletionMessageParam
import pandas as pd
from substantive.faircp.fairness.llm.llm_message_builder import (
    LlmMessageBuilder,
    VisionLlmMessageBuilder,
)


def get_llm_message_builder(cfg: dict, label_map: dict[int, str]) -> LlmMessageBuilder:
    match cfg["dataset"]:
        case "bios":
            return BiosLlmMessageBuilder(cfg, label_map)
        case "facet":
            return FacetLlmMessageBuilder(cfg, label_map)
        #case "fashion-mnist":
            #return FashionMnistLlmMessageBuilder(cfg, label_map)
        case "ravdess":
            return RavdessLlmMessageBuilder(cfg, label_map)
        case "acs-income":
            return ACSIncomeLlmMessageBuilder(cfg, label_map)
        case _:
            raise ValueError(f"Unknown dataset: {cfg['dataset']}")


class BiosLlmMessageBuilder(LlmMessageBuilder):
    def __init__(self, cfg: dict, label_map: dict[int, str]):
        super().__init__(cfg, label_map)

    def construct_messages(
        self, prompt: str, predictions: List[int], coverage: str
    ) -> List[ChatCompletionMessageParam]:
        user_prompt = self.format_text_question(prompt, predictions, coverage)
        return [
            {"role": "system", "content": self.system_instruct},
            {"role": "user", "content": user_prompt},
        ]

    def construct_control_messages(
        self, prompt: str
    ) -> List[ChatCompletionMessageParam]:
        user_prompt = self.format_control_text_question(prompt)
        return [
            {"role": "system", "content": self.system_instruct},
            {"role": "user", "content": user_prompt},
        ]

    def format_text_question(
        self, prompt: str, pred_ids: list[int], coverage: str
    ) -> str:
        options = ",".join(f"{self.label_map.get(i, f'Option {i}')}" for i in pred_ids)
        if self.user_prompt_cvg is None:
            coverage_info = ""
        else:
            coverage_info = (
                self.user_prompt_cvg.format(coverage=coverage)
                if coverage != "-1"
                else self.user_prompt_no_cvg
            )
        return self.user_prompt_template.format(
            prompt=prompt.strip(), options=options.strip(), coverage_info=coverage_info
        )

    def format_control_text_question(
        self,
        prompt: str,
    ) -> str:
        return self.user_prompt_control_template.format(prompt=prompt.strip())


class FacetLlmMessageBuilder(VisionLlmMessageBuilder):
    def _encode_image(self, prompt: str) -> str:
        """Open .jpg image, return base64 data URI."""
        with Image.open(prompt).convert("RGB") as img:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{img_b64}"


# class FashionMnistLlmMessageBuilder(VisionLlmMessageBuilder):
#     def __init__(self, cfg: dict, label_map: dict[int, str]):
#         super().__init__(cfg, label_map)
#         self.dataset = torchvision.datasets.FashionMNIST(
#             root=cfg["data_root"],
#             train=False,
#             download=True,
#             transform=torchvision.transforms.ToTensor(),
#         )

    # def _encode_image(self, prompt: str) -> str:
    #     """Take FashionMNIST tensor, convert to grayscale PNG base64."""
    #     idx = int(prompt)
    #     image_tensor, _ = self.dataset[idx]  # (1,H,W) tensor
    #     img = Image.fromarray((image_tensor.squeeze().numpy() * 255).astype("uint8"))
    #     buf = BytesIO()
    #     img.save(buf, format="PNG")  # use PNG for lossless grayscale
    #     img_bytes = buf.getvalue()
    #     img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    #     return f"data:image/png;base64,{img_b64}"


class RavdessLlmMessageBuilder(LlmMessageBuilder):
    def __init__(self, cfg: dict, label_map: dict[int, str]):
        super().__init__(cfg, label_map)
        self.base64 = cfg["audio_lm_base64"]

    def construct_messages(
        self, prompt: str, predictions: List[int], coverage: str
    ) -> List[ChatCompletionMessageParam]:
        user_prompt = self.format_question(predictions, coverage)
        return [
            {"role": "system", "content": self.system_instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            # "data": self._encode_audio_base64(prompt)
                            # if self.base64
                            # else None,
                            # "format": "wav",
                            # "path": str(Path(prompt).resolve()),
                            "data": self._encode_audio_base64(prompt),
                            "format": "wav",
                        },
                    },
                ],
            },
        ]

    def construct_control_messages(
        self, prompt: str
    ) -> List[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": self.system_instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt_control_template},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            # "data": self._encode_audio_base64(prompt)
                            # if self.base64
                            # else None,
                            # "format": "wav",
                            # "path": str(Path(prompt).resolve()),
                            "data": self._encode_audio_base64(prompt),
                            "format": "wav",
                        },
                    },
                ],
            },
        ]

    def _encode_audio_base64(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return b64


class ACSIncomeLlmMessageBuilder(LlmMessageBuilder):
    def __init__(self, cfg: dict, label_map: dict[int, str]):
        super().__init__(cfg, label_map)
        self.data_root = Path(cfg["data_root"])
        self.features_df = pd.read_csv(self.data_root / "income_raw_features.csv")
        # Prepare all_options string for system instruction
        self.all_options = ", ".join(
            f"{self.label_map.get(i, f'Option {i}')}"
            for i in sorted(self.label_map.keys())
        )

    def construct_messages(
        self, prompt: str, predictions: List[int], coverage: str
    ) -> List[ChatCompletionMessageParam]:
        idx = self._parse_index(prompt)
        row = self.features_df.iloc[idx].to_dict()

        # Format system_instruct with all_options
        system_instruct = self.system_instruct.format(all_options=self.all_options)

        user_prompt = self.format_text_question(row, predictions, coverage)
        return [
            {"role": "system", "content": system_instruct},
            {"role": "user", "content": user_prompt},
        ]

    def construct_control_messages(
        self, prompt: str
    ) -> List[ChatCompletionMessageParam]:
        idx = self._parse_index(prompt)
        row = self.features_df.iloc[idx].to_dict()

        # Format system_instruct with all_options
        system_instruct = self.system_instruct.format(all_options=self.all_options)

        user_prompt = self.format_control_text_question(row)
        return [
            {"role": "system", "content": system_instruct},
            {"role": "user", "content": user_prompt},
        ]

    def format_text_question(
        self, feature_dict: dict, pred_ids: list[int], coverage: str
    ) -> str:
        options = ",".join(f"{self.label_map.get(i, f'Option {i}')}" for i in pred_ids)
        if self.user_prompt_cvg is None:
            coverage_info = ""
        else:
            coverage_info = (
                self.user_prompt_cvg.format(coverage=coverage)
                if coverage != "-1"
                else self.user_prompt_no_cvg
            )
        
        return self.user_prompt_template.format(
            **feature_dict, options=options.strip(), coverage_info=coverage_info
        )

    def format_control_text_question(
        self,
        feature_dict: dict,
    ) -> str:
        return self.user_prompt_control_template.format(**feature_dict)

    def _parse_index(self, prompt: str) -> int:
        # prompt is expected to be like "tensor(123)" or just "123"
        try:
            if "tensor(" in prompt:
                return int(prompt.split("(")[1].split(")")[0])
            return int(prompt)
        except (ValueError, IndexError):
            # Fallback if parsing fails, though expected format is tensor(idx)
            # You might want to log this or raise an error
            return 0
