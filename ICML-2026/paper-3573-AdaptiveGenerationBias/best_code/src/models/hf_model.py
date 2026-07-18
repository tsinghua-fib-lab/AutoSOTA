from typing import List, Dict, Iterator, Tuple, Optional
from tqdm import tqdm
import torch
from src.configs import ModelConfig
from src.bias_pipeline.data_types.conversation import Conversation
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import BaseModel


class HFModel(BaseModel):
    curr_models: Dict[str, AutoModelForCausalLM] = {}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if (
            config.name not in self.curr_models
        ):  #  This is a hack to avoid loading the same model multiple times
            self.curr_models[config.name] = AutoModelForCausalLM.from_pretrained(
                config.name,
                torch_dtype=torch.float16 if config.dtype == "float16" else torch.float32,
                device_map=config.device,
            )
        self.model: AutoModelForCausalLM = self.curr_models[config.name]

        self.device = self.model.device
        if config.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def predict(self, input: Conversation, **kwargs):
        messages, input_str, input_ids = input.to_chat(tokenizer=self.tokenizer, tokenize=True)
        input_ids = input_ids.to(self.device)

        input_length = len(input_ids[0])

        output = self.model.generate(input_ids, **self.config.args)

        # For decoder only models:
        out_ids = output[:, input_length:]

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def predict_multi(
        self, inputs: List[Conversation], **kwargs
    ) -> Iterator[Tuple[Conversation, str]]:
        new_inputs = []
        for input in inputs:
            new_inputs.append(input.to_chat(tokenizer=self.tokenizer, tokenize=True))

        if "max_workers" in kwargs:
            batch_size = kwargs["max_workers"]
        else:
            batch_size = 1

        for i in tqdm(range(0, len(new_inputs), batch_size)):
            end = min(i + batch_size, len(new_inputs))
            new_inputs_batch = new_inputs[i:end]

            # Re-tokenize the batch to ensure that the batch is the same size
            new_inputs_batch_text = [x[1] for x in new_inputs_batch]

            model_inputs = self.tokenizer(
                new_inputs_batch_text,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(self.device)
            input_length = len(model_inputs["input_ids"][0])

            output = self.model.generate(**model_inputs, **self.config.args)

            outs_str = self.tokenizer.batch_decode(
                output[:, input_length:], skip_special_tokens=True
            )
            for j in range(len(outs_str)):
                yield (inputs[i + j], outs_str[j])

    def predict_string(self, input: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        input = [
            {
                "role": "system",
                "content": "You are an helpful assistant.",
            },
            {"role": "user", "content": input},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            input,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.device)

        input_length = len(input_ids[0])

        output = self.model.generate(input_ids, **self.config.args)

        # For decoder only models:
        out_ids = output[:, input_length:]

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
