from typing import Union
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig

class BaseConfig:
    def __init__(self, algorithm_config_path: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        path = f"config/{self.algorithm_name}.json" if algorithm_config_path is None else algorithm_config_path
        self.config_dict = load_config_file(path)
        if self.config_dict is None:
            raise FileNotFoundError(f"Cannot load config file: {path}")
        if kwargs:
            self.config_dict.update(kwargs)
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.transformers_config = transformers_config
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        raise NotImplementedError

    @property
    def algorithm_name(self) -> str:
        raise NotImplementedError

class BaseWatermark:
    def __init__(self, algorithm_config: str | BaseConfig, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        pass

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        encoded_unwatermarked_text = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        return self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]

    def detect_watermark(self, prompt: str, text: str, return_dict: bool = True, *args, **kwargs) -> Union[tuple, dict]:
        pass

    def get_data_for_visualize(self, text, *args, **kwargs):
        pass
