""" Abstract class for adaptor """

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from watermark_benchmark.utils.classes import Generation, WatermarkSpec


class Server(ABC):
    """
    Abstract class representing a server that can install a watermark into a model, run the server, and return a tokenizer.
    """

    @abstractmethod
    def install(self, watermark_engine) -> None:
        """Install watermark into model"""
        self.device = "cpu"

    @abstractmethod
    def run(
        self,
        inputs: List[str],
        config: Dict[str, Any],
        temp: float,
        keys: Optional[int] = None,
        ws: Optional[WatermarkSpec] = None,
        **kwargs
    ) -> List[Generation]:
        """Run server"""
        return []

    @abstractmethod
    def tokenizer(self):
        """Return tokenizer"""
        pass

    def devices(self):
        return self.devices

    @staticmethod
    def get_tokenizer(model):
        return AutoTokenizer.from_pretrained(model)
