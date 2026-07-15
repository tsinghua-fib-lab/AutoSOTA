import abc
from typing import Union, List
from omegaconf import DictConfig
from collections import OrderedDict


class BaseCompressor:
    def __init__(self, compressor_config: DictConfig):
        pass

    @abc.abstractmethod
    def compress_model(
        self,
        model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]],
        batched: bool = False,
    ) -> bytes:
        """
        Compress all the parameters of local model(s) for efficient communication. The local model can be batched as a list.
        :param model: local model parameters (can be nested)
        :param batched: whether the input is a batch of models
        :return: compressed model parameters as bytes
        """
        pass

    def decompress_model(
        self,
        compressed_model: bytes,
        model: Union[dict, OrderedDict],
        batched: bool = False,
    ) -> Union[OrderedDict, dict, List[Union[OrderedDict, dict]]]:
        """
        Decompress all the communicated model parameters. The local model can be batched as a list.
        :param compressed_model: compressed model parameters as bytes
        :param model: a model sample for de-compression reference
        :param batched: whether the input is a batch of models
        :return decompressed_model: decompressed model parameters
        """
        pass
