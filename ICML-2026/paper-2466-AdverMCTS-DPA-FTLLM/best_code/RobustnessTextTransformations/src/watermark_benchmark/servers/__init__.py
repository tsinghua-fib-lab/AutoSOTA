from .hf import HFServer
from .server import Server
from .vllm import VLLMServer


def get_model(inference_engine, config, **kwargs):
    if inference_engine == "vllm":
        return VLLMServer(config, **kwargs)
    elif inference_engine == "hf":
        return HFServer(config, **kwargs)
    else:
        raise NotImplementedError(f"{inference_engine} is not implemented")


ALL = ["Server", "VLLMServer", "HFServer", "get_model"]
