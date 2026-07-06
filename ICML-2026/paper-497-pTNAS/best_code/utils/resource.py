import gc
import torch
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from torch_frame.config.text_embedder import TextEmbedderConfig

global_kw_model = None
is_cuda_available = torch.cuda.is_available()


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

        # all available CUDA devices will be used
        self.pool = self.embedding_model.start_multi_process_pool()

    def embed(self, documents, verbose=False):

        # Run encode() on multiple GPUs
        embeddings = self.embedding_model.encode_multi_process(documents,
                                                               self.pool)
        return embeddings


def get_keyword_model():
    """lazy load the keyword model
    """
    global global_kw_model

    if global_kw_model is None:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        custom_embedder = CustomEmbedder(embedding_model=model)
        global_kw_model = KeyBERT(model=custom_embedder)
        # global_kw_model.to('cuda' if is_cuda_available else 'cpu')

    return global_kw_model


def free_keyword_model():
    global global_kw_model
    if global_kw_model is not None:
        del global_kw_model
        global_kw_model = None
        gc.collect()
        torch.cuda.empty_cache()


class GloveTextEmbedding:
    def __init__(self, name: str, device: Optional[torch.device
                                                   ] = None):
        self.model = SentenceTransformer(
            # "all-MiniLM-L12-v2",
            # "sentence-transformers/average_word_embeddings_glove.6B.300d",
            name,
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return torch.from_numpy(self.model.encode(sentences))


def get_text_embedder_cfg(
    model_name: str = "sentence-transformers/average_word_embeddings_glove.6B.300d",
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = 512
) -> TextEmbedderConfig:
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return TextEmbedderConfig(
        GloveTextEmbedding(model_name, device=device), batch_size = batch_size
    )
