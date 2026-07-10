from softmatcha import registry

from .base import Embedding

register, get_embedding = registry.setup("embedding")

from .fasttext import EmbeddingFasttext
from .gensim import EmbeddingGensim
from .transformers import EmbeddingTransformers
from .llama import EmbeddingLlama

__all__ = ["Embedding", "EmbeddingFasttext", "EmbeddingGensim", "EmbeddingTransformers", "EmbeddingLlama"]

GENSIM_PRETRAINED_MODELS = [
	"fasttext-wiki-news-subwords-300",
	"conceptnet-numberbatch-17-06-300",
	"word2vec-ruscorpora-300",
	"word2vec-google-news-300",
	"glove-wiki-gigaword-50",
	"glove-wiki-gigaword-100",
	"glove-wiki-gigaword-200",
	"glove-wiki-gigaword-300",
	"glove-twitter-25",
	"glove-twitter-50",
	"glove-twitter-100",
	"glove-twitter-200",
	# "__testing_word2vec-matrix-synopsis",
]
