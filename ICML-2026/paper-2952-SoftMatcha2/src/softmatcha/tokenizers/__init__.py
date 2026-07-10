from softmatcha import registry

from .base import Tokenizer

register, get_tokenizer = registry.setup("tokenizer")

from .fasttext import TokenizerFasttext
from .gensim import TokenizerGensim
from .llama import TokenizerLlama
from .transformers import TokenizerTransformers

__all__ = [
	"Tokenizer",
	"TokenizerTransformers",
	"TokenizerGensim",
	"TokenizerFasttext",
	"TokenizerLlama",
]
