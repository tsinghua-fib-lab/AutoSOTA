import pytest

from softmatcha.embeddings import EmbeddingGensim, EmbeddingTransformers
from softmatcha.tokenizers import TokenizerGensim, TokenizerTransformers


@pytest.fixture(scope="session")
def embed_bert():
	return EmbeddingTransformers.build(
		EmbeddingTransformers.Config("bert-base-uncased")
	)


@pytest.fixture(scope="session")
def embed_glove():
	return EmbeddingGensim.build(EmbeddingGensim.Config("glove-wiki-gigaword-300"))


@pytest.fixture(scope="session")
def tokenizer_bert():
	return TokenizerTransformers.build(
		TokenizerTransformers.Config("bert-base-uncased")
	)


@pytest.fixture(scope="session")
def tokenizer_glove():
	return TokenizerGensim.build(TokenizerGensim.Config("glove-wiki-gigaword-300"))
