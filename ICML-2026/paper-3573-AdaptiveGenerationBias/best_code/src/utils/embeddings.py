#!/usr/bin/env python3
"""
Embedding Utilities

This module provides functionality for generating text embeddings using either SentenceTransformers
or OpenAI's embedding API, computing similarity matrices, and creating visualizations. It includes
caching mechanisms for efficient reuse of embeddings and supports both embedding-based and fuzzy text similarity.

Usage:
    from src.utils.embeddings import EmbeddingManager

    # Using local SentenceTransformers (default)
    manager = EmbeddingManager()

    # Using OpenAI embedding API
    manager = EmbeddingManager(use_openai=True, openai_model="text-embedding-3-small")

    embeddings = manager.get_embeddings(["text1", "text2", "text3"])
    similarity_matrix = manager.compute_similarity_matrix(texts, method="embedding")

Note:
    For OpenAI embeddings, you need to set the OPENAI_API_KEY environment variable
    and install the openai package: pip install openai
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from rapidfuzz import fuzz, process, utils
import warnings

warnings.filterwarnings("ignore")

# Try to import sentence transformers, fallback if not available
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import OpenAI, fallback if not available
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EmbeddingManager:
    """
    Manages text embeddings with caching and similarity computation.

    This class provides a unified interface for:
    - Generating embeddings using SentenceTransformers
    - Caching embeddings for efficiency
    - Computing similarity matrices (embedding-based or fuzzy)
    - Creating t-SNE visualizations
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the SentenceTransformer model to use (when use_openai=False)
            use_openai: Whether to use OpenAI embedding API instead of local model
            openai_model: OpenAI embedding model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        self._embeddings_cache = {}
        self._sentence_model = None
        self._openai_client = None

        if use_openai:
            # Initialize OpenAI client
            if OPENAI_AVAILABLE:
                try:
                    self._openai_client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
                    print(f"Initialized OpenAI embeddings with model: {openai_model}")
                except Exception as e:
                    print(f"Failed to initialize OpenAI client: {e}")
                    self._openai_client = None
            else:
                print("Warning: openai package not available. Install with: pip install openai")
        else:
            # Initialize sentence transformer model if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self._sentence_model = SentenceTransformer(model_name)
                    print(f"Loaded sentence transformer model: {model_name}")
                except Exception as e:
                    print(f"Failed to load sentence transformer model: {e}")
                    self._sentence_model = None
            else:
                print(
                    "Warning: sentence-transformers not available. Embedding similarity will be disabled."
                )

    @property
    def is_available(self) -> bool:
        """Check if embeddings are available (either OpenAI or sentence transformers)."""
        if self.use_openai:
            return self._openai_client is not None
        else:
            return self._sentence_model is not None

    def clear_cache(self):
        """Clear the embeddings cache."""
        self._embeddings_cache.clear()

    def _get_openai_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings from OpenAI API for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings, or None if API call fails
        """
        if not self._openai_client:
            self._openai_client = openai.OpenAI()

        try:
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-large", input=texts
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")
            return None

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings for a list of texts, with caching.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings, or None if embedding service unavailable
        """
        if not self.is_available:
            return None

        embeddings = []
        to_embed = []

        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            if text in self._embeddings_cache:
                embeddings.append((i, self._embeddings_cache[text]))
            else:
                to_embed.append((i, text))

        # Generate new embeddings for uncached texts
        if to_embed:
            texts_to_embed = [t[1] for t in to_embed]

            if False:
                new_embeds = self._get_openai_embeddings(texts_to_embed)
            else:
                new_embeds = self._sentence_model.encode(texts_to_embed)

            if new_embeds is not None:
                for (i, text), embed in zip(to_embed, new_embeds):
                    self._embeddings_cache[text] = embed
                    embeddings.append((i, embed))

        # Sort embeddings back to original order
        embeddings.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embeddings])

    def compute_similarity_matrix(
        self, texts: List[str], method: str = "embedding"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute similarity matrix for a list of texts.

        Args:
            texts: List of text strings
            method: Similarity method - "embedding" or "fuzzy"

        Returns:
            Tuple of (similarity_matrix, embeddings). Embeddings is None for fuzzy method.
        """
        if method == "embedding":
            embeddings = self.get_embeddings(texts)
            if embeddings is None:
                provider = "OpenAI API" if self.use_openai else "sentence-transformers"
                raise ValueError(f"Embedding method requires {provider} to be available")
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix, embeddings

        elif method == "fuzzy":
            similarity_matrix = process.cdist(
                texts,
                texts,
                scorer=fuzz.WRatio,
                processor=utils.default_process,
                workers=-1,  # all CPU cores
                dtype=np.uint8,
            )
            return similarity_matrix, None

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def find_most_similar(
        self, texts: List[str], method: str = "embedding", top_k: int = 3
    ) -> List[List[Tuple[int, float]]]:
        """
        Find the most similar texts for each text in the input list.

        Args:
            texts: List of text strings
            method: Similarity method - "embedding" or "fuzzy"
            top_k: Number of top similar texts to return for each input text

        Returns:
            List where each element contains tuples of (index, similarity_score)
            for the top_k most similar texts to the text at that position
        """
        similarity_matrix, _ = self.compute_similarity_matrix(texts, method)

        # For each row, select top k values by similarity (excluding self)
        top_sim_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k - 1 : -1]

        results = []
        for i, indices in enumerate(top_sim_indices):
            # Get similarity scores and filter out self-similarity
            similarities = [(idx, similarity_matrix[i][idx]) for idx in indices if idx != i]
            # Sort by similarity score descending
            similarities.sort(key=lambda x: x[1], reverse=True)
            results.append(similarities[:top_k])

        return results

    def create_tsne_visualization(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        method: str = "embedding",
        **tsne_kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Create 2D t-SNE visualization of text similarities.

        Args:
            texts: List of text strings
            metadata: Optional list of metadata dicts for each text
            method: Similarity method - "embedding" or "fuzzy"
            **tsne_kwargs: Additional arguments for TSNE

        Returns:
            DataFrame with x, y coordinates and metadata, or None if error
        """
        try:
            if len(texts) < 3:  # t-SNE needs at least 3 points
                return None

            similarity_matrix, embeddings = self.compute_similarity_matrix(texts, method)

            # Set up t-SNE parameters
            perplexity = min(30, len(texts) - 1)
            tsne_params = {
                "n_components": 2,
                "random_state": 42,
                "perplexity": perplexity,
                "init": "random" if method == "fuzzy" else "pca",
                "metric": "precomputed" if method == "fuzzy" else "euclidean",
            }
            tsne_params.update(tsne_kwargs)

            tsne = TSNE(**tsne_params)

            # Apply t-SNE
            if method == "fuzzy":
                tsne_results = tsne.fit_transform(similarity_matrix)
            else:
                tsne_results = tsne.fit_transform(embeddings)

            # Create result DataFrame
            result_data = {
                "text": texts,
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
            }

            # Add metadata if provided
            if metadata:
                for i, meta in enumerate(metadata):
                    if meta:
                        for key, value in meta.items():
                            if key not in result_data:
                                result_data[key] = [None] * len(texts)
                            result_data[key][i] = value

            return pd.DataFrame(result_data)

        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")
            return None


class QuestionSimilarityAnalyzer:
    """
    Specialized analyzer for computing question similarities with additional context.

    This class extends the basic embedding functionality to work specifically with
    question data that includes metadata like question IDs, bias scores, etc.
    """

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize the question similarity analyzer.

        Args:
            embedding_manager: Optional pre-configured EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()

    def compute_question_similarities(
        self, questions_df: pd.DataFrame, method: str = "embedding"
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """
        Compute similarity matrix for questions from a DataFrame.

        Args:
            questions_df: DataFrame with columns 'question_id' and 'question_text'
            method: Similarity method - "embedding" or "fuzzy"

        Returns:
            Tuple of (similarity_df, similarity_matrix, embeddings)
        """
        if questions_df.empty:
            return pd.DataFrame(), np.ndarray(0), None

        # Get unique questions
        unique_questions = questions_df.drop_duplicates(subset=["question_id"])
        questions_list = unique_questions[["question_id", "question_text"]].to_dict("records")

        if len(questions_list) < 2:
            return pd.DataFrame(), np.ndarray(0), None

        print(f"Computing {method} similarities for {len(questions_list)} questions...")

        # Compute similarity matrix
        texts = [q["question_text"] for q in questions_list]
        similarity_matrix, embeddings = self.embedding_manager.compute_similarity_matrix(
            texts, method
        )

        # Create similarity DataFrame with detailed pairs
        similarity_data = []
        top_k = 3

        # For each row select top k values by similarity
        top_sim_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k - 1 : -1]

        for i1, q1 in enumerate(questions_list):
            for sim_idx in top_sim_indices[i1]:
                q2 = questions_list[sim_idx]

                if q1["question_id"] == q2["question_id"]:
                    continue

                sim_score = similarity_matrix[i1][sim_idx]
                similarity_data.append(
                    {
                        "question_id_1": q1["question_id"],
                        "question_id_2": q2["question_id"],
                        "question_text_1": q1["question_text"],
                        "question_text_2": q2["question_text"],
                        "similarity_score": sim_score,
                    }
                )

        similarity_df = pd.DataFrame(similarity_data)
        return similarity_df, similarity_matrix, embeddings

    def get_similar_questions(
        self,
        question_id: str,
        similarity_df: pd.DataFrame,
        conversations_df: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get most similar questions to a given question with bias score context.

        Args:
            question_id: Target question ID
            similarity_df: DataFrame from compute_question_similarities
            conversations_df: DataFrame with bias scores for questions
            top_k: Number of similar questions to return

        Returns:
            List of similar question data with bias and fitness scores
        """
        if similarity_df.empty:
            return []

        # Get similarities for the specified question
        similar_questions = similarity_df[similarity_df["question_id_1"] == question_id].copy()

        if similar_questions.empty:
            return []

        # Enhance with bias score information
        normalized_similarities = []
        for _, row in similar_questions.iterrows():
            # Get conversations with this question to fetch bias scores
            convs = conversations_df[conversations_df["question_id"] == row["question_id_2"]]

            result = {
                "question_id": row["question_id_2"],
                "question_text": row["question_text_2"],
                "similarity_score": row["similarity_score"],
            }

            if not convs.empty:
                result["question_bias"] = (convs["bias_score"].mean(), convs["bias_score"].std())
                if "fitness_score" in convs.columns:
                    result["question_fitness"] = (
                        convs["fitness_score"].mean(),
                        convs["fitness_score"].std(),
                    )
            else:
                result["question_bias"] = None
                result["question_fitness"] = None

            normalized_similarities.append(result)

        # Sort by similarity score and return top k
        normalized_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

        return normalized_similarities[:top_k]
