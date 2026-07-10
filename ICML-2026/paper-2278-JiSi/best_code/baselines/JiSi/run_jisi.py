"""
JiSi route-and-aggregate runner.

The runner consumes pre-split JiSi JSONL files, builds an embedding bank from
the support set, performs query-response mixed routing, and optionally runs the
adaptive aggregation stage.
"""
import json

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import copy
import shutil
import random

from tqdm import tqdm
import tiktoken
import yaml
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import asyncio
import jsonlines

from generators.factory import create_generator

# Import local modules
from .config import JiSiConfig, setup_logging
from .utils.utils import (
    generate_general_with_cache,
    inject_references_to_messages,
    setup_model_config,
)
from common.cache.decorator import create_api_cache_decorator


class JiSi:
    """
    JiSi route-and-aggregate engine.

    JiSi uses support-set retrieval to estimate instance-level model capability,
    then either routes to a selected expert or aggregates selected expert
    responses.

    Attributes:
        config (JiSiConfig): Configuration parameters
        embedder: Embedding generation service
        tokenizer: Tiktoken tokenizer for precise token counting
        available_models (List[str]): Available model names from data
    """

    def __init__(self, config: JiSiConfig):
        """
        Initialize the JiSi runner.

        Args:
            config: Configuration object with all parameters

        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If embedding service is unreachable
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        if self.config.api_config_path:
            setup_model_config(self.config.api_config_path)
            self.logger.info(f"Loaded API model config: {self.config.api_config_path}")

        # Initialize embedding service
        try:
            self.embedder = self._create_embedding_generator()
            resolved_model_name = getattr(self.embedder, "model_name", config.embedding_model)
            self.logger.info(f"Initialized embedding generator: {resolved_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding generator: {e}")
            raise ConnectionError(f"Cannot connect to embedding service: {e}")

        # Initialize tiktoken encoder for precise token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.embedding_model)
            self.logger.info(f"Using tiktoken encoder for {config.embedding_model}")
        except Exception:
            print("Failed to initialize tiktoken encoder")
            embedding_tokenizer_path = getattr(self.embedder, "model_path", None)
            if embedding_tokenizer_path:
                print(f"Using given embedding model tokenizer {embedding_tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_path)
            else:
                # Fallback to cl100k_base encoding (used by most OpenAI models)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.logger.warning(f"Fallback to cl100k_base encoding for tokenization")
        try:
            self.ds_tokenizer = AutoTokenizer.from_pretrained(config.deepseek_tokenizer_path)
            self.logger.info(f"Using length tokenizer: {config.deepseek_tokenizer_path}")
        except Exception as exc:
            self.ds_tokenizer = self.tokenizer
            self.logger.warning(
                "Failed to load deepseek_tokenizer_path=%s (%s). Falling back to the embedding tokenizer.",
                config.deepseek_tokenizer_path,
                exc,
            )
        # Initialize model components (will be set during training)
        self.available_models: List[str] = []
        self.embedding_bank = None
        self.train_data = None
        self.test_data = None
        self.question_bank = None
        self.response_models: List[str] = []
        self.model_id_map: Dict[str, int] = {}
        self.cache_decorator = None
        if self.config.cache_config is not None:
            with open(self.config.cache_config, "r") as f:
                cache_cfg = yaml.safe_load(f)
            self.cache_decorator = create_api_cache_decorator(cache_cfg)
        self.build_api()


    def build_api(self):
        def _generate_with_references_with_cache(
                model,
                messages,
                references=[],
                max_tokens=2048,
                temperature=0.7,
                top_p=1.0,
                logprobs=None,
                agg_prompt='normal',
                ref_score=None
        ):
            """
            Generate with references using generate_general_with_cache (cached version).
            Returns GeneratorOutput format.
            """
            if len(references) > 0:
                messages = inject_references_to_messages(messages, references, agg_prompt=agg_prompt,
                                                         ref_score=ref_score)
            generate_fn = generate_general_with_cache
            if self.cache_decorator is not None:
                generate_fn = self.cache_decorator(generate_general_with_cache)
            return generate_fn(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs
            )


        def _generate_with_references_api(
                model: str,
                messages,
                temperature: float = 0.7,
                top_p: float = 1.0,
                max_tokens: int = 2048,
                references=None,
                logprobs=None,
                agg_prompt='normal',
                ref_score=None
        ):
            """
            Generate an aggregator response from prepared JiSi references.
            Returns GeneratorOutput format that can be cached.
            """
            if references is None:
                references = []

            result = _generate_with_references_with_cache(
                model=model,
                messages=messages,
                references=references,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                agg_prompt=agg_prompt,
                ref_score=ref_score
            )
            return result


        async def _async_generate_with_references(
                model: str,
                messages,
                temperature: float = 0.7,
                top_p: float = 1.0,
                max_tokens: int = 2048,
                references=None,
                logprobs=None,
                agg_prompt='normal',
                ref_score=None
        ):
            loop = asyncio.get_running_loop()
            executor = getattr(self, '_thread_executor', None)
            return await loop.run_in_executor(
                executor,
                _generate_with_references_api,
                model,
                messages,
                temperature,
                top_p,
                max_tokens,
                references,
                logprobs,
                agg_prompt,
                ref_score,
            )
        self.async_generate_with_references = _async_generate_with_references

    def _validate_data_item(self, item: Dict, line_num: int) -> bool:
        """
        Validate a single data item for required fields and format.

        Args:
            item: Data item to validate
            line_num: Line number for error reporting

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["query", "records"]

        for field in required_fields:
            if field not in item:
                self.logger.error(f"Line {line_num}: Missing required field '{field}'")
                return False

        if not isinstance(item["query"], str) or not item["query"].strip():
            self.logger.error(f"Line {line_num}: Query must be a non-empty string")
            return False

        if not isinstance(item["records"], dict) or not item["records"]:
            self.logger.error(f"Line {line_num}: Records must be a non-empty dict")
            return False

        # Validate and clean records format (model_name -> float or boolean)
        for model_name, result in item["records"].items():
            if result is None:
                # Convert None to 0.0 for missing/failed results
                item["records"][model_name] = 0.0
            elif isinstance(result, bool):
                # Convert boolean to float (True -> 1.0, False -> 0.0)
                item["records"][model_name] = 1.0 if result else 0.0
            elif isinstance(result, (int, float)):
                # Accept numeric values, convert to float
                item["records"][model_name] = float(result)
            else:
                self.logger.error(f"Line {line_num}: Record for '{model_name}' must be numeric, boolean, or null, got {type(result)}")
                return False

        return True

    def load_and_split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load data from pre-split train and test files.

        Returns:
            Tuple of (train_data, test_data)
        """
        train_file = Path(self.config.train_data_path)
        test_file = Path(self.config.test_data_path)

        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        print(f"Loading pre-split data:")
        print(f"  Train: {train_file}")
        print(f"  Test: {test_file}")

        # Load train data
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if self._validate_data_item(item, line_num):
                        train_data.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON at train line {line_num}: {e}")

        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if self._validate_data_item(item, line_num):
                        test_data.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON at test line {line_num}: {e}")

        print(f"\nLoaded {len(train_data)} train items and {len(test_data)} test items")

        if self.config.excluded_datasets and self.config.dataset_exclusion_mode == "hard":
            original_train_count = len(train_data)
            original_test_count = len(test_data)
            train_data = [
                item for item in train_data
                if item.get("dataset", "default") not in self.config.excluded_datasets
            ]
            test_data = [
                item for item in test_data
                if item.get("dataset", "default") not in self.config.excluded_datasets
            ]
            print(
                "Hard-excluded datasets: "
                f"{original_train_count - len(train_data)} train items and "
                f"{original_test_count - len(test_data)} test items removed"
            )

        return train_data, test_data

    def _truncate_text(self, text: str) -> str:
        """Trim embedding input to the configured token budget."""
        max_tokens = self.config.max_tokens
        if max_tokens <= 0:
            return text

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        tail_tokens = tokens[-max_tokens:]
        try:
            return self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(tail_tokens)

    def _get_embedding_batch(self, queries_batch: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of queries."""
        embeddings = []
        for query in queries_batch:
            embedding_output = self.embedder.generate_embedding(self._truncate_text(query))
            embedding = np.array(embedding_output.embeddings, dtype=float)
            if embedding.size == 0:
                raise RuntimeError("Received empty embedding from generator")
            embeddings.append(embedding)
        return embeddings

    # ------------------------------------------------------------------
    # Embedding generator helper
    # ------------------------------------------------------------------
    def _create_embedding_generator(self):
        """Instantiate shared EmbeddingGenerator based on router configuration."""

        cache_config = None
        model_config: Dict[str, Any] = {
            "generator_type": "embedding",
            "api_model_name": self.config.embedding_model,
            "name": self.config.embedding_model,
            "base_url": self.config.embedding_base_url,
            "api_key": self.config.embedding_api_key,
            "timeout": 600,
        }

        if self.config.embedding_config_path:
            config_path = Path(self.config.embedding_config_path)
            with open(config_path, "r", encoding="utf-8") as fp:
                shared_config = yaml.safe_load(fp)

            # Merge model settings from shared config
            file_model_cfg = shared_config.get("embedding_model", {}) or {}
            model_config.update({k: v for k, v in file_model_cfg.items() if v is not None})

            cache_config = shared_config.get("cache")

        # Resolve API key placeholders (environment variable names)
        api_key = model_config.get("api_key")
        if isinstance(api_key, str) and api_key.isupper() and "_" in api_key:
            model_config["api_key"] = os.getenv(api_key, api_key)

        # Ensure required fields are present
        required_fields = ["api_model_name", "base_url", "api_key"]
        missing = [field for field in required_fields if not model_config.get(field)]
        if missing:
            raise ValueError(f"Embedding configuration missing required fields: {', '.join(missing)}")

        return create_generator(model_config, cache_config)

    def _generate_embeddings_concurrent(self, queries: List[str]) -> List[np.ndarray]:
        """Generate embeddings for queries using concurrent processing."""
        batch_size = max(1, len(queries) // self.config.max_workers)
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks and create a mapping from future to batch index
            future_to_batch = {executor.submit(self._get_embedding_batch, batch): i
                              for i, batch in enumerate(query_batches)}

            all_embeddings = [None] * len(query_batches)
            with tqdm(total=len(queries), desc="Generating embeddings", miniters=1, mininterval=0.1) as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    batch_embeddings = future.result()
                    all_embeddings[batch_idx] = batch_embeddings
                    pbar.update(len(batch_embeddings))

        # Flatten the results maintaining original order
        return [emb for batch in all_embeddings for emb in batch]

    def build_embedding_bank(self, train_data: List[Dict]):
        """Build the embedding bank from training data."""
        queries = [item["query"] for item in train_data]
        train_query_embed_cache = Path(self.config.baseline_scores_path).parent.parent / "train_query_embed.tar"
        if not os.path.exists(train_query_embed_cache):
            train_embeddings = self._generate_embeddings_concurrent(queries)
            train_embeddings = F.normalize(torch.tensor(train_embeddings).to(self.device), dim=-1)
            torch.save(train_embeddings, train_query_embed_cache)
        else:
            train_embeddings = torch.load(train_query_embed_cache, map_location=self.device).to(self.device)
        # Get available model
        model_set = set()
        train_records = [item["records"] for item in train_data]
        for rec in train_records:
            if isinstance(rec, dict):
                model_set.update([m for m in rec.keys() if m is not None])

        # Apply excluded_models filter
        if self.config.excluded_models:
            model_set = {m for m in model_set if m not in self.config.excluded_models}
        self.available_models = sorted(model_set)

        # build question bank
        self.question_bank = {'model_pred_dict': {m:[] for m in self.available_models}, 'query_array': []}
        for item in train_data:
            self.question_bank['query_array'].append(item['query'])
            for m in self.available_models:
                self.question_bank['model_pred_dict'][m].append(item['records'].get(m, 0.0))
        for m in self.available_models:
            self.question_bank['model_pred_dict'][m] = torch.tensor(self.question_bank['model_pred_dict'][m])
        self.question_bank['query_array'] = np.array(self.question_bank['query_array'])
        # Idea 3: Pre-compute per-dataset per-model accuracy priors
        self.dataset_prior = {}
        dataset_model_correct = {}
        dataset_model_total = {}
        for item in train_data:
            ds = item.get('dataset', 'default')
            if ds not in dataset_model_correct:
                dataset_model_correct[ds] = {}
                dataset_model_total[ds] = {}
            for m in self.available_models:
                dataset_model_correct[ds].setdefault(m, 0.0)
                dataset_model_total[ds].setdefault(m, 0)
                val = item['records'].get(m, None)
                if val is not None:
                    dataset_model_correct[ds][m] += float(val)
                    dataset_model_total[ds][m] += 1
        for ds in dataset_model_correct:
            self.dataset_prior[ds] = {}
            for m in self.available_models:
                total = dataset_model_total[ds][m]
                self.dataset_prior[ds][m] = dataset_model_correct[ds][m] / max(total, 1)
        return train_embeddings

    def route_queries_batch(self, queries: List[str], dataset_labels: List[str] = None):
        """Route multiple queries in batch for better efficiency."""
        weighted_score = self.config.weighted_score
        k = max(1, min(self.config.rag_num, len(self.train_data or [])))
        query_embeddings = self._generate_embeddings_concurrent(queries)
        query_embeddings = torch.tensor(query_embeddings).to(self.device)
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        scores = (query_embeddings @ self.embedding_bank.T) * 100 # [q_batch, emb_num]
        scores_topk_value, scores_topk = scores.topk(scores.size(1), dim=-1)
        results = {
            'rag_model_list': [],
            'rag_score_list': [],
            'score_topk_list': [],
            'score_topk_value_list': [],
        }

        for i in range(len(queries)):
            threshold_idx = min(k - 1, scores_topk_value.size(1) - 1)
            threshold_bound = scores_topk_value[i][threshold_idx] * self.config.rag_thres
            scores_topk_i = scores_topk[i][scores_topk_value[i] > threshold_bound].cpu()
            if scores_topk_i.numel() == 0:
                scores_topk_i = scores_topk[i][:1].cpu()
            # Idea 2: Discriminative Support Set Filtering
            # Filter support examples where ALL models agree (correct or incorrect)
            # as they provide zero discriminative signal for model selection
            if scores_topk_i.numel() > 0:
                all_correctness = torch.stack([
                    self.question_bank["model_pred_dict"][m][scores_topk_i]
                    for m in self.available_models
                ])  # [num_models, num_support]
                mean_correctness = all_correctness.float().mean(dim=0)  # [num_support]
                disc_mask = (mean_correctness > 0.1) & (mean_correctness < 0.9)
                disc_indices = scores_topk_i[disc_mask]
                min_examples = min(5, scores_topk_i.numel())
                if disc_indices.numel() >= min_examples:
                    scores_topk_i = disc_indices
            model_profile = {}
            for m in self.available_models:
                if weighted_score:
                    distance_weight = scores_topk_value[i][:len(scores_topk_i)].cpu() / 100
                    is_correct_score = self.question_bank['model_pred_dict'][m][scores_topk_i]
                    denominator = distance_weight.sum().clamp_min(1e-8)
                    model_profile[m] = ((distance_weight * is_correct_score).sum() / denominator).item()
                else:
                    model_profile[m] = (self.question_bank['model_pred_dict'][m][scores_topk_i]).sum().item()
            # Idea 3: Blend per-dataset prior bias into model scores (alpha=0.10)
            if dataset_labels is not None and i < len(dataset_labels):
                ds = dataset_labels[i]
                if ds in self.dataset_prior:
                    alpha = 0.10
                    for m in self.available_models:
                        if m in self.dataset_prior[ds]:
                            model_profile[m] = (1 - alpha) * model_profile[m] + alpha * self.dataset_prior[ds][m]
            model_profile_sorted = sorted(list(zip(range(len(model_profile)), model_profile.items())),
                                          key=lambda x: x[1][1], reverse=True)
            model_index_sorted, model_profile_sorted_zip = list(zip(*model_profile_sorted))
            rag_model_list, rag_score_list = list(zip(*model_profile_sorted_zip))
            results['rag_model_list'].append(rag_model_list)
            results['rag_score_list'].append(rag_score_list)
            results['score_topk_list'].append(scores_topk_i)
            results['score_topk_value_list'].append(scores_topk_value[i][:len(scores_topk_i)] / 100)
        return results

    def second_route_batch(self, batch_results_first_roll):
        weighted_score = self.config.weighted_score
        length_score_order = self.config.dev_length_score_order
        agg_model_num = self.config.sample_n
        subset_p = self.config.dev_subset_p
        embed_sim_score_mode = self.config.dev_embed_sim_score_mode # ['s2s', 'a2a', 'a2a-strong-x']
        length_score_coef = self.config.dev_length_score_coef
        query_score_coef = self.config.dev_query_score_coef

        rag_model_list_first_roll = batch_results_first_roll['rag_model_list']
        rag_score_value_list_first_roll = batch_results_first_roll['score_topk_value_list']
        score_topk_list = batch_results_first_roll['score_topk_list']
        batch_queries_idx = batch_results_first_roll['batch_idx']
        batch_num = len(rag_model_list_first_roll)
        self._get_response_model_order()
        results = {
            'rag_model_list': [],
            'rag_score_list': [],
            'score_topk_list': [],
            'score_topk_value_list': []
        }
        for i in range(batch_num):
            model_profile = {}
            model_list_first_roll_order_i_all = rag_model_list_first_roll[i]
            rag_model_list_first_roll_i = model_list_first_roll_order_i_all[:agg_model_num]
            score_topk_list_i = score_topk_list[i]
            rag_score_value_first_roll_i = rag_score_value_list_first_roll[i]
            batch_queries_idx_i = batch_queries_idx[i]
            second_rag_num = max(1, min(len(score_topk_list_i), int(len(score_topk_list_i) * subset_p)))
            resposne_embed_sim_score = self.compute_embed_sim_score(rag_model_list_first_roll_i, score_topk_list_i, batch_queries_idx_i, model_list_first_roll_order_i_all, embed_sim_score_mode=embed_sim_score_mode)
            # compute the response length score
            length_score = self.compute_length_score(rag_model_list_first_roll_i, score_topk_list_i, batch_queries_idx_i, score_order=length_score_order)
            refine_response_score = (1 - length_score_coef) * resposne_embed_sim_score + length_score_coef * length_score
            refine_score = (1 - query_score_coef) * refine_response_score + query_score_coef * rag_score_value_first_roll_i
            scores_topk_value, scores_topk = refine_score.topk(second_rag_num, dim=-1)
            scores_topk = score_topk_list_i[scores_topk.cpu()]
            for m in self.available_models:
                if weighted_score:
                    distance_weight = scores_topk_value.cpu()
                    is_correct_score = self.question_bank['model_pred_dict'][m][scores_topk]
                    denominator = distance_weight.sum().clamp_min(1e-8)
                    model_profile[m] = ((distance_weight * is_correct_score).sum() / denominator).item()
                else:
                    model_profile[m] = (self.question_bank['model_pred_dict'][m][scores_topk]).sum().item()
            model_profile_sorted = sorted(list(zip(range(len(model_profile)), model_profile.items())),
                                          key=lambda x: x[1][1], reverse=True)
            model_index_sorted, model_profile_sorted_zip = list(zip(*model_profile_sorted))
            rag_model_list, rag_score_list = list(zip(*model_profile_sorted_zip))
            results['rag_model_list'].append(rag_model_list)
            results['rag_score_list'].append(rag_score_list)
            results['score_topk_list'].append(scores_topk)
            results['score_topk_value_list'].append(refine_score)
        batch_results_first_roll['rag_model_list'] = results['rag_model_list']
        batch_results_first_roll['rag_score_list'] = results['rag_score_list']
        batch_results_first_roll['score_topk_list'] = results['score_topk_list']
        batch_results_first_roll['score_topk_value_list'] = results['score_topk_value_list']
        return batch_results_first_roll

    def compute_length_score(self, rag_model_list_first_roll_i, score_topk_list_i, batch_queries_idx_i, score_order=1):
        if len(rag_model_list_first_roll_i) >= 3:
            needle_model_range = [0, 1, 2]
        else:
            needle_model_range = [0]
        needle_models = np.array(rag_model_list_first_roll_i)[needle_model_range]
        # needle_models = ['qwen3-235b-a22b-thinking-2507']
        model_id = [self.model_id_map[m] for m in needle_models]
        rag_train_response_length = self.train_response_length[score_topk_list_i][:, model_id] # [qestion_n, model_n]
        test_response_length = self.test_response_length[batch_queries_idx_i:batch_queries_idx_i + 1][
            :, model_id]  # [1, model_n]
        if "proxy_s2s" in self.config.dev_embed_sim_score_mode:
            proxy_num = int(self.config.dev_embed_sim_score_mode.split('_')[-1])
            proxy_train_response_length = self.train_response_length[score_topk_list_i[:proxy_num]][
                :, model_id]  # [qestion_n, model_n]
            if score_order == 1:
                diff_train_proxy = (rag_train_response_length[:,None] - proxy_train_response_length[None]).abs()
                length_score = (1 - diff_train_proxy / (diff_train_proxy.max(dim=0, keepdim=True)[0]+1e-8)).mean(dim=(1,2))
            elif score_order == 2:
                diff_train_proxy = (rag_train_response_length[:,None]**2 - proxy_train_response_length[None]**2).abs()
                length_score = (1 - diff_train_proxy / (diff_train_proxy.max(dim=0, keepdim=True)[0]+1e-8)).mean(dim=(1,2))
        else:
            if score_order == 1:
                diff_train_test = (rag_train_response_length - test_response_length).abs()
                length_score = (1 - diff_train_test / (diff_train_test.max(dim=0, keepdim=True)[0]+1e-8)).mean(dim=-1)
            elif score_order == 2:
                diff_train_test = (rag_train_response_length**2 - test_response_length**2).abs()
                length_score = (1 - diff_train_test / (diff_train_test.max(dim=0, keepdim=True)[0]+1e-8)).mean(dim=-1)
        return length_score

    def compute_embed_sim_score(self, rag_model_list_first_roll_i, score_topk_list_i, batch_queries_idx_i, model_list_first_roll_order_i_all, embed_sim_score_mode):
        if embed_sim_score_mode == 's2s':
            if len(rag_model_list_first_roll_i) >= 3:
                needle_model_range = [0,1,2]
            else:
                needle_model_range = [0]
            needle_models = np.array(rag_model_list_first_roll_i)[needle_model_range]
            # needle_models = ['qwen3-235b-a22b-thinking-2507']
            model_train_res_embed = self._build_response_embedding_bank(needle_models, score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]
            model_test_res_embed = self._build_response_embedding_bank(needle_models, [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            response_embed_sim_score = \
            (model_train_res_embed.transpose(0, 1) @ model_test_res_embed.permute(1, 2, 0)).mean(dim=0)[:, 0]
        elif "proxy_s2s" in embed_sim_score_mode:
            proxy_num = int(embed_sim_score_mode.split('_')[-1])
            model_num = 2
            try_range = range(1,2)
            model_test_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            model_train_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i, score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]


            # model_train_top_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i, score_topk_list_i[:proxy_num],
            #                                                             split='train')  # [ 10, model_n, d]
            model_train_top_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                            score_topk_list_i,
                                                                            split='train')  # [ 10, model_n, d]
            # resposne_embed_sim_score = \
            #     (model_train_res_embed.transpose(0, 1) @ model_train_top_res_embed.permute(1, 2, 0)).mean(dim=0).mean(dim=-1)

            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1) @ model_train_top_res_embed.permute(1, 2, 0)).mean(dim=0)
            response_embed_sim_score_top_model = \
                (model_train_res_embed[:, try_range, :].transpose(0, 1) @ model_test_res_embed[:, try_range, :].permute(1, 2, 0)).mean(dim=0)
            response_embed_sim_score_top_model_max = response_embed_sim_score_top_model.argmax()
            response_embed_sim_score_true = \
                (model_train_res_embed.transpose(0, 1) @ model_test_res_embed.permute(1, 2, 0)).mean(dim=0)
            # response_embed_sim_score = response_embed_sim_score[:, (response_embed_sim_score - response_embed_sim_score_true).abs().mean(dim=0).argmin()]
            response_embed_sim_score = response_embed_sim_score[:, response_embed_sim_score_top_model_max]
        elif "needle_proxy_s2s" in embed_sim_score_mode:
            try_range = range(1,2)
            model_test_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            model_train_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i, score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]


            # model_train_top_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i, score_topk_list_i[:proxy_num],
            #                                                             split='train')  # [ 10, model_n, d]
            model_train_top_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                            score_topk_list_i,
                                                                            split='train')  # [ 10, model_n, d]
            # resposne_embed_sim_score = \
            #     (model_train_res_embed.transpose(0, 1) @ model_train_top_res_embed.permute(1, 2, 0)).mean(dim=0).mean(dim=-1)

            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1) @ model_train_top_res_embed.permute(1, 2, 0)).mean(dim=0)

            response_embed_sim_score_top_model = \
                (model_train_res_embed[:, try_range, :].transpose(0, 1) @ model_test_res_embed[:, try_range, :].permute(1, 2, 0)).mean(dim=0)

            response_embed_sim_score_top_model_max = response_embed_sim_score_top_model.argmax()
            response_embed_sim_score_true = \
                (model_train_res_embed.transpose(0, 1) @ model_test_res_embed.permute(1, 2, 0)).mean(dim=0)
            # response_embed_sim_score = response_embed_sim_score[:, (response_embed_sim_score - response_embed_sim_score_true).abs().mean(dim=0).argmin()]
            response_embed_sim_score = response_embed_sim_score[:, response_embed_sim_score_top_model_max]

        elif embed_sim_score_mode == 's2s-weak':
            model_train_res_embed = self._build_response_embedding_bank(model_list_first_roll_order_i_all[-3:], score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]
            model_test_res_embed = self._build_response_embedding_bank(model_list_first_roll_order_i_all[-3:],
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1) @ model_test_res_embed.permute(1, 2, 0)).mean(dim=0)[:, 0]
        elif embed_sim_score_mode == 's2s-mid':
            model_train_res_embed = self._build_response_embedding_bank(model_list_first_roll_order_i_all[3:6], score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]
            model_test_res_embed = self._build_response_embedding_bank(model_list_first_roll_order_i_all[3:6],
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1) @ model_test_res_embed.permute(1, 2, 0)).mean(dim=0)[:, 0]
        elif embed_sim_score_mode == 'a2a':
            model_train_res_embed = self._build_response_embedding_bank(self.available_models, score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]
            model_test_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1)[:,None] @ model_test_res_embed.permute(1, 2, 0)[None]).mean(dim=(0,1))[:, 0]
        elif 'a2a-strong' in embed_sim_score_mode:
            strong_num = int(embed_sim_score_mode.split('-')[-1])
            model_train_res_embed = self._build_response_embedding_bank(model_list_first_roll_order_i_all[:strong_num], score_topk_list_i,
                                                                        split='train')  # [ rag_num, model_n, d]
            model_test_res_embed = self._build_response_embedding_bank(rag_model_list_first_roll_i,
                                                                       [batch_queries_idx_i],
                                                                       split="test")  # [1, model_n,  d]
            response_embed_sim_score = \
                (model_train_res_embed.transpose(0, 1)[:,None] @ model_test_res_embed.permute(1, 2, 0)[None]).mean(dim=(0,1))[:, 0]

        return response_embed_sim_score

    def _get_response_model_order(self) -> List[str]:
        """Return the stable model order used by response embeddings."""
        if self.response_models:
            return self.response_models
        if not self.train_data:
            raise RuntimeError("train_data must be loaded before building response embeddings")

        raw_output = self.train_data[0].get("raw_output", {})
        ordered_models = [m for m in raw_output if m in self.available_models]
        ordered_models.extend([m for m in raw_output if m not in ordered_models])
        if not ordered_models:
            raise ValueError("JiSi data must include raw_output model responses for response-side routing")

        self.response_models = ordered_models
        self.model_id_map = {m: i for i, m in enumerate(self.response_models)}
        return self.response_models

    def _get_response_text(self, item: Dict[str, Any], model: str) -> str:
        response = item.get("raw_output", {}).get(model, "")
        return response if isinstance(response, str) else str(response)

    def _get_response_completion_tokens(self, item: Dict[str, Any], model: str) -> int:
        usage = item.get("usages", {}).get(model, {})
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(completion_tokens, (int, float)) and completion_tokens >= 0:
            return int(completion_tokens)
        return len(self.ds_tokenizer.encode(self._get_response_text(item, model)))

    def _build_response_embedding_bank(self, rag_model_list_first_roll, score_topk_list, split):
        all_model_response_embedding = self.train_response_embed if split == 'train' else self.test_response_embed
        self._get_response_model_order()
        missing_models = [m for m in rag_model_list_first_roll if m not in self.model_id_map]
        if missing_models:
            raise ValueError(f"Missing raw_output responses for models: {missing_models}")
        model_id = [self.model_id_map[m] for m in rag_model_list_first_roll]
        return all_model_response_embedding[score_topk_list][:, model_id]

    def build_train_response_embedding(self):
        response_models = self._get_response_model_order()
        response_list = []
        model_num = len(response_models)
        data_num = len(self.train_data)
        train_response_embed_cache = Path(self.config.baseline_scores_path).parent.parent / "train_response_embed.tar"
        if not os.path.exists(train_response_embed_cache):
            for d in self.train_data:
                for m in response_models:
                    response_list.append(self._get_response_text(d, m) or " ")
            all_model_response_embedding = self._generate_embeddings_concurrent(response_list)
            train_response_embed = torch.tensor(all_model_response_embedding).to(self.device)
            train_response_embed = F.normalize(train_response_embed, dim=-1)
            train_response_embed = train_response_embed.view(data_num, model_num, -1)
            torch.save(train_response_embed, train_response_embed_cache)
        else:
            train_response_embed = torch.load(train_response_embed_cache, map_location=self.device).to(self.device)
            if train_response_embed.size(0) != data_num or train_response_embed.size(1) != model_num:
                raise ValueError(
                    f"Cached train_response_embed shape {tuple(train_response_embed.shape)} does not match "
                    f"current data/model shape ({data_num}, {model_num}, dim). Remove {train_response_embed_cache}."
                )
        # build length matrix
        response_length_list = []
        for d in self.train_data:
            for m in response_models:
                response_length_list.append(self._get_response_completion_tokens(d, m))
        response_length_tensor = torch.tensor(response_length_list).to(self.device).view(data_num, model_num)
        self.train_response_length = response_length_tensor
        self.train_response_embed = train_response_embed

    def build_test_response_embedding(self):
        response_models = self._get_response_model_order()
        model_num = len(response_models)
        data_num = len(self.actual_test_data)
        test_response_embed_cache = Path(self.config.baseline_scores_path).parent.parent / "test_response_embed.tar"
        if not os.path.exists(test_response_embed_cache):
            response_list = []
            for d in self.actual_test_data:
                for m in response_models:
                    response_list.append(self._get_response_text(d, m) or " ")
            all_model_response_embedding = self._generate_embeddings_concurrent(response_list)
            test_response_embed = torch.tensor(all_model_response_embedding).to(self.device)
            test_response_embed = F.normalize(test_response_embed, dim=-1)
            test_response_embed = test_response_embed.view(data_num, model_num, -1)
            torch.save(test_response_embed, test_response_embed_cache)
        else:
            test_response_embed = torch.load(test_response_embed_cache, map_location=self.device).to(self.device)
            if test_response_embed.size(0) != data_num or test_response_embed.size(1) != model_num:
                raise ValueError(
                    f"Cached test_response_embed shape {tuple(test_response_embed.shape)} does not match "
                    f"current data/model shape ({data_num}, {model_num}, dim). Remove {test_response_embed_cache}."
                )
        # build length matrix
        response_length_list = []
        for d in self.actual_test_data:
            for m in response_models:
                response_length_list.append(self._get_response_completion_tokens(d, m))
        response_length_tensor = torch.tensor(response_length_list).to(self.device).view(data_num, model_num)
        self.test_response_length = response_length_tensor
        self.test_response_embed = test_response_embed

    def evaluate_routing(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate routing performance on test data."""
        # Filter out excluded datasets from test data (safety check)
        filtered_test_data = []
        excluded_count = 0
        for item in test_data:
            dataset_name = item.get('dataset', 'default')
            if dataset_name in self.config.excluded_datasets:
                excluded_count += 1
                continue
            filtered_test_data.append(item)

        if excluded_count > 0:
            self.logger.warning(f"Filtered out {excluded_count} test items from excluded datasets (should have been filtered earlier)")

        actual_test_data = filtered_test_data
        self.actual_test_data = actual_test_data
        results = {
            'total_queries': len(actual_test_data),
            'correct_routes': 0,
            'dataset_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'model_selection_stats': Counter(),
            'dataset_model_selection': defaultdict(lambda: Counter()),
            'dataset_model_accuracy': defaultdict(lambda: defaultdict(list)),
            'routing_details': [],
            'accuracy': 0.0,
            'ood_accuracy': 0.0,
            'non_ood_accuracy': 0.0,
            'ood_sample_avg': 0.0,
            'non_ood_sample_avg': 0.0,
            'all_sample_avg': 0.0,
        }

        # Extract all queries for batch processing
        queries = [item["query"] for item in actual_test_data]
        if not queries:
            self.logger.warning("No test queries remain after dataset filtering")
            return results

        self.build_train_response_embedding()
        self.build_test_response_embedding()
        print(f"\n=== Evaluating routing on {len(actual_test_data)} test queries ===")

        queries_idx = [i for i, item in enumerate(actual_test_data)]
        print(f"Running query-response-based routing for {len(queries)} queries...")
        select_round = 1
        # Use batch routing for efficiency
        with tqdm(total=len(queries), desc="Batch retrieval routing") as pbar:
            batch_size = min(self.config.routing_batch_size, len(queries))
            all_routing_results, all_routing_scores = [], []
            all_routing_results_first_roll, all_routing_scores_first_roll = [], []

            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_queries_idx = queries_idx[i:i + batch_size]
                batch_results = self.route_queries_batch(batch_queries, [actual_test_data[j]["dataset"] for j in batch_queries_idx])
                batch_results['batch_idx'] = batch_queries_idx
                batch_results_first_roll = copy.deepcopy(batch_results)
                for _ in range(select_round):
                    batch_results = self.second_route_batch(batch_results)
                route_models, route_scores = batch_results['rag_model_list'], batch_results['rag_score_list']
                all_routing_results.extend(route_models)
                all_routing_scores.extend(route_scores)
                all_routing_results_first_roll.extend(batch_results_first_roll['rag_model_list'])
                all_routing_scores_first_roll.extend(batch_results_first_roll['rag_score_list'])
                pbar.update(len(batch_queries))

        print("Processing routing results...")
        if self.config.mode == 'router':
            results = self.run_router_mode(actual_test_data, all_routing_results, results)
        elif self.config.mode == 'aggregator':
            output_path = self.run_aggregator_mode(actual_test_data, all_routing_results, all_routing_scores,
                                        all_routing_results_first_roll, all_routing_scores_first_roll, results)
            results['aggregation_output_path'] = output_path
            return results
        else:
            pass
        # Calculate accuracy separately for OOD and non-OOD datasets
        ood_accuracies = []
        non_ood_accuracies = []

        # Also calculate sample-level accuracies
        ood_correct_samples = 0
        ood_total_samples = 0
        non_ood_correct_samples = 0
        non_ood_total_samples = 0

        for dataset, perf in results['dataset_performance'].items():
            if perf['total'] > 0:
                dataset_accuracy = perf['correct'] / perf['total']
                # Store accuracy as percentage (2 decimal places, no % symbol)
                perf['accuracy'] = round(dataset_accuracy * 100, 2)

                if dataset in self.config.ood_datasets:
                    ood_accuracies.append(dataset_accuracy)
                    ood_correct_samples += perf['correct']
                    ood_total_samples += perf['total']
                else:
                    non_ood_accuracies.append(dataset_accuracy)
                    non_ood_correct_samples += perf['correct']
                    non_ood_total_samples += perf['total']

        # Store separate dataset-level accuracies as percentages (2 decimal places)
        results['ood_accuracy'] = round((sum(ood_accuracies) / len(ood_accuracies) if ood_accuracies else 0.0) * 100, 2)
        results['non_ood_accuracy'] = round((sum(non_ood_accuracies) / len(non_ood_accuracies) if non_ood_accuracies else 0.0) * 100, 2)

        # Store sample-level accuracies as percentages (2 decimal places)
        results['ood_sample_avg'] = round((ood_correct_samples / ood_total_samples if ood_total_samples > 0 else 0.0) * 100, 2)
        results['non_ood_sample_avg'] = round((non_ood_correct_samples / non_ood_total_samples if non_ood_total_samples > 0 else 0.0) * 100, 2)
        results['all_sample_avg'] = round((results['correct_routes'] / len(results['routing_details']) if results['routing_details'] else 0.0) * 100, 2)

        # Calculate overall accuracy as average of all datasets (as percentage, 2 decimal places)
        all_accuracies = ood_accuracies + non_ood_accuracies
        results['accuracy'] = round((sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0) * 100, 2)

        # Add cost analysis using filtered test data
        cost_analysis = self._analyze_routing_costs(actual_test_data, results['routing_details'], results)
        results['cost_analysis'] = cost_analysis

        # Add baseline comparison data for export
        baseline_scores = self._load_baseline_scores()
        if baseline_scores:
            baseline_analysis = self._analyze_baseline_comparison(baseline_scores, results, actual_test_data)
            results['baseline_analysis'] = baseline_analysis

        return results

    def run_router_mode(self, actual_test_data, all_routing_results, results):
        with tqdm(total=len(actual_test_data), desc="Evaluating results") as pbar:
            for i, test_item in enumerate(actual_test_data):
                selected_models = all_routing_results[i]
                selection_count = max(1, min(self.config.max_router, self.config.top_k))
                selected_models = selected_models[:selection_count]
                routing_result = {
                    'dataset': test_item['dataset'],
                    'index': test_item.get('index', -1),
                    'selected_models': list(selected_models),
                    'is_correct': 0.0,
                    'true_records': {k: v for k, v in test_item['records'].items()
                                     if k in self.available_models}
                }

                # Check if routing is correct - use the best score from selected models
                if selected_models:
                    max_score = 0.0
                    for model_name in selected_models:
                    # for model_name in self.available_models:
                        if model_name in test_item['records']:
                            score = test_item['records'][model_name]
                            if score > max_score:
                                max_score = score
                    routing_result['is_correct'] = max_score

                # Update statistics - use float score for correct routes
                results['correct_routes'] += routing_result['is_correct']
                results['dataset_performance'][routing_result['dataset']]['correct'] += routing_result['is_correct']

                results['dataset_performance'][routing_result['dataset']]['total'] += 1

                # Record model selection
                dataset = routing_result['dataset']
                for model in routing_result['selected_models']:
                    results['model_selection_stats'][model] += 1
                    results['dataset_model_selection'][dataset][model] += 1
                    results['dataset_model_accuracy'][dataset][model].append(routing_result['is_correct'])

                results['routing_details'].append(routing_result)

                current_accuracy = results['correct_routes'] / len(results['routing_details'])
                pbar.set_postfix({'accuracy': f'{current_accuracy:.4f}'})
        return results

    def _filter_models_with_outputs(self, test_item, models, scores):
        raw_output = test_item.get('raw_output', {})
        filtered_models, filtered_scores = [], []
        for model, score in zip(models, scores):
            response = raw_output.get(model, "")
            if isinstance(response, str) and response.strip():
                filtered_models.append(model)
                filtered_scores.append(score)
        return filtered_models, filtered_scores

    def _fallback_models_with_outputs(self, test_item):
        raw_output = test_item.get('raw_output', {})
        fallback_models = [
            model for model in self.available_models
            if isinstance(raw_output.get(model, ""), str) and raw_output.get(model, "").strip()
        ]
        if fallback_models:
            return fallback_models
        return [
            model for model, response in raw_output.items()
            if isinstance(response, str) and response.strip()
        ]

    def run_aggregator_mode(self, actual_test_data, all_routing_results, all_routing_scores,
                        all_routing_results_first_roll, all_routing_scores_first_roll, results):
        select_n = self.config.select_n
        config_file_path = self.config.config_file_path
        config_base_name = (
            os.path.splitext(os.path.basename(config_file_path))[0]
            if config_file_path else "jisi_cli"
        )
        config_name = f"{config_base_name}_{self.config.agg_model}"
        result_dir = os.path.join(self.config.result_dir, config_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        if config_file_path and os.path.exists(config_file_path):
            shutil.copy2(config_file_path, os.path.join(result_dir, os.path.basename(config_file_path)))
        output_res_path = os.path.join(result_dir, f"result.jsonl")
        if os.path.exists(output_res_path):
            with jsonlines.Reader(open(output_res_path, 'r', encoding='utf-8')) as f:
                done_list = list(f)
            done_run_id = [q['run_id'] for q in done_list]
        else:
            done_run_id = []
        # Use one executor to consume all pending items without batch-level blocking.
        done_run_id_set = set(done_run_id)
        pending_ids = [i for i in range(len(actual_test_data)) if i not in done_run_id_set]
        with tqdm(total=len(actual_test_data), desc="Evaluating results") as pbar:
            if done_run_id_set:
                pbar.update(len(done_run_id_set))
            if pending_ids:
                max_workers = min(self.config.process_batch_size, len(pending_ids))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_id = {}
                    for data_id in pending_ids:
                        test_item = actual_test_data[data_id]
                        all_routing_results_filter, all_routing_scores_filter = self._filter_models_with_outputs(
                            test_item,
                            all_routing_results[data_id],
                            all_routing_scores[data_id],
                        )
                        all_routing_results_filter_first_roll, all_routing_scores_filter_first_roll = self._filter_models_with_outputs(
                            test_item,
                            all_routing_results_first_roll[data_id],
                            all_routing_scores_first_roll[data_id],
                        )
                        if not all_routing_results_filter:
                            fallback_models = self._fallback_models_with_outputs(test_item)
                            if not fallback_models:
                                raise ValueError(f"No non-empty raw_output found for test item {data_id}")
                            all_routing_results_filter = fallback_models[:1]
                            all_routing_scores_filter = [1.0]
                        if not all_routing_results_filter_first_roll:
                            all_routing_results_filter_first_roll = all_routing_results_filter
                            all_routing_scores_filter_first_roll = all_routing_scores_filter
                        task_args = [
                            data_id,
                            all_routing_results_filter[:min(select_n, len(all_routing_results_filter))],
                            all_routing_scores_filter[:min(select_n, len(all_routing_scores_filter))],
                            all_routing_results_filter_first_roll[:min(select_n, len(all_routing_results_filter_first_roll))],
                            all_routing_scores_filter_first_roll[:min(select_n, len(all_routing_scores_filter_first_roll))],
                            test_item
                        ]
                        future = executor.submit(self._run_aggregator_mode_single_instance, *task_args)
                        future_to_id[future] = data_id
                    # Writes happen on the main thread; flush so resumed runs see progress promptly.
                    with open(output_res_path, 'a', encoding='utf-8') as out_fp:
                        with jsonlines.Writer(out_fp) as f_result_handle:
                            for future in as_completed(future_to_id):
                                data_id_range_i, response_pred_dict = future.result()
                                each = actual_test_data[data_id_range_i]
                                each.update(response_pred_dict)
                                each["run_id"] = data_id_range_i
                                f_result_handle.write(each)
                                out_fp.flush()
                                pbar.update(1)
        print("*"*20+"Finish generating results!"+"*"*20)
        print("*"*20+"Attention!!! For JiSi aggregator mode, the test will be conducted in other file!" + "*"*20)
        return output_res_path

    def auto_clean_think(self, references):
        """Reduce aggregation input size while preserving earlier experts' reasoning.

        Tail experts are cleaned first, one at a time. If removing full thinking
        blocks from every expert is still not enough, only the think tags are
        stripped as a final fallback.
        """
        agg_input_max_tokens = 90000
        new_refs = copy.deepcopy(references)
        n = len(new_refs)
        if n == 0:
            return new_refs

        def total_tokens(refs):
            return sum(len(self.ds_tokenizer.encode(r)) for r in refs)

        if total_tokens(new_refs) <= agg_input_max_tokens:
            return new_refs

        for j in range(n - 1, -1, -1):
            new_refs[j] = self.clean_think([new_refs[j]])[0]
            if total_tokens(new_refs) <= agg_input_max_tokens:
                return new_refs

        return [self.clean_think_token(r) for r in new_refs]

    def _run_aggregator_mode_single_instance(self, data_id, selected_models, routing_scores, selected_models_fr, routing_scores_fr, test_item):
        # Each worker thread owns an event loop and executor to avoid cross-thread deadlocks.
        import threading
        thread_id = threading.current_thread().ident
        if not hasattr(self, '_thread_executors'):
            self._thread_executors = {}
        if thread_id not in self._thread_executors:
            self._thread_executors[thread_id] = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"async_exec_{thread_id}")

        original_executor = getattr(self, '_thread_executor', None)
        self._thread_executor = self._thread_executors[thread_id]

        async def wrap_rag_moa_test_(data_id, selected_models, routing_scores, selected_models_fr, routing_scores_fr, test_item):
            # Get all configs
            force_agg_num = self.config.dev_force_agg_num
            agg_N = self.config.agg_N
            max_tokens = self.config.agg_max_tokens
            ppl_coef = self.config.ppl_coef
            messages = [{"role": "user", "content": test_item['query']}]
            _, expert_model_source, _, agg_model_source = self.config.dev_re_route_mode.split('_')
            # Get references of all experts
            references_models = selected_models if expert_model_source == 'response' else selected_models_fr
            reference_scores = routing_scores if expert_model_source == 'response' else routing_scores_fr
            references = [
                self._get_response_text(test_item, m)
                for m in references_models
                if self._get_response_text(test_item, m).strip()
            ]
            if not references:
                raise ValueError(f"No non-empty expert references available for test item {data_id}")
            references = references[:force_agg_num]
            reference_scores = list(reference_scores)[:len(references)]
            # Adaptive Routing-Aggregation Switch
            agg_ref_num = self.routing_aggregation_switch(reference_scores, references)
            if agg_ref_num < 3:
                print(f"switch to agg num: {agg_ref_num}")
            references = references[:agg_ref_num]
            reference_scores = reference_scores[:agg_ref_num]
            # clean the reference for less tokens
            if agg_ref_num > 1:
                references = self.auto_clean_think(references)
                sub_ref = [references]
                # Get the aggregation results
                agg_candidate_models = selected_models if agg_model_source == 'response' else selected_models_fr
                agg_candidate_models = [
                    m for m in agg_candidate_models
                    if self._get_response_text(test_item, m).strip()
                ]
                model = self.config.agg_model
                if self.config.agg_model == "auto":
                    model = agg_candidate_models[0] if agg_candidate_models else 'No agg model'
                raw_responses = []
                raw_responses_token_num = []
                if self.config.agg_model == "auto":
                    for k in range(len(agg_candidate_models)):
                        model = agg_candidate_models[k]
                        if 'deepseek-v3-' in model:
                            run_max_tokens = min(max_tokens, 16384)
                        else:
                            run_max_tokens = max_tokens
                        agg_task_list = [{
                            "model": model,
                            "messages": messages,
                            "temperature": self.config.agg_temperature,
                            "top_p": 1.0,
                            "max_tokens": run_max_tokens,
                            "references": sub_ref_i,
                            "logprobs": None,
                            "agg_prompt": self.config.dev_agg_prompt,
                            "ref_score": reference_scores
                        } for sub_ref_i in sub_ref]
                        agg_tasks = [self.async_generate_with_references(**agg_task_list_i) for agg_task_list_i in
                                     agg_task_list]
                        try:
                            raw_responses_out = await asyncio.gather(*agg_tasks)
                            raw_responses_token_num = [r.completion_tokens for r in raw_responses_out]
                            raw_responses = [r.output for r in raw_responses_out]
                            for r in raw_responses:
                                if "Generation failed" in r:
                                    raise ValueError("Generation failed, change to another model")
                        except Exception:
                            continue
                        has_none_response = False
                        for raw_response_i in raw_responses:
                            if len(raw_response_i) == 0:
                                has_none_response = True
                                break
                        if has_none_response:
                            continue
                        break
                    if not raw_responses:
                        fallback_response = references[0]
                        raw_responses = [fallback_response]
                        raw_responses_token_num = [len(self.ds_tokenizer.encode(fallback_response))]
                        model = 'No agg model'
                else:
                    run_max_tokens = max_tokens
                    agg_task_list = [{
                        "model": model,
                        "messages": messages,
                        "temperature": self.config.agg_temperature,
                        "max_tokens": run_max_tokens,
                        "references": sub_ref_i,
                        "logprobs": None,
                        "agg_prompt": self.config.dev_agg_prompt,
                        "ref_score": reference_scores
                    } for sub_ref_i in sub_ref]
                    agg_tasks = [self.async_generate_with_references(**agg_task_list_i) for agg_task_list_i in
                                 agg_task_list]
                    try:
                        raw_responses_out = await asyncio.gather(*agg_tasks)
                        raw_responses_token_num = [r.completion_tokens for r in raw_responses_out]
                        raw_responses = [r.output for r in raw_responses_out]
                        for r in raw_responses:
                            if "Generation failed" in r:
                                raise ValueError("Generation failed, change to another model")
                    except Exception as exc:
                        print(f'Aggregator error ({exc}); using the top expert response.')
                        fallback_response = references[0]
                        raw_responses = [fallback_response]
                        raw_responses_token_num = [len(self.ds_tokenizer.encode(fallback_response))]
                select_score = {i: {'ppl_score': 0.0, 'sc_score': 0.0, 'total_score': 0.0} for i in
                                range(len(raw_responses))}
                # select the best result
                if not isinstance(raw_responses[0], str):
                    mean_cumulative_logprob = [r['cumulative_logprob'] for r in raw_responses]
                    ppl = np.exp(mean_cumulative_logprob)
                    responses = [r['response'] for r in raw_responses]
                    ppl_score = 1 - ppl
                else:
                    responses = raw_responses
                    mean_cumulative_logprob = [None for r in raw_responses]
                    ppl, ppl_score = None, None
                # Use self consistency and ppl
                sc_memory = {}
                add_scores = torch.tensor([1.0])
                for add_score_i, response in zip(add_scores, responses):
                    sc_memory[response] = add_score_i.item()
                for i, response_i in enumerate(responses):
                    select_score[i]['sc_score'] = sc_memory[response_i] / agg_N
                    if ppl_score is not None:
                        select_score[i]['ppl_score'] = ppl_score[i]
                        select_score[i]['total_score'] = sc_memory[response_i] / agg_N + ppl_coef * ppl_score[i]
                    else:
                        select_score[i]['ppl_score'] = None
                        select_score[i]['total_score'] = sc_memory[response_i]
                response = responses[sorted(select_score, key=lambda x: select_score[x]['total_score'], reverse=True)[0]]
            else:
                response = references[0]
                responses = [response]
                raw_responses_token_num = [len(self.ds_tokenizer.encode(response))]
                select_score = {}
                model = 'No agg model'
                sc_memory = {}
            # return all the results
            return_dict = {}
            return_dict['sc_memory'] = sc_memory
            return_dict['response'] = response
            return_dict['n_response'] = responses
            return_dict['n_response_usage'] = raw_responses_token_num
            return_dict['rag_model'] = selected_models
            return_dict['rag_model_fr'] = selected_models_fr
            return_dict['routing_scores'] = routing_scores
            return_dict['select_score'] = select_score
            return_dict['agg_model'] = model
            return_dict['agg_ref_num'] = agg_ref_num
            return (data_id, return_dict)

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        try:
            result = new_loop.run_until_complete(wrap_rag_moa_test_(data_id, selected_models, routing_scores, selected_models_fr, routing_scores_fr, test_item))
            return result
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)
            self._thread_executor = original_executor

    def routing_aggregation_switch(self, routing_scores, references):
        divide_t = self.config.divide_t
        cut_length = self.config.cut_length
        routing_scores_arr = np.array(routing_scores)
        if routing_scores_arr.size == 0 or not references:
            return 1
        max_score = routing_scores_arr.max()
        if max_score <= 0:
            return 1
        routing_scores_norm = routing_scores_arr / max_score
        filter_num = (routing_scores_norm >= divide_t).sum()
        agg_num = max(1, min(int(filter_num), len(references)))
        if filter_num > 2:
            ref_tokens_num = sum([len(self.ds_tokenizer.encode(r)) for r in references])
            if ref_tokens_num > cut_length:
                agg_num = max(1, min(int(filter_num - 1), len(references)))
        return int(agg_num)

    def _analyze_routing_costs(self, test_data: List[Dict], routing_details: List[Dict], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the cost implications of routing decisions.
        
        Args:
            test_data: Original test dataset with cost information
            routing_details: Detailed routing results
            results: Overall routing results for context
            
        Returns:
            Cost analysis dictionary
        """
        total_cost = 0.0
        total_queries = len(test_data)
        model_costs = defaultdict(float)
        model_queries = defaultdict(int)
        cost_per_dataset = defaultdict(float)
        correct_cost = 0.0
        incorrect_cost = 0.0
        ood_total_cost = 0.0
        non_ood_total_cost = 0.0

        for i, (test_item, routing_result) in enumerate(zip(test_data, routing_details)):
            selected_models = routing_result['selected_models']
            is_correct = routing_result['is_correct']
            dataset = routing_result['dataset']
            
            # Skip excluded datasets (safety check - should already be filtered)
            if dataset in self.config.excluded_datasets:
                continue
            
            # Get cost for selected models
            query_cost = 0.0
            usages = test_item.get('usages')
            
            for model_name in selected_models:
                cost = None
                if (usages is not None and 
                    model_name in usages and 
                    isinstance(usages[model_name], dict) and
                    'cost' in usages[model_name]):
                    cost = usages[model_name]['cost']
                
                # If no cost data available, assign zero cost
                if cost is None or not isinstance(cost, (int, float)) or cost < 0:
                    cost = 0.0  # No cost penalty for missing data
                    
                query_cost += cost
                model_costs[model_name] += cost
                model_queries[model_name] += 1
            
            total_cost += query_cost
            cost_per_dataset[dataset] += query_cost

            # Track OOD vs non-OOD costs
            if dataset in self.config.ood_datasets:
                ood_total_cost += query_cost
            else:
                non_ood_total_cost += query_cost

            if is_correct:
                correct_cost += query_cost
            else:
                incorrect_cost += query_cost
        
        # Get accuracy from results for cost efficiency calculation
        accuracy = results.get('accuracy', 0.0)
        correct_routes = results.get('correct_routes', 0)
        
        # Calculate cost-efficiency metrics
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0
        cost_per_correct = correct_cost / correct_routes if correct_routes > 0 else 0.0
        
        # Cost efficiency: accuracy per unit cost (using simple average accuracy)
        overall_cost_efficiency = accuracy / (avg_cost_per_query + 1e-8)
        
        cost_analysis = {
            'total_cost': total_cost,
            'avg_cost_per_query': avg_cost_per_query,
            'cost_per_correct_prediction': cost_per_correct,
            'cost_efficiency': overall_cost_efficiency,
            'model_costs': dict(model_costs),
            'dataset_costs': dict(cost_per_dataset),
            'cost_distribution': {
                'correct_predictions': correct_cost,
                'incorrect_predictions': incorrect_cost
            },
            'ood_total_cost': ood_total_cost,
            'non_ood_total_cost': non_ood_total_cost
        }
        
        return cost_analysis

    def _load_baseline_scores(self) -> Dict[str, Dict[str, float]]:
        """Load baseline scores from configured path."""
        baseline_path = Path(self.config.baseline_scores_path)
        
        if not baseline_path.exists():
            self.logger.error(f"Baseline scores file not found: {baseline_path}")
            raise FileNotFoundError(f"Baseline scores file not found: {baseline_path}")
        
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_scores = json.load(f)
                self.logger.info(f"Loaded baseline scores from {baseline_path}")
                return baseline_scores
        except Exception as e:
            self.logger.error(f"Failed to load baseline scores from {baseline_path}: {e}")
            raise RuntimeError(f"Failed to load baseline scores: {e}")

    def _calculate_baseline_analysis(self, baseline_scores: Dict[str, Dict[str, float]], 
                                   results: Dict[str, Any], test_data: List[Dict], 
                                   dataset_filter: str = "all") -> Dict[str, Any]:
        """
        Calculate baseline analysis for specified dataset type.
        
        Args:
            baseline_scores: Baseline scores from config/baseline.json
            results: Router evaluation results
            test_data: Test data with cost information
            dataset_filter: "all", "ood", or "non_ood"
            
        Returns:
            Baseline analysis dictionary for specified dataset type
        """
        # Calculate per-model baseline summary
        model_summaries = []
        total_cost_by_model = {}
        
        # Calculate total costs per model from test data if available
        if test_data:
            for item in test_data:
                dataset_name = item.get('dataset', 'default')
                # Skip excluded datasets from cost calculation
                if dataset_name in self.config.excluded_datasets:
                    continue
                
                # Filter by dataset type
                if dataset_filter == "ood" and dataset_name not in self.config.ood_datasets:
                    continue
                elif dataset_filter == "non_ood" and dataset_name in self.config.ood_datasets:
                    continue
                    
                usages = item.get('usages', {})
                for model_name, usage in usages.items():
                    if isinstance(usage, dict) and 'cost' in usage:
                        cost = usage.get('cost', 0.0)
                        if isinstance(cost, (int, float)) and cost > 0:
                            total_cost_by_model[model_name] = total_cost_by_model.get(model_name, 0.0) + cost
        
        for model, scores in baseline_scores.items():
            # Skip excluded models
            if model in self.config.excluded_models:
                continue
                
            # Calculate average score (excluding null values and excluded datasets)
            valid_scores = []
            total_datasets = 0
            for dataset, score in scores.items():
                if dataset in self.config.excluded_datasets:
                    continue
                
                # Filter by dataset type
                if dataset_filter == "ood" and dataset not in self.config.ood_datasets:
                    continue
                elif dataset_filter == "non_ood" and dataset in self.config.ood_datasets:
                    continue
                    
                total_datasets += 1
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    valid_scores.append(score)
            
            if total_datasets > 0:  # Only include models that have relevant datasets
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                
                # Count dataset coverage (excluding excluded datasets)
                dataset_coverage = f"{len(valid_scores)}/{total_datasets}"
                
                # Get total cost
                total_cost = total_cost_by_model.get(model, 0.0)
                
                model_summaries.append({
                    'model': model,
                    'avg_score': avg_score,
                    'total_cost': total_cost,
                    'dataset_coverage': dataset_coverage,
                    'valid_datasets': len(valid_scores)
                })
        
        # Sort by average score descending
        model_summaries.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # Find best overall baseline
        best_baseline = model_summaries[0] if model_summaries else None
        
        # Calculate per-dataset comparison
        dataset_comparisons = []
        for dataset, perf in results['dataset_performance'].items():
            # Skip excluded datasets (should already be filtered from results)
            if dataset in self.config.excluded_datasets:
                continue
            
            # Filter by dataset type
            if dataset_filter == "ood" and dataset not in self.config.ood_datasets:
                continue
            elif dataset_filter == "non_ood" and dataset in self.config.ood_datasets:
                continue
                
            router_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
            
            # Find best baseline model for this dataset
            dataset_baselines = {}
            for model, scores in baseline_scores.items():
                if model in self.config.excluded_models:
                    continue
                    
                score = scores.get(dataset)
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    dataset_baselines[model] = score
            
            if dataset_baselines:
                best_model = max(dataset_baselines.items(), key=lambda x: x[1])
                best_baseline_score = best_model[1]
                improvement = router_accuracy - best_baseline_score
                
                dataset_comparisons.append({
                    'dataset': dataset,
                    'router_accuracy': router_accuracy,
                    'best_baseline_score': best_baseline_score,
                    'best_baseline_model': best_model[0],
                    'improvement': improvement
                })
        
        return {
            'model_summaries': model_summaries,
            'best_overall_baseline': best_baseline,
            'dataset_comparisons': dataset_comparisons
        }

    def _analyze_baseline_comparison(self, baseline_scores: Dict[str, Dict[str, float]], 
                                    results: Dict[str, Any], test_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze baseline comparison data for export, separated by OOD vs Non-OOD when applicable.
        
        Args:
            baseline_scores: Baseline scores from config/baseline.json
            results: Router evaluation results
            test_data: Test data with cost information
            
        Returns:
            Baseline analysis dictionary
        """
        if self.config.ood_datasets:
            # Calculate separate analyses for OOD and Non-OOD datasets
            non_ood_analysis = self._calculate_baseline_analysis(baseline_scores, results, test_data, "non_ood")
            ood_analysis = self._calculate_baseline_analysis(baseline_scores, results, test_data, "ood")
            overall_analysis = self._calculate_baseline_analysis(baseline_scores, results, test_data, "all")
            
            return {
                'non_ood_analysis': non_ood_analysis,
                'ood_analysis': ood_analysis,
                'overall_analysis': overall_analysis,
                # Backward compatibility - use overall analysis for legacy fields
                'model_summaries': overall_analysis['model_summaries'],
                'best_overall_baseline': overall_analysis['best_overall_baseline'],
                'dataset_comparisons': overall_analysis['dataset_comparisons']
            }
        else:
            # Backward compatibility - return overall analysis when no OOD datasets configured
            return self._calculate_baseline_analysis(baseline_scores, results, test_data, "all")

    def _calculate_baseline_metrics(self, baseline_scores: Dict[str, Dict[str, float]], 
                                   test_data: List[Dict], dataset_filter: str = "all") -> List[Tuple]:
        """
        Calculate baseline model metrics for specified dataset type.
        
        Args:
            baseline_scores: Baseline scores by model and dataset
            test_data: Test data with cost information
            dataset_filter: "all", "ood", or "non_ood"
            
        Returns:
            List of tuples (model, avg_score, total_cost, coverage)
        """
        # Calculate total costs from test data if available
        model_costs = {}
        if test_data:
            model_cost_data = defaultdict(list)
            for item in test_data:
                dataset_name = item.get('dataset', 'default')
                # Skip excluded datasets from cost calculation
                if dataset_name in self.config.excluded_datasets:
                    continue
                
                # Filter by dataset type
                if dataset_filter == "ood" and dataset_name not in self.config.ood_datasets:
                    continue
                elif dataset_filter == "non_ood" and dataset_name in self.config.ood_datasets:
                    continue
                    
                usages = item.get('usages')
                if usages:
                    for model_name, usage in usages.items():
                        if isinstance(usage, dict) and 'cost' in usage:
                            cost = usage['cost']
                            if isinstance(cost, (int, float)) and cost >= 0:
                                model_cost_data[model_name].append(cost)
            
            # Calculate total costs
            for model_name, costs in model_cost_data.items():
                if costs:
                    model_costs[model_name] = sum(costs)

        model_averages = []
        for model, scores in baseline_scores.items():
            # Skip excluded models
            if model in self.config.excluded_models:
                continue
                
            # Calculate average excluding null values and excluded datasets
            valid_scores = []
            total_datasets = 0
            for dataset, score in scores.items():
                if dataset in self.config.excluded_datasets:
                    continue
                
                # Filter by dataset type
                if dataset_filter == "ood" and dataset not in self.config.ood_datasets:
                    continue
                elif dataset_filter == "non_ood" and dataset in self.config.ood_datasets:
                    continue
                    
                total_datasets += 1
                if score is not None:
                    score = score / 100.0 if score > 1.0 else score
                    valid_scores.append(score)
            
            if valid_scores and total_datasets > 0:
                avg_score = sum(valid_scores) / len(valid_scores)
                coverage = f"{len(valid_scores)}/{total_datasets}"
                total_cost = model_costs.get(model, 0.0)
                model_averages.append((model, avg_score, total_cost, coverage))
        
        # Sort by average score (descending)
        model_averages.sort(key=lambda x: x[1], reverse=True)
        return model_averages

    def _print_baseline_summary(self, baseline_scores: Dict[str, Dict[str, float]], test_data: List[Dict] = None):
        """Print baseline model performance summary with averages and total costs, separated by OOD vs Non-OOD."""
        
        if self.config.ood_datasets:
            # Print Non-OOD baseline summary
            print(f"\nNon-OOD Baseline Model Performance Summary:")
            print(f"{'Model':<35} {'Average Score':<12} {'Total Cost':<12} {'Dataset Coverage'}")
            print("-" * 77)
            
            non_ood_averages = self._calculate_baseline_metrics(baseline_scores, test_data, "non_ood")
            
            for model, avg_score, total_cost, coverage in non_ood_averages:
                model_short = model.split('/')[-1]  # Show only model name without provider
                cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
                print(f"{model_short:<35} {avg_score:.4f}        {cost_str:<12} {coverage}")
            
            if non_ood_averages:
                best_model = non_ood_averages[0]
                best_total_cost = best_model[2]
                cost_info = f", total cost: ${best_total_cost:.4f}" if best_total_cost > 0 else ""
                print(f"\nBest Non-OOD Baseline: {best_model[0].split('/')[-1]} (avg: {best_model[1]:.4f}{cost_info})")
            
            # Print OOD baseline summary
            print(f"\nOOD Baseline Model Performance Summary:")
            print(f"{'Model':<35} {'Average Score':<12} {'Total Cost':<12} {'Dataset Coverage'}")
            print("-" * 77)
            
            ood_averages = self._calculate_baseline_metrics(baseline_scores, test_data, "ood")
            
            for model, avg_score, total_cost, coverage in ood_averages:
                model_short = model.split('/')[-1]  # Show only model name without provider
                cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
                print(f"{model_short:<35} {avg_score:.4f}        {cost_str:<12} {coverage}")
            
            if ood_averages:
                best_model = ood_averages[0]
                best_total_cost = best_model[2]
                cost_info = f", total cost: ${best_total_cost:.4f}" if best_total_cost > 0 else ""
                print(f"\nBest OOD Baseline: {best_model[0].split('/')[-1]} (avg: {best_model[1]:.4f}{cost_info})")
                
        else:
            # Print overall baseline summary (backward compatibility)
            print(f"\nBaseline Model Performance Summary:")
            print(f"{'Model':<35} {'Average Score':<12} {'Total Cost':<12} {'Dataset Coverage'}")
            print("-" * 77)
            
            all_averages = self._calculate_baseline_metrics(baseline_scores, test_data, "all")
            
            for model, avg_score, total_cost, coverage in all_averages:
                model_short = model.split('/')[-1]  # Show only model name without provider
                cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
                print(f"{model_short:<35} {avg_score:.4f}        {cost_str:<12} {coverage}")
            
            if all_averages:
                best_model = all_averages[0]
                best_total_cost = best_model[2]
                cost_info = f", total cost: ${best_total_cost:.4f}" if best_total_cost > 0 else ""
                print(f"\nBest Overall Baseline: {best_model[0].split('/')[-1]} (avg: {best_model[1]:.4f}{cost_info})")

    def print_evaluation_results(self, results: Dict[str, Any], test_data: List[Dict] = None):
        """Print detailed evaluation results with baseline comparison."""
        print(f"\n{'='*50}")
        print("JISI ROUTING EVALUATION RESULTS")
        print(f"{'='*50}")
        
        # Display excluded models/datasets if any
        if self.config.excluded_models:
            print(f"Excluded Models: {', '.join(self.config.excluded_models)}")
        if self.config.excluded_datasets:
            print(f"Excluded Datasets: {', '.join(self.config.excluded_datasets)}")
        if self.config.excluded_models or self.config.excluded_datasets:
            print()
        
        print(f"Overall Accuracy (Dataset-Avg): {results['accuracy']:.4f}")
        print(f"Overall Accuracy (Sample-Avg): {results.get('all_sample_avg', 0.0):.4f}")
        # Display OOD vs Non-OOD accuracy breakdown if OOD datasets are configured
        if self.config.ood_datasets:
            print(f"\nIn-Domain Accuracy (Dataset-Avg): {results['non_ood_accuracy']:.4f}")
            print(f"In-Domain Accuracy (Sample-Avg): {results.get('non_ood_sample_avg', 0.0):.4f}")
            print(f"OOD Accuracy (Dataset-Avg): {results['ood_accuracy']:.4f}")
            print(f"OOD Accuracy (Sample-Avg): {results.get('ood_sample_avg', 0.0):.4f}")
            print(f"OOD Datasets: {', '.join(self.config.ood_datasets)}")
        
        # Load baseline scores for comparison
        baseline_scores = self._load_baseline_scores()
        
        # Print baseline model averages if available
        if baseline_scores:
            self._print_baseline_summary(baseline_scores, test_data)
        
        # Group datasets by in-domain vs OOD
        in_domain_datasets = []
        ood_datasets_found = []

        for dataset in results['dataset_performance'].keys():
            if dataset in self.config.ood_datasets:
                ood_datasets_found.append(dataset)
            else:
                in_domain_datasets.append(dataset)

        in_domain_datasets.sort()
        ood_datasets_found.sort()

        # Print In-Domain datasets first
        if in_domain_datasets:
            print(f"\n{'='*70}")
            print(f"IN-DOMAIN DATASETS PERFORMANCE")
            print(f"{'='*70}")
            if baseline_scores:
                print(f"{'Dataset':<15} {'Router':<8} {'Baseline':<9} {'Best Model':<12} {'Improvement':<12} {'Cost':<10}")
                print("-" * 80)
            else:
                print(f"{'Dataset':<15} {'Accuracy':<10} {'Samples':<10} {'Cost':<10}")
                print("-" * 50)

            for dataset in in_domain_datasets:
                perf = results['dataset_performance'][dataset]
                router_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                dataset_cost = results.get('cost_analysis', {}).get('dataset_costs', {}).get(dataset, 0.0)

                if baseline_scores:
                    # Find best baseline model for this dataset (excluding excluded models)
                    dataset_baselines = {}
                    for model, scores in baseline_scores.items():
                        # Skip excluded models
                        if model in self.config.excluded_models:
                            continue

                        score = scores.get(dataset)
                        if score is not None:
                            # Convert percentage to decimal if needed
                            score = score / 100.0 if score > 1.0 else score
                            dataset_baselines[model] = score
                        else:
                            dataset_baselines[model] = 0.0

                    if dataset_baselines:
                        best_model = max(dataset_baselines.items(), key=lambda x: x[1])
                        best_baseline = best_model[1]
                        improvement = router_accuracy - best_baseline
                        improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.000"

                        print(f"{dataset:<15} {router_accuracy:.4f}    {best_baseline:.4f}     {best_model[0].split('/')[-1]:<12} {improvement_str:<12} ${dataset_cost:.4f}")
                    else:
                        print(f"{dataset:<15} {router_accuracy:.4f}    {'N/A':<9} {'N/A':<12} {'N/A':<12} ${dataset_cost:.4f}")
                else:
                    print(f"{dataset:<15} {router_accuracy:.4f}     ({perf['correct']}/{perf['total']})   ${dataset_cost:.4f}")

        # Print OOD datasets
        if ood_datasets_found:
            print(f"\n{'='*70}")
            print(f"OUT-OF-DISTRIBUTION (OOD) DATASETS PERFORMANCE")
            print(f"{'='*70}")
            if baseline_scores:
                print(f"{'Dataset':<15} {'Router':<8} {'Baseline':<9} {'Best Model':<12} {'Improvement':<12} {'Cost':<10}")
                print("-" * 80)
            else:
                print(f"{'Dataset':<15} {'Accuracy':<10} {'Samples':<10} {'Cost':<10}")
                print("-" * 50)

            for dataset in ood_datasets_found:
                perf = results['dataset_performance'][dataset]
                router_accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                dataset_cost = results.get('cost_analysis', {}).get('dataset_costs', {}).get(dataset, 0.0)

                if baseline_scores:
                    # Find best baseline model for this dataset (excluding excluded models)
                    dataset_baselines = {}
                    for model, scores in baseline_scores.items():
                        # Skip excluded models
                        if model in self.config.excluded_models:
                            continue

                        score = scores.get(dataset)
                        if score is not None:
                            # Convert percentage to decimal if needed
                            score = score / 100.0 if score > 1.0 else score
                            dataset_baselines[model] = score
                        else:
                            dataset_baselines[model] = 0.0

                    if dataset_baselines:
                        best_model = max(dataset_baselines.items(), key=lambda x: x[1])
                        best_baseline = best_model[1]
                        improvement = router_accuracy - best_baseline
                        improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.000"

                        print(f"{dataset:<15} {router_accuracy:.4f}    {best_baseline:.4f}     {best_model[0].split('/')[-1]:<12} {improvement_str:<12} ${dataset_cost:.4f}")
                    else:
                        print(f"{dataset:<15} {router_accuracy:.4f}    {'N/A':<9} {'N/A':<12} {'N/A':<12} ${dataset_cost:.4f}")
                else:
                    print(f"{dataset:<15} {router_accuracy:.4f}     ({perf['correct']}/{perf['total']})   ${dataset_cost:.4f}")
        
        print(f"\nModel Selection Frequency:")
        total_selections = sum(results['model_selection_stats'].values())
        for model, count in results['model_selection_stats'].most_common():
            percentage = count / total_selections * 100 if total_selections > 0 else 0
            print(f"  {model:30}: {count:4d} ({percentage:5.1f}%)")
        
        # Print per-dataset model selection and performance
        if 'dataset_model_selection' in results and results['dataset_model_selection']:
            print(f"\nPer-Dataset Model Selection & Performance:")
            print(f"Dataset         Model                    Selection   Accuracy")
            print(f"-" * 58)
            
            for dataset in sorted(results['dataset_model_selection'].keys()):
                dataset_selections = results['dataset_model_selection'][dataset]
                dataset_total = sum(dataset_selections.values())
                
                first_model = True
                for model, count in dataset_selections.most_common():
                    percentage = count / dataset_total * 100 if dataset_total > 0 else 0
                    
                    # Calculate accuracy for this model on this dataset
                    accuracy_scores = results['dataset_model_accuracy'][dataset][model]
                    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
                    
                    dataset_label = dataset if first_model else ""
                    print(f"{dataset_label:<15} {model:<24} {count:3d} ({percentage:4.1f}%)  {avg_accuracy:.4f}")
                    first_model = False
        
        # Print overall model statistics
        if 'model_selection_stats' in results and results['model_selection_stats']:
            print(f"\nOverall Model Statistics:")
            print(f"Model                           Total Selection   Selection Rate   Avg Accuracy")
            print(f"-" * 77)
            
            for model, count in results['model_selection_stats'].most_common():
                percentage = count / total_selections * 100 if total_selections > 0 else 0
                
                # Calculate overall accuracy for this model across all datasets
                all_scores = []
                for dataset in results['dataset_model_accuracy']:
                    if model in results['dataset_model_accuracy'][dataset]:
                        all_scores.extend(results['dataset_model_accuracy'][dataset][model])
                
                avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
                print(f"{model:<31} {count:<13} {percentage:5.1f}%          {avg_accuracy:.4f}")
        
        # Print cost analysis
        if 'cost_analysis' in results:
            cost_analysis = results['cost_analysis']
            
            print(f"\n{'='*50}")
            print("COST-EFFICIENCY ANALYSIS")
            print(f"{'='*50}")
            
            print(f"Total Cost: ${cost_analysis['total_cost']:.4f}")
            print(f"Average Cost per Query: ${cost_analysis['avg_cost_per_query']:.4f}")
            print(f"Cost per Correct Prediction: ${cost_analysis['cost_per_correct_prediction']:.4f}")
            print(f"Cost Efficiency (Accuracy/Cost): {cost_analysis['cost_efficiency']:.4f}")

            # Show OOD vs non-OOD cost breakdown if OOD datasets configured
            if self.config.ood_datasets:
                print(f"\nOOD vs In-Domain Cost Breakdown:")
                print(f"  In-Domain Total Cost: ${cost_analysis.get('non_ood_total_cost', 0.0):.4f}")
                print(f"  OOD Total Cost: ${cost_analysis.get('ood_total_cost', 0.0):.4f}")

            print(f"\nCost by Dataset:")
            for dataset, cost in cost_analysis['dataset_costs'].items():
                avg_cost = cost / results['dataset_performance'][dataset]['total']
                print(f"  {dataset:15}: ${cost:.4f} (${avg_cost:.4f} per query)")
            
            print(f"\nModel Cost Distribution:")
            total_model_cost = sum(cost_analysis['model_costs'].values())
            for model, cost in sorted(cost_analysis['model_costs'].items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (cost / total_model_cost) * 100 if total_model_cost > 0 else 0
                print(f"  {model:30}: ${cost:8.4f} ({percentage:5.1f}%)")

    def clean_think(self, references, max_tokens=0):
        new_references = []
        for r in references:
            r = r.split('</think>')[-1]
            r = r.split('## Final Response')[-1]
            r = r.split('</thought>')[-1]
            new_references.append(r)
        return new_references

    def clean_think_token(self, reference):
        think_tokens = ['<think>', '</think>', '</thought>', '<thought>']
        for think_token in think_tokens:
            reference = reference.replace(think_token, '')
        return reference

    def run_routing(self):
        """Run the complete JiSi retrieval-routing process."""
        print("Starting JiSi")
        # Load and split data
        train_data, test_data = self.load_and_split_data()
        self.train_data = train_data
        self.test_data = test_data
        # Build retrieval embedding bank
        self.embedding_bank = self.build_embedding_bank(train_data)
        # Evaluate routing
        results = self.evaluate_routing(test_data)
        
        if self.config.mode == 'router':
            self.print_evaluation_results(results, test_data)
        elif self.config.mode == 'aggregator':
            output_path = results.get('aggregation_output_path')
            if output_path:
                print(f"Aggregation output: {output_path}")
        
        return results


def main():
    """
    Main entry point for JiSi.
    
    Parses command line arguments, initializes configuration, and runs the routing evaluation.
    Supports loading configuration from environment variables and command line overrides.
    """
    parser = argparse.ArgumentParser(
        description='JiSi - collective routing and aggregation for LLM collaboration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with train/test data
    export EMBEDDING_API_KEY="your-key"
    python -m baselines.JiSi.run_jisi --config baselines/JiSi/config/jisi/main.example.json

    # CLI-only router run
    python -m baselines.JiSi.run_jisi \
        --train-data data/jisi/seed42_split0.7/train.jsonl \
        --test-data data/jisi/seed42_split0.7/test.jsonl \
        --baseline-scores data/jisi/seed42_split0.7/baseline_scores.json

    # Use configuration file
    python -m baselines.JiSi.run_jisi --config path/to/jisi_config.json
        """
    )
    
    parser.add_argument('--train-data', type=str,
                       help='Path to train JSONL file')
    parser.add_argument('--test-data', type=str,
                       help='Path to test JSONL file')
    parser.add_argument('--baseline-scores', type=str,
                       help='Path to baseline_scores.json')
    parser.add_argument('--output', type=str, 
                       help='Output file path for results (JSON format)')
    parser.add_argument('--config', type=str, 
                       help='Configuration file path (JSON format)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_router', type=int, default=1, 
                       help='Number of top models to select (default: 1)')
    parser.add_argument('--max_tokens', type=int, default=7500,
                       help='Maximum tokens per query (default: 7500)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--excluded_models', type=str,
                       help='Comma-separated list of models to exclude from routing')
    parser.add_argument('--excluded_datasets', type=str,
                       help='Comma-separated list of datasets to exclude from evaluation')
    parser.add_argument('--ood_datasets', type=str,
                       help='Comma-separated list of out-of-distribution datasets for separate evaluation')
    parser.add_argument('--dataset_exclusion_mode', choices=['soft', 'hard'], default='hard',
                       help='Dataset exclusion mode: soft (exclude from eval only) or hard (exclude completely)')
    # The unique parameters of JiSi
    parser.add_argument('--mode', type=str, choices=['router', 'aggregator'],
                        default='router',
                        help='JiSi mode')
    parser.add_argument('--embedding-config', type=str,
                        help='Path to embedding_config YAML')
    parser.add_argument('--api-config', type=str,
                        help='Path to OpenAI-compatible model API config JSON')
    parser.add_argument('--cache-config', type=str,
                        help='Path to optional MySQL API cache YAML')
    parser.add_argument('--embedding-model', type=str,
                        help='Embedding model name')
    parser.add_argument('--embedding-base-url', type=str,
                        help='OpenAI-compatible embedding endpoint base URL')
    parser.add_argument('--embedding-api-key', type=str,
                        help='Embedding API key or environment variable name')
    parser.add_argument('--deepseek-tokenizer-path', type=str,
                        help='Tokenizer path/name used for aggregation length accounting')
    parser.add_argument('--result-dir', type=str,
                        help='Directory for JiSi aggregation outputs')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # try:
    # Create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = JiSiConfig.from_file(args.config)
    else:
        if not args.train_data or not args.test_data or not args.baseline_scores:
            parser.error("--train-data, --test-data, and --baseline-scores are required when not using --config")

        logger.info("Loading configuration from environment and arguments")
        # Parse excluded models and datasets from command line
        excluded_models = []
        if args.excluded_models:
            excluded_models = [model.strip() for model in args.excluded_models.split(",") if model.strip()]

        excluded_datasets = []
        if args.excluded_datasets:
            excluded_datasets = [dataset.strip() for dataset in args.excluded_datasets.split(",") if dataset.strip()]

        ood_datasets = []
        if args.ood_datasets:
            ood_datasets = [dataset.strip() for dataset in args.ood_datasets.split(",") if dataset.strip()]

        config = JiSiConfig.from_env(
            train_data_path=args.train_data,
            test_data_path=args.test_data,
            baseline_scores_path=args.baseline_scores,
            seed=args.seed,
            max_router=args.max_router,
            max_tokens=args.max_tokens,
            excluded_models=excluded_models,
            excluded_datasets=excluded_datasets,
            ood_datasets=ood_datasets,
            dataset_exclusion_mode=args.dataset_exclusion_mode,
            mode=args.mode,
            embedding_config_path=args.embedding_config,
            api_config_path=args.api_config,
            cache_config=args.cache_config,
            embedding_model=args.embedding_model,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
            deepseek_tokenizer_path=args.deepseek_tokenizer_path,
            result_dir=args.result_dir
        )

    logger.info(f"Configuration: {config.to_dict()}")

    # Initialize and run router
    logger.info("Initializing JiSi")
    router = JiSi(config)


    logger.info("Starting routing evaluation")
    results = router.run_routing()

    # Save results if output path specified
    if args.output:
        from datetime import datetime
        results_serializable = {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'results': {
                'accuracy': results.get('accuracy', 0.0),
                'correct_routes': results.get('correct_routes', 0),
                'total_queries': results.get('total_queries', 0),
                'dataset_performance': dict(results.get('dataset_performance', {})),
                'model_selection_stats': dict(results.get('model_selection_stats', {})),
                'aggregation_output_path': results.get('aggregation_output_path'),
                # Add OOD/non-OOD metrics
                'ood_accuracy': results.get('ood_accuracy', 0.0),
                'non_ood_accuracy': results.get('non_ood_accuracy', 0.0),
                'ood_sample_avg': results.get('ood_sample_avg', 0.0),
                'non_ood_sample_avg': results.get('non_ood_sample_avg', 0.0),
                'all_sample_avg': results.get('all_sample_avg', 0.0),
                # Add cost analysis
                'cost_analysis': results.get('cost_analysis', {}),
                # Add baseline analysis if available
                'baseline_analysis': results.get('baseline_analysis', {})
            }
        }

        # Use the output path specified by user (support both absolute and relative paths)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")
        print(f"\nResults saved to: {output_path}")

    logger.info("Routing evaluation completed successfully")


if __name__ == "__main__":
    main()

