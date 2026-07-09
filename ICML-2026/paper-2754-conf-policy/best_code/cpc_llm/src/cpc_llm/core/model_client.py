import gc
import logging
import numpy as np
import os
import random
import s3fs
import tempfile
import time
import torch
import torch.distributed.checkpoint as dist_cp

from ..infrastructure.retry import cuda_retry
from ..test_functions.finetune_utils import (
    integer_list_length_or_none,
    maybe_log,
    remove_integer_padding_from_list,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple, Union


CUDA_ERROR = getattr(torch.cuda, "CudaError", RuntimeError)


def wait_for_gpu_availability(
    device: Optional[str] = None,
    max_wait_time: int = 172800,  # 3600, # Maximum wait time in seconds (48 hour default)
    check_interval: int = 30,  # Check every 30 seconds
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Wait for GPU to become available before proceeding.

    Args:
        device: Device string ('cuda', 'cpu', etc.). If None or 'cpu', returns immediately.
        max_wait_time: Maximum time to wait in seconds (default: 3600 = 1 hour)
        check_interval: Time between checks in seconds (default: 5)
        logger: Optional logger for status messages

    Returns:
        True if GPU is available, False if max_wait_time exceeded
    """
    # If device is None or 'cpu', no need to wait
    if device is None or device == "cpu":
        return True

    # If device is 'gpu', convert to 'cuda'
    if device == "gpu":
        device = "cuda"

    # Only wait if device is 'cuda'
    if device != "cuda":
        return True

    start_time = time.time()
    attempts = 0

    while time.time() - start_time < max_wait_time:
        try:
            # Try to initialize CUDA and check if it's available
            if torch.cuda.is_available():
                # Try to allocate a small tensor to verify GPU is actually accessible
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    if logger:
                        logger.info(
                            f"GPU is now available after {attempts * check_interval} seconds"
                        )
                    return True
                except (CUDA_ERROR, RuntimeError) as e:
                    # GPU exists but is busy/unavailable
                    if logger:
                        logger.warning(
                            f"GPU detected but busy/unavailable (attempt {attempts + 1}): {e}. "
                            f"Waiting {check_interval} seconds before retry..."
                        )
            else:
                if logger:
                    logger.warning(
                        f"CUDA not available (attempt {attempts + 1}). "
                        f"Waiting {check_interval} seconds before retry..."
                    )
        except Exception as e:
            if logger:
                logger.warning(
                    f"Error checking GPU availability (attempt {attempts + 1}): {e}. "
                    f"Waiting {check_interval} seconds before retry..."
                )

        attempts += 1
        time.sleep(check_interval)

    # Max wait time exceeded
    if logger:
        logger.error(
            f"GPU did not become available within {max_wait_time} seconds. "
            f"Total attempts: {attempts}"
        )
    return False


class ModelClient:
    """HuggingFace causal LM wrapper for generation and likelihood computation.

    Handles CUDA initialization with retry/backoff, supports loading from
    local paths, S3 URIs, and sharded FSDP checkpoints.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        hf_model_name: str = None,  # The name of the model in the HF hub. Only necessary
        # if loading from a sharded FSDP checkpoint.
        logger: logging.Logger = None,
        temperature: float = 1.0,
        max_generate_length: int = 500,
        device: Optional[str] = None,
        **model_init_args,
    ):
        self.model_name_or_path = model_name_or_path
        self.hf_model_name = hf_model_name
        self.logger = logger

        self.max_generate_length = max_generate_length
        self.temperature = temperature

        self.device = device
        self.model_init_args = model_init_args

        # Determine the actual device to use
        actual_device = device
        if actual_device is None:
            # Try CUDA first, wait for it, then fall back to CPU if needed
            # Don't check torch.cuda.is_available() here as GPU might be temporarily busy
            actual_device = "cuda"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA
        if actual_device == "cuda":
            # Add a small random delay to stagger CUDA initialization across processes
            # This helps avoid race conditions when multiple processes try to set CUDA devices simultaneously
            delay = random.uniform(0.1, 0.5)  # Random delay between 0.1-0.5 seconds
            time.sleep(delay)

            maybe_log(
                self.logger,
                "Waiting for GPU to become available...",
                level="info",
            )
            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800,  # Wait up to 48 hours
                check_interval=30,  # Check every 30 seconds
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Falling back to CPU.",
                    level="warning",
                )
                actual_device = "cpu"
                self.device = "cpu"

        if model_name_or_path.startswith("s3://"):
            self._load_from_s3(model_name_or_path)
        elif (
            os.path.exists(model_name_or_path)
            and not os.path.exists(f"{model_name_or_path}/config.json")
            and os.path.exists(f"{model_name_or_path}/pytorch_model_fsdp_0")
        ):
            maybe_log(
                self.logger,
                f"{model_name_or_path} is a sharded FSDP model. Attempting to consolidate and convert.",
                level="info",
            )
            self.model, self.config, self.tokenizer = self._convert_checkpoint(
                hf_model_name=hf_model_name,
                fsdp_model_path=f"{model_name_or_path}/pytorch_model_fsdp_0",
                output_path=model_name_or_path,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            extra_model_kwargs = {**self.model_init_args}
            if "gg-hf/gemma" in self.model_name_or_path:
                extra_model_kwargs["torch_dtype"] = torch.bfloat16
                # Only use device_map="cuda" if we're actually using CUDA
                if actual_device == "cuda":
                    extra_model_kwargs["device_map"] = "cuda"

            self.model = cuda_retry(
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    **extra_model_kwargs,
                ),
                stagger=False,
                on_retry=lambda: (
                    wait_for_gpu_availability(
                        device=actual_device,
                        max_wait_time=172800,
                        check_interval=30,
                        logger=self.logger,
                    )
                    if actual_device == "cuda"
                    else None
                ),
                logger=self.logger,
                operation="model loading",
            )

            if self.device is not None:
                self.model = self.model.to(self.device)
            elif actual_device is not None and actual_device != "cpu":
                # If device was originally None but we determined it should be cuda, move to cuda
                self.model = self.model.to(actual_device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(
                    f"Tokenizer for {self.model_name_or_path} does not have a PAD token."
                )
        if hasattr(self.config, "max_position_embeddings"):
            self.max_length = self.config.max_position_embeddings
        elif hasattr(self.config, "max_sequence_length"):
            self.max_length = self.config.max_sequence_length
        elif hasattr(self.config, "n_positions"):
            self.max_length = self.config.n_positions
        else:
            raise ValueError(
                f"Could not find max_position_embeddings, n_positions, or max_sequence_length in model config for {self.model_name_or_path}"
            )

    def _load_from_s3(self, s3_uri: str):
        # Determine the actual device to use
        actual_device = self.device
        if actual_device is None:
            # Try CUDA first, wait for it, then fall back to CPU if needed
            # Don't check torch.cuda.is_available() here as GPU might be temporarily busy
            actual_device = "cuda"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA
        if actual_device == "cuda":
            # Add a small random delay to stagger CUDA initialization across processes
            delay = random.uniform(0.1, 0.5)  # Random delay between 0.1-0.5 seconds
            time.sleep(delay)

            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800,  # Wait up to 48 hours
                check_interval=30,  # Check every 30 seconds
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Falling back to CPU.",
                    level="warning",
                )
                actual_device = "cpu"
                self.device = "cpu"

        self.s3 = s3fs.S3FileSystem()
        with tempfile.TemporaryDirectory() as td:
            maybe_log(self.logger, f"Downloading model from {s3_uri} into {td}.")
            if not td.endswith("/"):
                td += "/"
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            s3_uri += (
                "*"  # will copy only the files in the directory, not any nested ones
            )
            self.s3.get(s3_uri, td)
            self.config = AutoConfig.from_pretrained(td)
            self.model_init_args["trust_remote_code"] = True

            self.model = cuda_retry(
                lambda: AutoModelForCausalLM.from_pretrained(
                    td, **self.model_init_args
                ),
                stagger=False,
                on_retry=lambda: (
                    wait_for_gpu_availability(
                        device=actual_device,
                        max_wait_time=172800,
                        check_interval=30,
                        logger=self.logger,
                    )
                    if actual_device == "cuda"
                    else None
                ),
                logger=self.logger,
                operation="S3 model loading",
            )

            if self.device is not None:
                self.model = self.model.to(self.device)
            elif actual_device is not None and actual_device != "cpu":
                self.model = self.model.to(actual_device)
            self.tokenizer = AutoTokenizer.from_pretrained(td)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _convert_checkpoint(
        self, hf_model_name: str, fsdp_model_path: str, output_path: str
    ):
        """
        hf_model_name: Name of model in HF Hub, e.g. "gpt2".
        fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_fsdp_x`
        output_path: output path to save the converted checkpoint
        """
        # Determine the actual device to use
        actual_device = self.device
        if actual_device is None:
            # Try CUDA first, wait for it, then fall back to CPU if needed
            # Don't check torch.cuda.is_available() here as GPU might be temporarily busy
            # Note: Checkpoint conversion requires GPU, so this will raise an error if GPU unavailable
            actual_device = "cuda"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA (required for checkpoint conversion)
        if actual_device == "cuda":
            # Add a small random delay to stagger CUDA initialization across processes
            delay = random.uniform(0.1, 0.5)  # Random delay between 0.1-0.5 seconds
            time.sleep(delay)

            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800,  # 3600,  # Wait up to 48 hours
                check_interval=30,
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Checkpoint conversion requires GPU.",
                    level="error",
                )
                raise RuntimeError(
                    "GPU required for checkpoint conversion but not available"
                )

        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

        model = cuda_retry(
            lambda: AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, **self.model_init_args
            ).cuda(),
            stagger=False,
            on_retry=lambda: wait_for_gpu_availability(
                device="cuda",
                max_wait_time=172800,
                check_interval=30,
                logger=self.logger,
            ),
            logger=self.logger,
            operation="checkpoint conversion",
        )

        model = self._load_sharded_model_single_gpu(model, fsdp_model_path)
        model.save_pretrained(output_path, max_shard_size="10GB")
        tokenizer.save_pretrained(output_path)
        return model, config, tokenizer

    def _load_sharded_model_single_gpu(self, model, model_path):
        state_dict = {"model": model.state_dict()}

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(model_path),
            no_dist=True,
        )

        result = model.load_state_dict(state_dict["model"])
        maybe_log(
            self.logger,
            f"Sharded state checkpoint loaded from {model_path}. Result: {result}",
            level="info",
        )
        return model

    def _apply_chat_template(self, msgs) -> str:
        if self.tokenizer.chat_template is not None:
            chat_msgs = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                f"Chat template does not exist for model {self.model_name_or_path}."
            )
        return chat_msgs

    def _tokenize_batch(
        self,
        text_batch: Union[List[List[Dict[str, str]]], List[str]],
        max_generate_length: Optional[int] = None,
    ) -> torch.tensor:
        """Tokenize either a batch of messages or a single text string."""
        if not text_batch:
            return None
        if max_generate_length is not None:
            max_gen_length = max_generate_length
        else:
            max_gen_length = self.max_generate_length
        max_context_length = self.max_length - max_gen_length
        if isinstance(text_batch[0], list):
            # text batch is list of messages, so apply chat template
            texts_to_tokenize = [self._apply_chat_template(text) for text in text_batch]
        else:
            texts_to_tokenize = text_batch
        tokenized = self.tokenizer(
            texts_to_tokenize,
            truncation=True,
            padding="longest",
            max_length=max_context_length,
            return_tensors="pt",
        )
        return tokenized

    def _chat_hf_model(
        self, msgs: Union[List[Dict[str, str]], str], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion for either a series of messages or a single text prompt.
        """
        tokenized = self._tokenize_batch([msgs]).input_ids
        if self.device is not None:
            tokenized = tokenized.to(self.device)
        outputs = self.model.generate(tokenized, **kwargs)
        return {"output_strs": self.tokenizer.batch_decode(outputs)}

    def chat_single_turn(self, msgs: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate a completion from a list of chat messages.

        Args:
            msgs: List of message dicts with "role" and "content" keys.
            **kwargs: Additional arguments passed to ``model.generate()``.

        Returns:
            Dict with "output_strs" containing the decoded output sequences.
        """
        return self._chat_hf_model(msgs, **kwargs)

    def chat_raw_logits(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        """Alternative API for getting all logits."""
        return self.model(input_ids, attention_mask=attention_mask).logits

    def chat_single_turn_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate a completion from a raw text prompt.

        Wraps the text in a user message and delegates to ``_chat_hf_model``.

        Args:
            text: Raw text prompt.
            **kwargs: Additional arguments passed to ``model.generate()``.

        Returns:
            Dict with "output_strs" containing the decoded output sequences.
        """
        msgs = [{"role": "user", "content": text}]
        return self._chat_hf_model(msgs, **kwargs)

    @torch.no_grad()
    def generate_texts_batched(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        return_likelihoods=False,
        subsample_seeds=False,
        logger: logging.Logger = None,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]]:
        """
        Runs batched generation. If return_likelihoods=True, returns a tuple containing
        the output sequences, the output tokens, and the output token log-likelihoods.
        Otherwise, returns only the output sequences.
        """
        output_strs = []
        all_output_token_ids = []
        all_token_logps = []
        # When returning likelihoods, output_scores stores full vocab logits
        # for every generated token.  Scale down the prompt batch size so the
        # effective batch (prompts * num_return_sequences) stays manageable.
        gen_cfg = kwargs.get("generation_config", None)
        num_return = kwargs.get("num_return_sequences", None)
        if num_return is None and gen_cfg is not None:
            num_return = getattr(gen_cfg, "num_return_sequences", 1) or 1
        num_return = num_return or 1
        if return_likelihoods and batch_size * num_return > 32:
            batch_size = max(1, 32 // num_return)
        for batch_start_idx in tqdm(
            range(0, len(input_texts), batch_size), desc="Batched generation..."
        ):
            if subsample_seeds:
                ## In this condition: Subsample input seed seqs uniformly with replacement
                ## (ensures that each output is sampled from the mixture of the prompt-conditional distributions)
                idx_input_texts = np.random.choice(
                    range(len(input_texts)), size=batch_size, replace=True
                )
                batch_texts = [input_texts[idx] for idx in idx_input_texts]
            else:
                ## In this condition: Move through all the input seeds in minibatches
                batch_end_idx = batch_start_idx + batch_size
                batch_texts = input_texts[batch_start_idx:batch_end_idx]

            input_ids = self._tokenize_batch(batch_texts).input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            gen_args = {**kwargs}
            if return_likelihoods:
                gen_args["return_dict_in_generate"] = True
                gen_args["output_scores"] = True
            outputs = self.model.generate(
                input_ids, **gen_args, tokenizer=self.tokenizer
            )
            output_token_ids = outputs.sequences if return_likelihoods else outputs
            # remove the input from each output
            completion_ids = output_token_ids[:, input_ids.shape[-1] :]
            decoded_outputs = self.tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output_strs.extend(decoded_outputs)
            if return_likelihoods:
                token_logps = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                for oti in completion_ids:
                    all_output_token_ids.append(oti)
                for o_logps in token_logps:
                    all_token_logps.append(o_logps)
        if return_likelihoods:
            return (output_strs, all_output_token_ids, all_token_logps)
        return output_strs

    @torch.no_grad()
    def compute_likelihoods(
        self,
        inputs: List[str],
        targets: List[str],
        batch_size: int = 16,
        device: str = "cuda",
        # add_start_token: bool = True,
        logger: logging.Logger = None,
    ) -> List[float]:
        """Compute length-normalized log-likelihoods of targets given inputs.

        Concatenates each (input, target) pair, runs a forward pass, and
        extracts the mean log-probability over target tokens.

        Args:
            inputs: List of input prompt strings.
            targets: List of target sequences to evaluate.
            batch_size: Number of (input, target) pairs per forward pass.
            device: Device for computation ("cuda", "cpu", or "gpu").
            logger: Optional logger (currently unused).

        Returns:
            List of length-normalized log-likelihoods, one per target.
        """
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be in ['gpu', 'cpu', 'cuda']."
            if device == "gpu":
                device = "cuda"
        else:
            device = self.device

        all_likelihoods = []
        for start_index in tqdm(
            range(0, len(inputs), batch_size), desc="Computing log likelihood..."
        ):
            end_index = min(start_index + batch_size, len(inputs))
            batch = [
                f"{input}{target}"
                for input, target in zip(
                    inputs[start_index:end_index], targets[start_index:end_index]
                )
            ]
            tokenized = self._tokenize_batch(batch, max_generate_length=0).to(
                self.device
            )
            outputs = self.model(**tokenized)
            scores = outputs.logits
            # probs = torch.exp(torch.nn.functional.log_softmax(scores, dim=-1))

            ### NOTE: Think this should be log_softmax (without exp as above)
            probs = torch.nn.functional.log_softmax(scores, dim=-1)  ###
            # logger.info(f"probs shape : {probs.shape}")
            # get probs and mask out both padding and input tokens
            inputs_tokenized = self._tokenize_batch(inputs[start_index:end_index])
            input_seq_lens = inputs_tokenized.attention_mask.sum(-1)
            input_tokens_mask = torch.LongTensor(
                [
                    [0 for _ in range(input_seq_lens[i].item())]
                    + [
                        1
                        for _ in range(
                            tokenized.input_ids.shape[1] - input_seq_lens[i].item()
                        )
                    ]
                    for i in range(tokenized.input_ids.shape[0])
                ]
            ).to(self.device)
            indexes = tokenized.input_ids[:, 1:]
            next_token_probs = (
                torch.gather(probs, -1, indexes.unsqueeze(-1)).squeeze(-1)
                * tokenized.attention_mask[:, 1:]
                * input_tokens_mask[:, 1:]
            )

            next_token_probs = next_token_probs.cpu().numpy()

            targets_seq_lens = (
                (tokenized.attention_mask.sum(-1) - input_seq_lens).cpu().numpy()
            )
            avg_likelihoods = list(next_token_probs.sum(-1) / targets_seq_lens)
            # logger.info(f"avg_likelihoods shape : {np.shape(avg_likelihoods)}")
            # logger.info(f"seq likelihood : {list(next_token_probs.prod(-1) / targets_seq_lens)}")

            all_likelihoods.extend(avg_likelihoods)
            # logger.info(f"all_likelihoods : {all_likelihoods}")

        return all_likelihoods

    @staticmethod
    def _expand_kv_cache(
        past_key_values,
        batch_size: int,
    ):
        """Expand a batch-1 KV cache to the target batch size.

        Always returns a DynamicCache so the result is compatible with
        modern transformers model forward passes.

        Args:
            past_key_values: KV cache returned by a model forward pass
                (DynamicCache or tuple-of-tuples).
            batch_size: Target batch dimension for the expanded cache.

        Returns:
            DynamicCache with batch dimension expanded.
        """
        from transformers.cache_utils import DynamicCache

        # Extract (key, value) pairs per layer from either format
        if hasattr(past_key_values, "key_cache"):
            kv_pairs = zip(past_key_values.key_cache, past_key_values.value_cache)
        else:
            kv_pairs = ((layer[0], layer[1]) for layer in past_key_values)

        expanded = DynamicCache()
        for k, v in kv_pairs:
            expanded.update(
                k.expand(batch_size, -1, -1, -1).contiguous(),
                v.expand(batch_size, -1, -1, -1).contiguous(),
                len(expanded),
            )
        return expanded

    # @staticmethod
    # def _filter_negative_one_fillers(sequence_str: str) -> str:
    #     """Remove negative integers from a string representation of a list.
        
    #     Args:
    #         sequence_str: String like "[1, 3, 2, -1, -1]"
            
    #     Returns:
    #         Filtered string like "[1, 3, 2]"
    #     """
    #     try:
    #         # Parse the string as a list of integers
    #         sequence_list = eval(sequence_str)
    #         if not isinstance(sequence_list, list):
    #             return sequence_str
            
    #         # Filter out negative integers
    #         filtered_list = [x for x in sequence_list if isinstance(x, (int, float)) and x == -1]
            
    #         # Convert back to string representation
    #         return str(filtered_list)
    #     except Exception:
    #         # If parsing fails, return the original string
    #         return sequence_str

    @torch.no_grad()
    def compute_likelihoods_avg(
        self,
        inputs: List[str],
        targets: List[str],
        batch_size: int = 10,
        logger: logging.Logger | None = None,
        test_fn_dim: Optional[int] = None,
    ) -> List[float]:
        """Compute likelihoods of target sequences, averaged over all input seeds.

        Uses KV caching: each input prefix is encoded once and the cached
        key/value states are reused across all target batches.

        Args:
            inputs: List of input prompt strings (seeds).
            targets: List of target sequences to evaluate.
            batch_size: Number of targets to process per forward pass.
            logger: Logger instance for status messages.
            test_fn_dim: When set, multiply likelihood by p(EOS | context) for
                targets whose integer list (after padding removal) parses and
                has length strictly less than this value.

        Returns:
            List of average likelihoods, one per target. Each value is
            mean_i(exp(sum_t(log p(target_t | input_i, target_<t)) [+ log p(EOS|·)])),
            where the EOS term is included only for short parseable integer lists
            when test_fn_dim is provided.
        """
        if logger is not None:
            logger.info(f"temperature : {self.temperature}")

        include_eos: Optional[List[bool]] = None
        eos_token_id: Optional[int] = None
        if test_fn_dim is not None:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.model_name_or_path} has no eos_token_id; "
                    "cannot score EOS likelihood."
                )
            include_eos = []
            for t in targets:
                t_filtered = remove_integer_padding_from_list(t)
                n = integer_list_length_or_none(t_filtered)
                include_eos.append(n is not None and n < test_fn_dim)

        all_likelihoods = torch.zeros(len(targets), dtype=torch.float64)

        # Use right-padding for targets so position IDs are correct with KV cache
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        # logger.info(f"inputs : {inputs}")
        # logger.info(f"targets : {targets}")

        try:
            for input_text in tqdm(
                inputs, desc="Computing log likelihoods averaged over inputs..."
            ):
                # logger.info(f"input_text : {input_text}")
                # Filter negative integers from input
                filtered_input = remove_integer_padding_from_list(input_text)
                # --- encode input prefix once, cache KV states ---
                input_tok = self._tokenize_batch(
                    [filtered_input], max_generate_length=0
                ).to(self.device)
                input_out = self.model(**input_tok, use_cache=True)
                kv_cache = input_out.past_key_values
                # Logit at last input position predicts the first target token
                input_last_logit = input_out.logits[:, -1:, :]
                input_len = input_tok.attention_mask.sum().item()

                # --- process targets in batches using cached KV ---
                for t_start in range(0, len(targets), batch_size):
                    t_end = min(t_start + batch_size, len(targets))
                    target_batch = targets[t_start:t_end]
                    cur_batch_size = len(target_batch)
                    # logger.info(f"target_batch : {target_batch}")

                    # Filter negative integers from targets
                    # filtered_targets = [self._filter_negative_one_fillers(t) for t in target_batch]
                    filtered_targets = []
                    for t in target_batch:
                        t_filtered = remove_integer_padding_from_list(t)
                        filtered_targets.append(t_filtered)
                        if len(t_filtered) < len(t):
                            logger.info(f"t : {t}")
                            logger.info(f"t_filtered : {t_filtered}")

                    target_tok = self._tokenize_batch(
                        filtered_targets, max_generate_length=0
                    ).to(self.device)

                    expanded_kv = self._expand_kv_cache(kv_cache, cur_batch_size)

                    # Attention mask: all-1s for cached input + target attention mask
                    input_mask = torch.ones(
                        cur_batch_size,
                        input_len,
                        device=self.device,
                        dtype=target_tok.attention_mask.dtype,
                    )
                    full_mask = torch.cat(
                        [input_mask, target_tok.attention_mask], dim=1
                    )

                    target_out = self.model(
                        input_ids=target_tok.input_ids,
                        attention_mask=full_mask,
                        past_key_values=expanded_kv,
                        use_cache=False,
                    )

                    # Stitch logits: input's last logit + target's all-but-last
                    # input_last_logit predicts target[0]
                    # target_out.logits[:, i] predicts target[i+1]
                    combined_logits = torch.cat(
                        [
                            input_last_logit.expand(cur_batch_size, -1, -1),
                            target_out.logits[:, :-1, :],
                        ],
                        dim=1,
                    )  # (batch, target_len, vocab)

                    scores = combined_logits / self.temperature
                    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)

                    # Gather log-prob of each actual target token
                    target_log_probs = torch.gather(
                        log_probs, -1, target_tok.input_ids.unsqueeze(-1)
                    ).squeeze(-1)

                    # Zero out padding positions
                    target_log_probs = target_log_probs * target_tok.attention_mask

                    # Sum log-probs per target sequence
                    log_liks = target_log_probs.sum(-1)  # (batch,)

                    if include_eos is not None:
                        last_pos = target_tok.attention_mask.sum(dim=1) - 1
                        batch_idx = torch.arange(
                            cur_batch_size, device=self.device
                        )
                        last_logits = (
                            target_out.logits[batch_idx, last_pos, :]
                            / self.temperature
                        )
                        eos_log_prob = torch.nn.functional.log_softmax(
                            last_logits, dim=-1
                        )[:, eos_token_id]
                        eos_mask = torch.tensor(
                            include_eos[t_start:t_end],
                            device=self.device,
                            dtype=last_logits.dtype,
                        )
                        valid_last = (last_pos >= 0).to(last_logits.dtype)
                        log_liks = log_liks + eos_log_prob * eos_mask * valid_last

                    # Accumulate exp(log_lik) for averaging across inputs
                    all_likelihoods[t_start:t_end] += torch.exp(log_liks.cpu().double())
        finally:
            self.tokenizer.padding_side = orig_padding_side

        all_likelihoods /= len(inputs)
        return all_likelihoods.tolist()
