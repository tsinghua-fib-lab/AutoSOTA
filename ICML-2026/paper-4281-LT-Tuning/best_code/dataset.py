from functools import partial
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm import tqdm
from utils import apply_chat_template_if_needed, print_list_in_json

def is_distributed():
    try:
        import torch.distributed as dist
        return dist.is_initialized()
    except:
        return False

def get_rank():
    if is_distributed():
        import torch.distributed as dist
        return dist.get_rank()
    return 0
class ThinkingTokenStrategy(ABC):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        tokens_per_stage: Union[float, int] = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.thinking_token_id = thinking_token_id
        self.tokens_per_stage = tokens_per_stage
        self.insertion_prob = insertion_prob
        self.secondary_insertion_prob = secondary_insertion_prob
        self._seed = seed
        self.generator = random.Random(seed) if seed is not None else random.Random()
        self.requires_serial_processing = False

    @abstractmethod
    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        """Return candidate indices (0-based) for inserting thinking tokens."""

    def _build_rng(
        self, sample: Dict[str, Any], scheduled_stage: int = 1
    ) -> random.Random:

        if self._seed is None:
            return self.generator

        sample_idx = int(sample.get("idx", 0))
        offset = 23
        derived_seed = (
            1000003 * sample_idx
            + 9176 * scheduled_stage
            + offset
            + self._seed
        ) & 0xFFFFFFFF
        return random.Random(derived_seed)

    def _select_indices(
        self,
        candidates: List[int],
        scheduled_stage: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[int]:

        if scheduled_stage <= 0 or not candidates:
            return []
        if self.tokens_per_stage < 1:
            target_n = int(len(candidates) * self.tokens_per_stage)
        else:
            target_n = int(self.tokens_per_stage * scheduled_stage)
        if target_n <= 0:
            return []

        target_n = min(target_n, len(candidates))
        if target_n < len(candidates):
            return sorted(rng.sample(candidates, target_n))
        return sorted(candidates[:target_n])

    def apply(
        self,
        sample: Dict[str, Any],
        scheduled_stage: int = 1,
        candidate_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:

        response = sample.get("response_tokenized", [])
        input_tokens = sample.get("question_tokenized", [])
        if len(response) == 0:
            return response, []
        rng = self._build_rng(sample, scheduled_stage)
        # default to global positions.
        if candidate_indices is not None:
            candidates = candidate_indices
        else:
            candidates = self._candidate_indices(sample=sample)

        selected = self._select_indices(
            candidates, scheduled_stage, rng=rng
        )
        if not selected:
            return sample.get("full_tokenized", []), []

        inserted_positions: List[int] = []
        offset = 0
        updated_input_ids = list(sample.get("full_tokenized", []))
        for idx in selected:
            if rng.random() > self.insertion_prob:
                continue
            if idx < len(input_tokens):
                print("********************************Warning: Thinking token inserted in input tokens region.********************************")
            
            insert_at = idx + offset

            if insert_at < 0:
                insert_at = 0

            updated_input_ids.insert(insert_at, self.thinking_token_id)
            inserted_positions.append(insert_at)
            offset += 1

            if (
                self.secondary_insertion_prob > 0.0
                and rng.random() < self.secondary_insertion_prob
            ):
                insert_at += 1
                updated_input_ids.insert(insert_at, self.thinking_token_id)
                inserted_positions.append(insert_at)
                offset += 1

        return updated_input_ids, inserted_positions

class RandomThinkingStrategy(ThinkingTokenStrategy):

    def _candidate_indices(
        self, flat_steps: Sequence[int], sample: Dict[str, Any]
    ) -> List[int]:

        # Every token can be a potential insertion point (except the first one)
        return list(range(1, len(flat_steps)))


class ArithmeticThinkingStrategy(ThinkingTokenStrategy):

    OPERATORS = set("+-=*/%()")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        tokens_per_stage: int = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: Optional[int] = None,
        operator_regex: Optional[str] = None,
    ) -> None:

        super().__init__(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_insertion_prob,
            seed=seed,
        )

        self._operator_regex = re.compile(operator_regex) if operator_regex else None

    def _is_numeric_or_operator(self, text: str) -> bool:

        stripped = text.strip()
        if not stripped:
            return False

        if self._operator_regex and self._operator_regex.search(stripped):
            return True

        if any(char in self.OPERATORS for char in stripped):
            return True

        return stripped.replace(".", "", 1).isdigit()

    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        candidates: List[int] = []
        input_length = len(sample.get("question_tokenized", []))
        full_length = len(sample.get("full_tokenized", []))
        for idx in range(input_length, full_length):
            token_id = sample["full_tokenized"][idx]
            token_text = self.tokenizer.decode([token_id])
            if self._is_numeric_or_operator(token_text):
                candidates.append(idx)
            if token_text.strip() == "###":
                candidates.append(idx+1)
        return candidates


class ConfidenceThinkingStrategy(ThinkingTokenStrategy):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        model: Optional[torch.nn.Module],
        tokens_per_stage: Union[float, int] = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: Optional[int] = None,
        probability_threshold: float = 0.05,
        max_sequence_length: int = 2048,
    ) -> None:

        super().__init__(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_insertion_prob,
            seed=seed,
        )

        if model is None:
            raise ValueError(
                "ConfidenceThinkingStrategy requires a model instance for probability estimation."
            )

        self.model = model
        self.model_device = next(model.parameters()).device
        self.threshold = float(max(min(probability_threshold, 1.0), 0.0))
        self.max_sequence_length = max_sequence_length
        self.requires_serial_processing = True

    def batch_candidate_indices(
        self, samples: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """Process a batch of samples to find candidate indices for thinking token insertion."""
        
        if not samples:
            return []
        
        # Prepare batch data
        batch_messages = [
            [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["reasoning_chain"] + "\n### " + sample["answer"]}
            ] for sample in samples
        ]
        batch_input_ids = [
            self.tokenizer.encode(
                apply_chat_template_if_needed(self.tokenizer, message),
                add_special_tokens=False,
            )
            for message in batch_messages
        ]
        batch_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in batch_input_ids]
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="right"
        )
        batch_input = {
            "input_ids": padded_sequences.to(self.model_device),
            "attention_mask": (padded_sequences != self.tokenizer.pad_token_id).long().to(self.model_device),
        }
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**batch_input)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        # Process each sequence in the batch
        all_candidates = [[] for _ in range(len(samples))]
        for idx, sample in enumerate(samples):
            # Get actual sequence length from tokenized input
            user_msg = [{"role": "user", "content": sample["question"]}]
            user_text = apply_chat_template_if_needed(self.tokenizer, user_msg)
            question_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)
            question_len = len(question_tokens)
            # Compute probabilities of the ground-truth next tokens
            next_tokens = batch_input_ids[idx][1:].to(self.model_device)
            log_probs = F.log_softmax(logits[idx, : len(batch_input_ids[idx]), :], dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
            token_probs = token_log_probs.exp().tolist()
            for pos in range(question_len, len(batch_input_ids[idx])):
                if token_probs[pos - 1] < self.threshold:
                    all_candidates[idx].append(pos)

        return all_candidates

    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        """Find candidate indices for a single sample by calling batch processing with batch size 1."""
        results = self.batch_candidate_indices([sample])
        return results[0] if results else []


def build_thinking_strategy(
    configs: Any,
    tokenizer: PreTrainedTokenizerBase,
    thinking_token_id: int,
    model: Optional[torch.nn.Module] = None,
) -> ThinkingTokenStrategy:

    strategy_name = getattr(configs, "thinking_strategy", "random").lower()
    tokens_per_stage = getattr(
        configs,
        "tokens_per_stage",
        getattr(configs, "c_thought", 1),
    )
    insertion_prob = getattr(configs, "thinking_insertion_prob", 1.0)
    secondary_prob = getattr(configs, "thinking_secondary_insertion_prob", 0.0)
    seed = getattr(configs, "seed", None)

    def _resolve_stage_value(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    if strategy_name == "arithmetic":
        operator_regex = getattr(configs, "thinking_operator_regex", None)
        return ArithmeticThinkingStrategy(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_prob,
            seed=seed,
            operator_regex=operator_regex,
        )

    if strategy_name in {"confidence","reinforce", "policy"}:
        prob_threshold = _resolve_stage_value(
            getattr(configs, "reinforce_prob_threshold", 0.05), 0.05
        )
        max_eval_len = int(
            _resolve_stage_value(
                getattr(configs, "reinforce_max_eval_length", 2048), 2048
            )
        )
        return ConfidenceThinkingStrategy(
            tokenizer,
            thinking_token_id,
            model=model,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_prob,
            seed=seed,
            probability_threshold=prob_threshold,
            max_sequence_length=max_eval_len,
        )

    if strategy_name != "random":
        raise ValueError(f"Unsupported thinking strategy: {strategy_name}")

    return RandomThinkingStrategy(
        tokenizer,
        thinking_token_id,
        tokens_per_stage=tokens_per_stage,
        insertion_prob=insertion_prob,
        secondary_insertion_prob=secondary_prob,
        seed=seed,
    )


def get_dataset(path, dataset_name="gsm8k", max_size=1000000000):
    """Load raw dataset without tokenization.
    
    Tokenization will be done later in get_question_latent_dataset or 
    get_cot_latent_dataset to allow for different tokenization strategies.
    """
    if path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(path)][:max_size]
    else:
        data = json.load(open(path))[:max_size]
    def extract_reasoning_chain_from_answer(answer: str) -> str:
        """Extract reasoning chain from the answer field."""
        return answer.split("####")[0].strip()

    def extract_answer_from_answer_field(answer: str) -> str:
        """Extract final answer from the answer field."""
        return answer.split("####")[-1].strip()
    
    if dataset_name == "gsm8k":
        data = [
            {
                "idx": idx,
                "question": d["question"],
                "reasoning_chain": "\n".join(d["steps"]) if "steps" in d else extract_reasoning_chain_from_answer(d["answer"]),
                "answer": d["answer"] if d["answer"].startswith("###") else extract_answer_from_answer_field(d["answer"]),
            } 
             for idx, d in enumerate(data)
        ]
    else:
        raise ValueError(f"Unsupported Dataset Now: {dataset_name}")
    
    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})
    
    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    thinking_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this
        E.g.,
        
        xxxxxxxxxx<thinking><thinking>xxxxx--
        -----xxxxx<thinking>xxxxxxxx-------
        ---xxxxxxx<thinking><thinking>xxxxxxx


        ("x" is word token, "-" is pad token)
        """
        if self.thinking_id is not None:
            earliest_thinking = [
                feature["input_ids"].index(self.thinking_id)
                for feature in features
                if self.thinking_id in feature["input_ids"]
            ]

            if (
                len(earliest_thinking) > 0
            ):  # if there are continuous thoughts in the sequence
                latest_earliest_thinking = max(earliest_thinking)
                for feature in features:
                    if self.thinking_id in feature["input_ids"]:
                        n_tok_pad = latest_earliest_thinking - feature["input_ids"].index(
                            self.thinking_id
                        )
                    else:
                        n_tok_pad = 0
                    feature["position_ids"] = [0] * n_tok_pad + list(
                        range(len(feature["input_ids"]))
                    )
                    feature["input_ids"] = [
                        self.tokenizer.pad_token_id
                    ] * n_tok_pad + feature["input_ids"]
                    if "labels" in feature:
                        feature["labels"] = [
                            self.label_pad_token_id
                        ] * n_tok_pad + feature["labels"]
                    feature["attention_mask"] = [
                        0
                    ] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids" and k != "thinking_positions"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_dataset(
    stage_type: str,
    base_dataset_valid,
    tokenizer: PreTrainedTokenizerBase,
):
    """Generate question dataset with tokenization."""

    def process_dataset(sample):
        # Apply chat template if available, otherwise use plain text
        message = [
            {
                "role": "user",
                "content": sample["question"],
            }
        ]

        question = apply_chat_template_if_needed(
            tokenizer, message
        )
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)

        return {
            "input_ids": question_tokenized,
            "idx": sample["idx"],
            "attention_mask": [1] * len(question_tokenized),
            "position_ids": list(range(len(question_tokenized))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    stage_type: str,
    base_dataset,
    configs,
    strategy: ThinkingTokenStrategy,
    tokenizer: PreTrainedTokenizerBase,
    shuffle: bool = False,
    return_text: bool = False,
    debug_num: Optional[int] = None,
):
    """Generate CoT latent dataset with tokenization."""
    thinking_answer_prob = getattr(configs, "thinking_answer_prob", 0.0)
    thinking_answer_extra_prob = getattr(
        configs, "thinking_answer_extra_prob", 0.0
    )
    if isinstance(thinking_answer_prob, list):
        thinking_answer_prob = thinking_answer_prob[1]
        thinking_answer_extra_prob = thinking_answer_extra_prob[1]
        print(f"Using first element of thinking_answer_prob list: {thinking_answer_prob}")
        print(f"Using first element of thinking_answer_extra_prob list: {thinking_answer_extra_prob}")

    if debug_num is not None:
        base_dataset = base_dataset.select(range(debug_num))

    def process_dataset(sample, candidate_indices=None, strategy=None):
        reasoning_list = sample['reasoning_chain'].split('\n')
        full_response = ""
        for idx in range(len(reasoning_list)):
            full_response += f'## Step {idx + 1}: {reasoning_list[idx]}\n'

        full_response += f"The final answer is:\n### {sample['answer']}"
        
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": full_response}
        ]
        
        # Apply chat template if available, returns formatted text string
        formatted_text = apply_chat_template_if_needed(tokenizer, messages)
        
        # Tokenize the formatted text
        full_tokenized = tokenizer.encode(formatted_text, add_special_tokens=False)
        
        # Find where assistant response starts by tokenizing user part only
        user_messages = [{"role": "user", "content": sample["question"]}]
        user_formatted = apply_chat_template_if_needed(tokenizer, user_messages)
        user_tokenized = tokenizer.encode(user_formatted, add_special_tokens=False)
        question_length = len(user_tokenized)

        # Extract question and response tokens from full sequence
        question_tokenized = full_tokenized[:question_length]

        # Create a tokenized sample for strategy processing
        tokenized_sample = {
            'question': sample["question"],
            'reasoning_chain': sample['reasoning_chain'],
            'answer': sample['answer'],
            "question_tokenized": question_tokenized,
            "response_tokenized": full_tokenized[question_length:],
            "full_tokenized": full_tokenized
        }
        
        question_tokens = list(tokenized_sample["question_tokenized"])
        response_tokens = list(tokenized_sample["response_tokenized"])
        thinking_positions: List[int] = []
        
        if strategy is not None:
            decorated_input_ids, inserted_positions = strategy.apply(tokenized_sample, candidate_indices=candidate_indices)
            thinking_positions = inserted_positions
        else:
            decorated_input_ids = full_tokenized
        # add eot
        tokens = decorated_input_ids + [tokenizer.eos_token_id]

        labels = (
            [-100] * len(question_tokens)
            + tokens[len(question_tokens):]
        )

        if return_text:
            label_tokens = [tok for tok in labels if tok != -100]
            return {
                "idx": sample["idx"],
                "input_text": tokenizer.decode(
                    tokens, skip_special_tokens=False
                ),
                "label_text": tokenizer.decode(
                    label_tokens, skip_special_tokens=False
                ),
            }
        dataset_save_path = getattr(configs, "dataset_save_path", None)
        if dataset_save_path is not None and get_rank() == 0:
            os.makedirs(dataset_save_path, exist_ok=True)
            stage_suffix = getattr(
                configs, "current_stage_label", stage_type
            )
            if configs.save_dataset or sample["idx"] < 50:
                with open(
                    f"{dataset_save_path}/{configs.name}_{stage_suffix}_dataset.jsonl",
                    "a",
                ) as f:
                    json_record = {
                        "idx": sample["idx"],
                        "input_text": tokenizer.decode(
                            tokens, skip_special_tokens=False
                        ),
                        "label_text": tokenizer.decode(
                            [tok for tok in labels if tok != -100], skip_special_tokens=False
                        )
                    }
                    f.write(json.dumps(json_record) + "\n")
        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "thinking_positions": thinking_positions if thinking_positions else None,
        }

    # Use multiprocessing for common stage (no CUDA), disable for stages requiring serial processing with CUDA
    # For "pause" stage_type, we want to use strategy even though mode is "common"
    if configs.current_stage_mode == "common" and stage_type != "pause" and strategy is None:
        map_num_proc = 32
        process_dataset = partial(process_dataset, strategy=None)
    elif strategy is not None and strategy.requires_serial_processing:
        map_num_proc = None
        process_dataset = partial(process_dataset, strategy=strategy)
    else:
        map_num_proc = 32
        process_dataset = partial(process_dataset, strategy=strategy)
        
    print(f"Starting dataset processing for stage: {stage_type} with map_num_proc={map_num_proc}")
    if map_num_proc is not None:
        print("Using map function for dataset processing.")
        if get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset,
                remove_columns=list(base_dataset.features),
                num_proc=map_num_proc,
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        if is_distributed():
            import torch.distributed as dist
            dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        batch_size = configs.batch_size_eval if hasattr(configs, "batch_size_eval") else 8
        batch_start = 0
        print(f"Processing dataset in batches of size {batch_size} for stage: {stage_type}")
        processed_dataset = []
        for batch_start in tqdm(
            range(0, len(base_dataset), batch_size),
            desc=f"Processing dataset for stage: {stage_type}",
        ):
            batch_end = min(batch_start + batch_size, len(base_dataset))
            batch_samples = [
                base_dataset[i] for i in range(batch_start, batch_end)
            ]
            # if stage_type not in ["common", "explicit"]:
                    # import pdb; pdb.set_trace()
            candidate_indices = strategy.batch_candidate_indices(batch_samples)
            for i, sample in enumerate(batch_samples):
                processed_sample = process_dataset(
                    sample, candidate_indices=candidate_indices[i]
                )
                processed_dataset.append(processed_sample)
        dataset = processed_dataset
    print(f"✓ Dataset processing completed for stage: {stage_type}")
    print(f"Dataset size: {len(dataset)} samples")
    return dataset