import json
import logging
import numpy as np
import os
import pandas as pd
import s3fs
import torch
import torch.utils
import torch.utils.data
import wandb
import ast
import re

from botorch.test_functions import SyntheticTestFunction
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EvalPrediction,
    PreTrainedTokenizer,
)
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINING_ARGS_NAME,
    TRAINER_STATE_NAME,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

torch.set_printoptions(threshold=10_000)


def maybe_log(logger: logging.Logger, msg: str, level: str = "info"):
    """Log a message if a logger is provided, otherwise do nothing.

    Args:
        logger: Logger instance, or None to skip logging.
        msg: Message to log.
        level: Log level name (e.g. "info", "warning", "error").
    """
    if logger is None:
        return
    log_fn = getattr(logger, level)
    log_fn(msg)


def wandb_setup(cfg: DictConfig):
    """
    Runs `wandb.init` and `wandb.login`.
    The values in `cfg` are logged to the wandb run.
    """
    if not hasattr(cfg, "wandb_host") or cfg.wandb_host is None:
        cfg["wandb_host"] = "https://api.wandb.ai"

    if not hasattr(cfg, "wandb_mode") or cfg.wandb_mode is None:
        cfg["wandb_mode"] = "online"

    if not hasattr(cfg, "project_name") or cfg.project_name is None:
        cfg["project_name"] = "finetune_ehrlich"

    if not hasattr(cfg, "exp_name") or cfg.exp_name is None:
        cfg["exp_name"] = "default_group"

    wandb.login(host=cfg.wandb_host)

    wandb.init(
        project=cfg.project_name,
        mode=cfg.wandb_mode,
        group=cfg.exp_name,
        name=cfg.job_name,
        config=OmegaConf.to_container(cfg),
    )


# Function `strtobool` copied and adapted from `distutils` (as deprected
# in Python 3.10).
# Reference: https://github.com/python/cpython/blob/48f9d3e3faec5faaa4f7c9849fecd27eae4da213/Lib/distutils/util.py#L308-L321


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False booleans.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError: if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(
        f"Invalid truth value, it should be a string but {val} was provided instead."
    )


def check_if_checkpoint_complete(
    fs: s3fs.S3FileSystem,
    local_checkpoint_dir: str,
    s3_checkpoint_dir: str,
    num_gpus: int = 2,
    num_shards: int = 3,
    logger: logging.Logger = None,
) -> bool:
    """Check that a training checkpoint has all required files.

    Syncs missing files between local and S3 storage in both directions:
    downloads from S3 if a local file is missing, uploads to S3 if only
    the local copy exists.

    Args:
        fs: S3 filesystem client.
        local_checkpoint_dir: Local path to the checkpoint directory.
        s3_checkpoint_dir: S3 path to the checkpoint directory.
        num_gpus: Number of GPUs (determines expected rng_state files).
        num_shards: Number of model weight shards.
        logger: Optional logger for status messages.

    Returns:
        True if all required files are present (or were successfully synced),
        False if any file is missing from both local and S3.
    """
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    files_to_check_for = [
        "config.json",
        "generation_config.json",
        OPTIMIZER_NAME,
        SCHEDULER_NAME,
        TRAINING_ARGS_NAME,
        TRAINER_STATE_NAME,
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    files_to_check_for.extend(
        [
            f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            for i in range(1, num_shards + 1)
        ]
    )
    files_to_check_for.extend([f"rng_state_{i}.pth" for i in range(num_gpus)])
    if s3_checkpoint_dir.endswith("/"):
        s3_checkpoint_dir = s3_checkpoint_dir[:-1]
    if not local_checkpoint_dir.endswith("/"):
        local_checkpoint_dir += "/"
    for fn in files_to_check_for:
        if not os.path.exists(os.path.join(local_checkpoint_dir, fn)):
            maybe_log(
                logger,
                f"Checkpoint located at {local_checkpoint_dir} not complete. Missing {fn}.",
                level="warning",
            )
            # But if the file is available on S3, just copy it over!
            if fs.exists(f"{s3_checkpoint_dir}/{fn}"):
                maybe_log(
                    logger,
                    f"Found {fn} in {s3_checkpoint_dir}. Copying over.",
                    level="info",
                )
                fs.get(f"{s3_checkpoint_dir}/{fn}", local_checkpoint_dir)
            else:
                return False
        else:
            # Check that it also exists in s3!
            if not fs.exists(f"{s3_checkpoint_dir}/{fn}"):
                maybe_log(
                    logger,
                    f"{fn} found in {local_checkpoint_dir} but not on s3. Copying over to {s3_checkpoint_dir}/.",
                    level="info",
                )
                fs.put(f"{local_checkpoint_dir}{fn}", f"{s3_checkpoint_dir}/")
    return True


def find_and_log_checkpoints(
    fs: s3fs.S3FileSystem,
    s3_output_dir: str,
    local_output_dir: str,
    num_gpus: int = 2,
    num_shards: int = 3,
    logger: logging.Logger = None,
) -> Optional[str]:
    """
    If checkpoints are available, return the latest local checkpoint.
    Also ensure that the best checkpoint is saved and copied to local directory.
    """
    if not s3_output_dir.endswith("/"):
        s3_output_dir += "/"
    checkpoint_dirs = []
    if not fs.exists(s3_output_dir):
        maybe_log(logger, f"{s3_output_dir} does not exist.", level="info")
        return None
    for fp in fs.ls(s3_output_dir):
        if f"{PREFIX_CHECKPOINT_DIR}-" in fp:
            ckpt_num = fp.split(f"{PREFIX_CHECKPOINT_DIR}-")[-1]
            checkpoint_dirs.append((f"s3://{fp}", int(ckpt_num)))
    if len(checkpoint_dirs) == 0:
        return None
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda p: p[1])
    found_good_ckpt = False
    for ckpt_idx in range(len(checkpoint_dirs) - 1, -1, -1):
        latest_s3_checkpoint_dir = checkpoint_dirs[ckpt_idx][0]
        latest_local_checkpoint_dir = os.path.join(
            local_output_dir, f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_dirs[ckpt_idx][1]}"
        )
        if check_if_checkpoint_complete(
            fs,
            latest_local_checkpoint_dir,
            latest_s3_checkpoint_dir,
            num_gpus=num_gpus,
            num_shards=num_shards,
            logger=logger,
        ):
            found_good_ckpt = True
            break
    if not found_good_ckpt:
        return None

    # Now look for best checkpoint
    trainer_state = json.load(
        open(f"{latest_local_checkpoint_dir}/{TRAINER_STATE_NAME}")
    )
    best_ckpt_dir = trainer_state["best_model_checkpoint"]
    if best_ckpt_dir is None:
        return latest_local_checkpoint_dir
    best_ckpt_num = best_ckpt_dir.split(f"{PREFIX_CHECKPOINT_DIR}-")[-1]
    best_ckpt_results = check_if_checkpoint_complete(
        fs,
        best_ckpt_dir,
        f"{s3_output_dir}{PREFIX_CHECKPOINT_DIR}-{best_ckpt_num}",
        num_gpus=num_gpus,
        num_shards=num_shards,
        logger=logger,
    )
    if best_ckpt_results is None:
        maybe_log(
            logger,
            f"Found latest checkpoint {latest_local_checkpoint_dir} but not the best checkpoint {best_ckpt_dir}.",
            level="warning",
        )
        return None
    return latest_local_checkpoint_dir


def truncate_after_right_bracket(
    generated_str: str,
    logger: logging.Logger = None,
    return_num_chars_truncated: bool = False,
) -> Union[str, Tuple[str, int]]:
    """If the generated string contains a right bracket, truncate all content after it."""
    right_bracket = "]"
    num_truncated_chars = 0
    if right_bracket not in generated_str:
        if return_num_chars_truncated:
            return generated_str, num_truncated_chars
        return generated_str
    idx = generated_str.find(right_bracket)
    if idx != -1:
        content_to_truncate = generated_str[idx + 1 :]
        num_truncated_chars = len(content_to_truncate)
        truncated_str = generated_str[: idx + 1]
    else:
        truncated_str = generated_str
    if return_num_chars_truncated:
        return truncated_str, num_truncated_chars
    return truncated_str


def truncate_after_right_bracket_w_logps(
    output_token_ids: torch.Tensor,
    output_logps: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    length_normalized: bool = True,
) -> Tuple[str, float]:
    """
    Given the output token IDs and token log-likelihoods, truncate both
    after the first right bracket.

    output_token_ids: 1-D tensor containing the token IDs of the output sequence. Shape: (max_new_tokens,)
    output_logps: 1-D tensor containing the log-probs of each output token. Shape: (max_new_tokens,)
    tokenizer: pre-trained tokenizer
    length_normalized: If true, returns the average log prob. Otherwise, returns the sum.

    Returns a tuple of the truncated string and the sum or average of the log probs.
    """
    assert output_token_ids.shape == output_logps.shape
    rbrac_id = tokenizer.encode("]")[0]

    if rbrac_id in output_token_ids:
        idx = torch.where(output_token_ids == rbrac_id)[0][0].item() + 1
        output_token_ids = output_token_ids[:idx]
        output_logps = output_logps[:idx]
    output_str = tokenizer.decode(output_token_ids)
    cum_logps = torch.sum(output_logps)
    if length_normalized:
        cum_logps = cum_logps / len(output_logps)
    return (output_str, cum_logps.item())


def starts_with_other_tensor(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Checks if x (a 1-D tensor) starts with y (another 1-D tensor)"""
    return all(x[: len(y)] == y)


def load_test_fn_from_file(test_fn_fp: str, test_fn_type: str) -> SyntheticTestFunction:
    """Load a test function from a JSON file of saved hyperparameters.

    Args:
        test_fn_fp: Path to JSONL file containing test function parameters.
        test_fn_type: Either "ehrlich" or "rough_mt_fuji".

    Returns:
        Initialized SyntheticTestFunction instance.
    """
    test_fn_params = pd.read_json(test_fn_fp, orient="records", lines=True).to_dict(
        "records"
    )[0]
    if test_fn_type == "ehrlich":
        test_fn = Ehrlich(
            num_states=test_fn_params["num_states"],
            dim=test_fn_params["dim"],
            num_motifs=test_fn_params["num_motifs"],
            motif_length=test_fn_params["motif_length"],
            quantization=test_fn_params["quantization"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    else:
        test_fn = RoughMtFuji(
            dim=test_fn_params["dim"],
            additive_term=test_fn_params["additive_term"],
            random_term_std=test_fn_params["random_term_std"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    return test_fn


def parse_particle_and_score(
    input_str: str, test_fn: SyntheticTestFunction
) -> Optional[Tuple[List[int], float]]:
    """
    Checks that <input_str> can be parsed into a list with test_fn.dim integer elements.
    For the Ehrlich function, also checks the range of the values.

    If parsable, returns tuple of (particle, score). otherwise, returns None.
    """
    try:
        particle = json.loads(input_str)
    except (ValueError, TypeError):
        return None
    if not isinstance(particle, list):
        return None
    try:
        if any([int(x) != x for x in particle]):
            return None
    except (ValueError, TypeError):
        return None
    particle = [int(x) for x in particle]
    if len(particle) != test_fn.dim:
        return None
    if hasattr(test_fn, "num_states"):
        if any([x >= test_fn.num_states or x < 0 for x in particle]):
            return None
    score = test_fn(torch.LongTensor([particle])).item()
    return particle, score


# Flat integer lists embedded in prompts/targets (e.g. "<inc> [1, 2, -1]\n").
_LIST_LITERAL_PATTERN = re.compile(r"\[[^\]]*\]")


def _parse_list_literal(list_literal: str) -> Optional[List]:
    """Parse a bracketed list literal; return None if not a list."""
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(list_literal)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(value, list):
            return value
    return None


def integer_list_length_or_none(text: str) -> Optional[int]:
    """Return the length of a parseable list of integers in text, or None."""
    candidates: List[List] = []
    parsed = _parse_list_literal(text.strip())
    if parsed is not None:
        candidates.append(parsed)
    else:
        for match in _LIST_LITERAL_PATTERN.finditer(text):
            p = _parse_list_literal(match.group(0))
            if p is not None:
                candidates.append(p)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if any(int(x) != x for x in candidate):
                continue
            return len([int(x) for x in candidate])
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def _filter_pad_int_from_list_literal(list_literal: str, pad_int: int) -> str:
    """Remove pad_int entries from one [...] substring; return unchanged on parse failure."""
    sequence_list = _parse_list_literal(list_literal)
    if sequence_list is None:
        return list_literal
    filtered_list = [
        x
        for x in sequence_list
        if not (isinstance(x, (int, float)) and int(x) == pad_int)
    ]
    return json.dumps(filtered_list)


def remove_integer_padding_from_list(sequence_str: str, pad_int: int = -1) -> str:
    """Remove pad_int entries from every list literal in a string.

    Works on bare lists and on strings with prefixes/suffixes, e.g.
    ``"<inc> [12, 31, 2, 18, -1, -1]\\n"`` -> ``"<inc> [12, 31, 2, 18]\\n"``,
    or ``"Score: 1.23\\nParticle: [12, 31, -1]"``.

    Args:
        sequence_str: Text containing one or more ``[...]`` list literals.
        pad_int: Integer padding value to drop (default ``-1``).

    Returns:
        The input string with pad_int removed from each parsed list; unchanged
        if no list literals are found or a literal cannot be parsed.
    """
    if not sequence_str:
        return sequence_str

    def _replace_list(match: re.Match) -> str:
        return _filter_pad_int_from_list_literal(match.group(0), pad_int)

    return _LIST_LITERAL_PATTERN.sub(_replace_list, sequence_str)


def drop_pad_ints(particle: Iterable[Any], pad_int: int = -1) -> List[int]:
    """Drop pad_int entries from a particle list before JSON serialization."""
    return [int(x) for x in particle if int(x) != pad_int]


def parse_particle_and_score_permissive(
    input_str: str, test_fn: SyntheticTestFunction, cfg: DictConfig = None, logger: logging.Logger = None, pad_token: int = -1
) -> Optional[Tuple[List[int], float]]:
    """
    Checks that <input_str> can be parsed into a list with test_fn.dim integer elements.
    For the Ehrlich function, also checks the range of the values.

    If parsable, returns tuple of (particle, score). otherwise, returns None.
    """
    try:
        particle = json.loads(input_str)
    except (ValueError, TypeError):
        return None
    if not isinstance(particle, list):
        return None
    try:
        if any([int(x) != x for x in particle]):
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    particle = [
        int(x) for x in particle
    ]  ## Will return this one (only modifying length if needed)
    if len(particle) == 0:
        return None
    len_original_particle = len(particle)
    
    particle_for_scoring = (
        particle.copy()
    )  ## Will use this one for scoring (permissive score function)

    if len(particle) != test_fn.dim:
        particle_for_scoring = np.pad(
            particle_for_scoring,
            (0, max(0, test_fn.dim - len(particle_for_scoring))),
            mode="wrap",
        )[: test_fn.dim].tolist()
        if len(particle) < test_fn.dim:
            particle.extend([pad_token for i in range(test_fn.dim - len(particle))])
        else:
            particle = particle[: test_fn.dim]

    # if len_original_particle < cfg.min_rel_len_feasible_particle * test_fn.dim:
    #     return particle, float("inf")

    if hasattr(test_fn, "num_states"):
        if any([x >= test_fn.num_states or x < 0 for x in particle_for_scoring]):
            particle_for_scoring = np.pad(
                particle_for_scoring,
                (0, max(0, test_fn.dim - len(particle_for_scoring))),
                mode="wrap",
            )[: test_fn.dim].tolist()
            particle_for_scoring = np.clip(particle_for_scoring, a_min=0, a_max=test_fn.dim - 1)
    score = test_fn(torch.LongTensor([particle_for_scoring])).item() if len_original_particle >= cfg.min_rel_len_feasible_particle * test_fn.dim else float("inf")

    particle = np.clip(particle, a_min=-np.iinfo(np.int32).max, a_max=np.iinfo(np.int32).max)
    return particle, score


def preprocess_generations(
    generated_token_ids: torch.LongTensor,
    inputs: Mapping[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> torch.LongTensor:
    """
    Workaround for a bug where the HuggingFace Trainer evaluate() function
    gathers the generations across GPUs but not the inputs. Find the generations that map to the inputs on this device.
    """
    generated_token_ids[generated_token_ids == -100] = tokenizer.pad_token_id
    input_ids = inputs["input_ids"]
    input_ids[input_ids == -100] = tokenizer.pad_token_id

    if generated_token_ids.shape[0] > input_ids.shape[0]:
        generation_idxs = []
        for i in range(input_ids.shape[0]):
            ex_input_ids = input_ids[i]
            # get only the part of the input IDs that is before the target sequence
            target_idxs = torch.where(inputs["labels"][i] != -100)[0]
            if len(target_idxs) > 0:
                len_non_target_ids = target_idxs[0]
            else:
                len_non_target_ids = 0
            ex_input_ids = ex_input_ids[:len_non_target_ids]
            for j in range(generated_token_ids.shape[0]):
                if starts_with_other_tensor(generated_token_ids[j], ex_input_ids):
                    generation_idxs.append(j)
                    break
            if len(generation_idxs) != i + 1:
                generated_token_ids_str = "\n".join(
                    [f"{t}" for t in generated_token_ids]
                )
                raise ValueError(
                    "Could not find an input with IDs that the current generation starts with.\n"
                    + f"Original input:{input_ids[i]}\n{input_ids[i].shape}\n"
                    + f"Sliced input:\n{ex_input_ids}\n{ex_input_ids.shape}\n"
                    + f"Labels:\n{inputs['labels'][i]}\n{inputs['labels'][i].shape}\n"
                    + f"Generations:\n{generated_token_ids_str}\n{generated_token_ids.shape}"
                )
        generated_token_ids = generated_token_ids[generation_idxs]
    return generated_token_ids


def postprocess_generations(
    generated_token_ids: torch.LongTensor,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    tokenizer: PreTrainedTokenizer,
    logger: logging.Logger,
) -> Tuple[List[str], int]:
    """
    Remove the inputs from the output generations and truncate characters past the first ']'
    symbol. Decode generations.
    """
    predictions = []
    for i in range(generated_token_ids.shape[0]):
        ex_input_ids = input_ids[i]
        # get only the part of the input IDs that is before the target sequence
        target_idxs = torch.where(labels[i] != -100)[0]
        if len(target_idxs) > 0:
            len_non_target_ids = target_idxs[0]
        else:
            len_non_target_ids = len(labels[i])
        ########

        input_decode = tokenizer.decode(
            ex_input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        entire_decode = tokenizer.decode(
            generated_token_ids[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_generation = tokenizer.decode(
            generated_token_ids[i][len_non_target_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        logger.info(
            f"Entire decode: {entire_decode}.\nInput tokens: {input_decode}\nDecoded after removing initial prompt tokens: {decoded_generation}"
        )
        truncated, num_chars_truncated = truncate_after_right_bracket(
            decoded_generation, logger=logger, return_num_chars_truncated=True
        )
        predictions.append(truncated)
    return predictions, num_chars_truncated


class EvaluatorPlainPairs:
    """Evaluator for plain score-particle pair generations.

    Scores generated particles against the test function, tracking parsability,
    feasibility, and accuracy metrics across evaluation batches.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
        logger: logging.Logger = None,
    ):
        self.tokenizer = tokenizer
        self.test_fn = load_test_fn_from_file(cfg.test_fn_fp, cfg.test_fn_type)
        self.reset()
        self.logger = logger

    def reset(self):
        self.num_parsable = 0
        self.num_correct_length = 0
        self.num_values_in_range = 0
        self.num_feasible = 0
        self.num_total = 0
        self.num_chars_truncated = []
        self.scores = []
        self.expected_scores = []

    @torch.no_grad()
    def __call__(
        self, eval_pred: EvalPrediction, compute_result: bool = False
    ) -> Optional[Mapping[str, float]]:
        generated_token_ids, _, inputs = eval_pred
        generated_token_ids = preprocess_generations(
            generated_token_ids, inputs, self.tokenizer
        )

        self.num_total += generated_token_ids.shape[0]
        input_ids = inputs["input_ids"]
        predictions, num_chars_truncated = postprocess_generations(
            generated_token_ids,
            input_ids,
            inputs["labels"],
            self.tokenizer,
            self.logger,
        )
        self.num_chars_truncated.append(num_chars_truncated)
        input_strs = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        exp_scores = []
        for input_str in input_strs:
            input_str = input_str[: input_str.find("Particle:")]
            input_str = input_str[len("Score:") :]
            exp_score = float(input_str)
            exp_scores.append(exp_score)
        # make new lists for predictions and targets because not all
        # scores will be added (since some are NaN or inf)
        parsed_predictions = []
        targets = []
        for i in range(len(predictions)):
            pred = predictions[i]
            # first see if parsable as tensor of ints
            try:
                pred_list = json.loads(pred)
                assert isinstance(pred_list, list)
                particle = []
                for x in pred_list:
                    if int(x) != x:
                        continue
                    particle.append(int(x))
            except (json.JSONDecodeError, ValueError, TypeError, AssertionError):
                # not parsable
                maybe_log(
                    self.logger,
                    f"Prediction is not parsable. Continuing. Prediction:\n{pred}",
                    level="warning",
                )
                continue
            self.num_parsable += 1
            if len(particle) != self.test_fn.dim:
                maybe_log(
                    self.logger,
                    f"Prediction has the wrong length ({len(particle)} != {self.test_fn.dim}): {particle}",
                    level="warning",
                )
                continue
            self.num_correct_length += 1
            if any([x < 0 or x >= self.test_fn.num_states for x in particle]):
                maybe_log(
                    self.logger,
                    f"Prediction has some elements that are outside the range [0, {self.test_fn.num_states - 1}]: {particle}",
                    level="warning",
                )
                continue
            self.num_values_in_range += 1
            particle = torch.FloatTensor(particle)
            score = self.test_fn(particle.unsqueeze(0)).item()
            if score == float("inf"):
                maybe_log(
                    self.logger,
                    f"obs score {score} is inf. Continuing...",
                    level="warning",
                )
                continue
            self.num_feasible += 1
            if exp_scores[i] == float("inf"):
                maybe_log(
                    self.logger,
                    f"exp score {exp_scores[i]} is inf. Continuing...",
                    level="warning",
                )
                continue
            parsed_predictions.append(score)
            targets.append(exp_scores[i])
        self.scores.extend(parsed_predictions)
        self.expected_scores.extend(targets)

        if not compute_result:
            return
        self.expected_scores = np.array(self.expected_scores)
        self.scores = np.array(self.scores)
        squared_errors = (
            (self.expected_scores - self.scores) ** 2 if len(self.scores) > 0 else None
        )
        output_dict = {
            "%_parsable": self.num_parsable / self.num_total,
            "%_correct_length": (
                self.num_correct_length / self.num_parsable
                if self.num_parsable > 0
                else None
            ),
            "%_values_in_range": (
                self.num_values_in_range / self.num_correct_length
                if self.num_correct_length > 0
                else None
            ),
            "%_feasible": (
                self.num_feasible / self.num_values_in_range
                if self.num_values_in_range > 0
                else None
            ),
            "avg_score": np.mean(self.scores) if len(self.scores) > 0 else None,
            "avg_num_chars_truncated": np.mean(self.num_chars_truncated),
            "rmse": (
                np.sqrt(np.mean(squared_errors)) if squared_errors is not None else None
            ),
        }
        self.reset()
        return output_dict


def get_response_template_plain_pairs(tokenizer: PreTrainedTokenizer):
    """Return token IDs for the plain-pairs response template "\\nParticle:".

    Args:
        tokenizer: Tokenizer to encode the template string.

    Returns:
        List of token IDs for the response template.
    """
    response_template_with_context = "\nParticle:"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )
    return response_template_ids


def formatting_texts_func_plain_pairs(
    examples: Mapping[str, Iterable[Any]],
) -> List[str]:
    """Format examples as "Score: {score}\\nParticle: {json_list}" strings.

    Args:
        examples: Batch of examples with "score" and "particle" fields.

    Returns:
        List of formatted training strings.
    """
    output_texts = []
    for i in range(len(examples["score"])):
        score = examples["score"][i]
        particles = drop_pad_ints(json.loads(examples["particle"][i]))
        particles_str = json.dumps(particles)
        output_texts.append(f"Score: {score:.2f}\nParticle: {particles_str}")
    return output_texts


def formatting_texts_func_single_seq(
    examples: Mapping[str, Iterable[Any]],
) -> List[str]:
    """Format examples as plain pairs, auto-detecting particle and score field names.

    Supports multiple field naming conventions (e.g. "higher_score_particle",
    "chosen", "prompt", "particle") and their corresponding score fields.

    Args:
        examples: Batch of examples with auto-detected particle and score fields.

    Returns:
        List of formatted "Score: {score}\\nParticle: {json_list}" strings.
    """
    ## Get particle_field and score_field
    if "higher_score_particle" in examples:
        particle_field = "higher_score_particle"
    elif "lower_score_particle" in examples:
        particle_field = "lower_score_particle"
    elif "chosen" in examples:
        particle_field = "chosen"
    elif "prompt" in examples:
        particle_field = "prompt"
    elif "particle" in examples:
        particle_field = "particle"
    else:
        raise ValueError("No recognized particle field")

    if "higher_score_particle_score" in examples:
        score_field = "higher_score_particle_score"
    elif "lower_score_particle_score" in examples:
        score_field = "lower_score_particle_score"
    elif "cg_lik_ratio_opt_over_mix" in examples:
        score_field = "cg_lik_ratio_opt_over_mix"
    else:
        score_field = "score"

    ##
    output_texts = []
    for i in range(len(examples[score_field])):
        score = examples[score_field][i]
        particles = drop_pad_ints(examples[particle_field][i])
        particles_str = json.dumps(particles)
        output_texts.append(f"Score: {score:.2f}\nParticle: {particles_str}")
    return output_texts


def get_response_template_edit_pairs(tokenizer: PreTrainedTokenizer):
    """Return token IDs for the edit-pairs response template "\\n".

    Args:
        tokenizer: Tokenizer to encode the template string.

    Returns:
        List of token IDs for the response template.
    """
    response_template_with_context = "\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )
    return response_template_ids


def formatting_texts_func_edit_pairs(
    examples: Mapping[str, Iterable[Any]],
    include_target: bool = True,
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    prefix: str = "<inc>",
) -> List[str]:
    """Format edit pairs as "{prefix} {input}\\n{target}" strings for DPO training.

    Args:
        examples: Batch of examples with particle pair fields.
        include_target: Whether to include the target particle in the output.
        higher_score_particle_field: Field name for the input (higher-score) particle.
        lower_score_particle_field: Field name for the target (lower-score) particle.
        prefix: Control code prefix (e.g. "<inc>"). Empty string to omit.

    Returns:
        List of formatted training strings.
    """
    # lower score is better!
    output_texts = []
    if prefix:
        prefix_str = f"{prefix} "
    else:
        prefix_str = ""
    for i in range(len(examples[higher_score_particle_field])):
        input_particle = drop_pad_ints(examples[higher_score_particle_field][i])
        if include_target:
            target_particle = drop_pad_ints(examples[lower_score_particle_field][i])
            target_str = f"{json.dumps(target_particle)}"
        else:
            target_str = ""
        particles_str = f"{prefix_str}{json.dumps(input_particle)}\n{target_str}"
        output_texts.append(particles_str)
    return output_texts


class EvaluatorEditPairs:
    """Evaluator for edit-pair (particle improvement) generations.

    Scores generated particles against the test function, tracking improvement
    rewards, parsability, and quality metrics across evaluation batches.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
        logger: logging.Logger = None,
        control_code: str = "<inc>",
    ):
        self.tokenizer = tokenizer
        self.test_fn = load_test_fn_from_file(cfg.test_fn_fp, cfg.test_fn_type)
        self.reset()
        self.logger = logger
        self.control_code = control_code

    def reset(self):
        self.num_parsable = 0
        self.num_correct_length = 0
        self.num_values_in_range = 0
        self.num_feasible = 0
        self.num_total = 0
        self.num_repeated_input = 0
        self.num_chars_truncated = []
        self.scores = []
        self.rewards = []
        self.num_decreased_score = 0

    def get_reward(self, input_score: float, target_score: float) -> float:
        if input_score is None or np.isinf(input_score) or np.isnan(input_score):
            input_score = 0.0
        if target_score is None or np.isinf(target_score) or np.isnan(target_score):
            target_score = 0.0
        if input_score > target_score:
            return input_score - target_score
        return 0.0

    @torch.no_grad()
    def __call__(
        self, eval_pred: EvalPrediction, compute_result: bool = False
    ) -> Optional[Mapping[str, float]]:
        generated_token_ids, _, inputs = eval_pred
        generated_token_ids = preprocess_generations(
            generated_token_ids, inputs, self.tokenizer
        )
        self.num_total += generated_token_ids.shape[0]

        input_ids = inputs["input_ids"]
        predictions, num_chars_truncated = postprocess_generations(
            generated_token_ids,
            input_ids,
            inputs["labels"],
            self.tokenizer,
            self.logger,
        )
        self.num_chars_truncated.append(num_chars_truncated)

        input_strs = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        input_scores = []
        input_particles = []
        for input_str in input_strs:
            input_str = input_str[len(self.control_code) :]
            input_str = input_str[: input_str.find("\n")]
            input_particle = torch.FloatTensor(json.loads(input_str))
            input_particles.append(input_particle)
            input_score = self.test_fn(input_particle.unsqueeze(0)).item()
            input_scores.append(input_score)

        # make new lists for predictions because not all
        # scores will be added (since some are NaN or inf)
        parsed_predictions = []
        for i in range(len(predictions)):
            pred = predictions[i]
            # first see if parsable as tensor of ints
            try:
                pred_list = json.loads(pred)
                assert isinstance(pred_list, list)
                particle = []
                for x in pred_list:
                    if int(x) != x:
                        continue
                    particle.append(int(x))
            except (json.JSONDecodeError, ValueError, TypeError, AssertionError):
                # not parsable
                maybe_log(
                    self.logger,
                    f"Prediction is not parsable. Continuing. Prediction:\n{pred}",
                    level="warning",
                )
                continue
            self.num_parsable += 1
            if len(particle) != self.test_fn.dim:
                maybe_log(
                    self.logger,
                    f"Prediction has the wrong length ({len(particle)} != {self.test_fn.dim}): {particle}",
                    level="warning",
                )
                continue
            self.num_correct_length += 1
            if any([x < 0 or x >= self.test_fn.num_states for x in particle]):
                maybe_log(
                    self.logger,
                    f"Prediction has some elements that are outside the range [0, {self.test_fn.num_states - 1}]: {particle}",
                    level="warning",
                )
                continue
            self.num_values_in_range += 1
            particle = torch.FloatTensor(particle)
            score = self.test_fn(particle.unsqueeze(0)).item()
            reward = self.get_reward(input_scores[i], score)
            self.rewards.append(reward)
            # score is in (-1.0, infinity)
            if score < input_scores[i]:
                self.num_decreased_score += 1
            elif score == input_scores[i]:
                # check if the model outputted the same particle as was in the prompt
                is_same = all([x == y for x, y in zip(particle, input_particles[i])])
                if is_same:
                    self.num_repeated_input += 1

            if score == float("inf"):
                maybe_log(
                    self.logger,
                    f"obs score {score} is inf. Continuing...",
                    level="warning",
                )
                continue
            self.num_feasible += 1
            parsed_predictions.append(score)
        self.scores.extend(parsed_predictions)

        if not compute_result:
            return
        self.scores = np.array(self.scores)
        self.rewards = np.array(self.rewards)
        output_dict = {
            "%_parsable": self.num_parsable / self.num_total,
            "%_correct_length": (
                self.num_correct_length / self.num_parsable
                if self.num_parsable > 0
                else None
            ),
            "%_values_in_range": (
                self.num_values_in_range / self.num_correct_length
                if self.num_correct_length > 0
                else None
            ),
            "%_decreased_score": (
                self.num_decreased_score / self.num_values_in_range
                if self.num_values_in_range > 0
                else None
            ),
            "%_repeated_input": (
                self.num_repeated_input / self.num_values_in_range
                if self.num_values_in_range > 0
                else None
            ),
            "%_feasible": (
                self.num_feasible / self.num_values_in_range
                if self.num_values_in_range > 0
                else None
            ),
            "avg_score": np.mean(self.scores) if len(self.scores) > 0 else None,
            "min_score": np.min(self.scores) if len(self.scores) > 0 else None,
            "avg_num_chars_truncated": np.mean(self.num_chars_truncated),
            "avg_reward": np.mean(self.rewards) if len(self.rewards) > 0 else None,
            "max_reward": np.max(self.rewards) if len(self.rewards) > 0 else None,
        }
        maybe_log(
            self.logger,
            f"num_parsable: {self.num_parsable}, num_total: {self.num_total},"
            + f" num_correct_length: {self.num_correct_length}, "
            + f"num_values_in_range: {self.num_values_in_range}, "
            + f"num_decreased_score: {self.num_decreased_score}, "
            + f"num_repeated_input: {self.num_repeated_input}, "
            + f"num_feasible: {self.num_feasible} ",
            level="info",
        )
        self.reset()
        return output_dict


def get_ehrlich_rewards(
    input_scores: List[float], target_scores: List[float]
) -> torch.Tensor:
    """Compute margin-based rewards: max(input_score - target_score, 0).

    Infeasible scores (None, inf, NaN) are treated as 0.

    Args:
        input_scores: Scores of the input (higher-score) particles.
        target_scores: Scores of the target (lower-score) particles.

    Returns:
        FloatTensor of non-negative rewards, one per pair.
    """
    rewards = []
    for i, t in zip(input_scores, target_scores):
        if i is None or np.isinf(i) or np.isnan(i):
            i = 0.0  # set scores of infeasible particles to 0
        if t is None or np.isinf(t) or np.isnan(t):
            t = 0.0
        if i > t:
            rewards.append(i - t)
        else:
            rewards.append(0)
    return torch.FloatTensor(rewards)


def get_ehrlich_metrics_for_outputs(
    random_batches_dataset: torch.utils.data.Dataset,
    test_fn: SyntheticTestFunction,
    outputs: List[str],
    input_field_name: str,
    input_score_field_name: str,
) -> Dict[str, float]:
    """
    Returns both a dictionary of computed metrics as well as the list of scores.
    """
    input_particles = [d[input_field_name] for d in random_batches_dataset]
    input_scores = [d[input_score_field_name] for d in random_batches_dataset]
    total_chars_truncated = 0
    num_parsable = 0
    num_correct_length = 0
    num_values_in_range = 0
    num_feasible = 0
    num_repeated_input = 0
    all_scores_including_nulls = []
    num_decreased_score = 0
    rewards = []

    for i, o in enumerate(outputs):
        o, num_chars_truncated = truncate_after_right_bracket(
            o, return_num_chars_truncated=True
        )
        total_chars_truncated += num_chars_truncated
        try:
            o_list = json.loads(o)
            assert isinstance(o_list, list)
            assert all([int(x) == x for x in o_list])
            particle = [int(x) for x in o_list]
        except (json.JSONDecodeError, ValueError, TypeError, AssertionError):
            logging.warning(f"Prediction is not parsable. Continuing. Prediction:\n{o}")
            all_scores_including_nulls.append(None)
            continue
        num_parsable += 1
        if len(o_list) != test_fn.dim:
            all_scores_including_nulls.append(None)
            continue
        num_correct_length += 1
        if isinstance(test_fn, Ehrlich):
            if any([x < 0 or x >= test_fn.num_states for x in particle]):
                all_scores_including_nulls.append(None)
                continue
            num_values_in_range += 1
        if particle == input_particles[i]:
            num_repeated_input += 1
        particle = torch.FloatTensor(particle).unsqueeze(0)
        score = test_fn(particle).item()
        reward = get_ehrlich_rewards([input_scores[i]], [score])[0].item()
        rewards.append(reward)
        if input_scores[i] is None and score != float("inf"):
            num_decreased_score += 1
        elif input_scores[i] is not None and score < input_scores[i]:
            num_decreased_score += 1
        if score == float("inf"):
            all_scores_including_nulls.append(None)
            continue
        num_feasible += 1
        all_scores_including_nulls.append(score)
    num_total = len(outputs)
    scores_wo_nulls = [s for s in all_scores_including_nulls if s is not None]
    output_metrics = {
        "%_parsable": num_parsable / num_total,
        "%_correct_length": (
            num_correct_length / num_parsable if num_parsable > 0 else np.nan
        ),
        "%_decreased_score": (
            num_decreased_score / num_values_in_range
            if num_values_in_range > 0
            else np.nan
        ),
        "%_repeated_input": (
            num_repeated_input / num_values_in_range
            if num_values_in_range > 0
            else np.nan
        ),
        "%_feasible": (
            num_feasible / num_values_in_range if num_values_in_range > 0 else np.nan
        ),
        "avg_score": (np.mean(scores_wo_nulls) if len(scores_wo_nulls) > 0 else np.nan),
        "min_score": (np.min(scores_wo_nulls) if len(scores_wo_nulls) > 0 else np.nan),
        "avg_reward": (np.mean(rewards) if len(rewards) > 0 else np.nan),
        "max_reward": np.max(rewards) if len(rewards) > 0 else np.nan,
        "avg_num_chars_truncated": total_chars_truncated / num_total,
    }
    if isinstance(test_fn, Ehrlich):
        output_metrics["%_values_in_range"] = (
            num_values_in_range / num_correct_length
            if num_correct_length > 0
            else np.nan
        )
    return output_metrics
