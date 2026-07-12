import os
import random
import yaml
import torch
import numpy as np
from typing import Iterable, Optional
from typing import Any, Dict, List, Optional, Sequence, Union
from collections import OrderedDict

def get_longest_common_suffix(s1, s2):
    return os.path.commonprefix([s1[::-1], s2[::-1]])[::-1]

class Config:
    def __init__(self, dictionary):
        self.__dict__ = dictionary

def print_list_in_json(lst: List[Any]):
    import json
    print(json.dumps(lst, indent=2))

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prune_checkpoints(
    save_dir: str,
    prefix: str,
    max_to_keep: Optional[int],
    exclude_prefixes: Optional[Iterable[str]] = None,
):
    """Keep at most ``max_to_keep`` checkpoint files with the given prefix.

    Removes the oldest checkpoints (by modification time) beyond the limit.
    ``exclude_prefixes`` can be provided to ignore files that start with any of
    the specified prefixes.
    """

    if max_to_keep is None:
        return
    try:
        max_to_keep = int(max_to_keep)
    except (TypeError, ValueError):
        return
    if max_to_keep <= 0:
        return

    exclude_prefixes = list(exclude_prefixes or [])

    try:
        entries = os.listdir(save_dir)
    except FileNotFoundError:
        return

    candidates = []
    for name in entries:
        if not name.startswith(prefix):
            continue
        if any(name.startswith(ex) for ex in exclude_prefixes):
            continue
        path = os.path.join(save_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        candidates.append((mtime, name))

    if len(candidates) <= max_to_keep:
        return

    candidates.sort(key=lambda item: item[0])
    to_remove = candidates[: len(candidates) - max_to_keep]

    for _, name in to_remove:
        path = os.path.join(save_dir, name)
        try:
            os.remove(path)
        except FileNotFoundError:
            continue

def _stage_value(values: Union[Sequence[float], float, int], idx: int, default: float = 0.0):
        if isinstance(values, (list, tuple)):
            if not values:
                return default
            if idx < len(values):
                return values[idx]
            return values[-1]
        return values

def apply_chat_template_if_needed(tokenizer, messages):
    """Apply chat template if tokenizer supports it, otherwise return plain text."""
    # Check if tokenizer has chat_template attribute
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        try:
            # Try to apply chat template
            if messages[-1].get("role", "") == "assistant":
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    continue_final_message=True,
                )
            else:
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            # Fallback if chat template fails
            print(f"Warning: chat_template exists but failed to apply: {e}")
            pass
    
    # Fallback: concatenate message contents
    if isinstance(messages, list):
        return tokenizer.bos_token + "\n\n".join([msg.get("content", "") for msg in messages if isinstance(msg, dict)])
    return messages


class StageManager:
    def __init__(self, cfg):
        raw_limits = getattr(cfg, "labels_per_stage", None)
        if raw_limits is None:
            raise ValueError("labels_per_stage must be provided for multi-stage training")
        limits = [max(x, 0) for x in raw_limits]
        if not limits:
            raise ValueError("labels_per_stage must not be empty")
        if limits[0] != 0:
            raise ValueError("labels_per_stage must start with 0 for common CoT stage")

        self.cfg = cfg
        self.stage_token_limits = limits
        self.stage_epochs = [max(int(e), 0) for e in getattr(cfg, "stage_epochs", [])]
        self.stage_names = [str(name) for name in getattr(cfg, "stage_names", [])]
        self.stage_modes = [str(m) for m in getattr(cfg, "stage_modes", ["common", "hidden_state", "soft_fusion"])]

        # snapshot original per-stage parameters (could be scalars or sequences)
        self.base_insertion_prob = getattr(cfg, "thinking_insertion_prob", 1.0)
        self.base_secondary_prob = getattr(cfg, "thinking_secondary_insertion_prob", 0.0)
        self.base_prompt_tokens = getattr(cfg, "thinking_prompt_tokens", 0)
        self.base_reinforce_threshold = getattr(cfg, "reinforce_prob_threshold", 0.05)
        self.base_reinforce_max_len = getattr(cfg, "reinforce_max_eval_length", 2048)
        self.base_fusion_alpha = getattr(cfg, "fusion_alpha", [0.5, 0.5, 0.5, 0.5])
        self.base_fusion_top_p = getattr(cfg, "fusion_top_p", 0.9)
        self.base_fusion_temperature = getattr(cfg, "fusion_temperature", 1.0)

        self.stage_idx = self.stage_for_epoch(cfg.resume)
        self.current_mode = "common"
        self.current_label = ""
        self.apply(self.stage_idx)

    def stage_for_epoch(self, epoch_idx: int) -> int:
        """Determine which stage an epoch belongs to, skipping stages with 0 epochs."""
        epoch = max(int(epoch_idx), 0)
        for idx, duration in enumerate(self.stage_epochs):
            if idx >= len(self.stage_token_limits) - 1:
                break
            if duration <= 0:
                # Skip stages with 0 epochs
                continue
            if epoch < duration:
                return idx
            epoch -= duration
        return len(self.stage_token_limits) - 1

    def apply(self, stage_idx: int):
        stage_idx = max(0, min(stage_idx, len(self.stage_token_limits) - 1))
        tokens_cap = self.stage_token_limits[stage_idx]

        # Determine stage mode (common, hidden_state, soft_fusion)
        if stage_idx < len(self.stage_modes):
            stage_mode = self.stage_modes[stage_idx]
        else:
            stage_mode = "common" if tokens_cap == 0 else "hidden_state"

        insertion = float(_stage_value(self.base_insertion_prob, stage_idx, 0.0))
        secondary = float(_stage_value(self.base_secondary_prob, stage_idx, 0.0))
        prompt_tokens = int(_stage_value(self.base_prompt_tokens, stage_idx, 0))
        reinforce_threshold = float(
            _stage_value(self.base_reinforce_threshold, stage_idx, 0.05)
        )
        reinforce_max_len = int(
            _stage_value(self.base_reinforce_max_len, stage_idx, self.base_reinforce_max_len)
        )
        fusion_alpha = float(_stage_value(self.base_fusion_alpha, stage_idx, 0.5))
        fusion_top_p = float(_stage_value(self.base_fusion_top_p, stage_idx, 0.9))
        fusion_temperature = float(_stage_value(self.base_fusion_temperature, stage_idx, 1.0))

        label = (
            self.stage_names[stage_idx]
            if stage_idx < len(self.stage_names)
            else f"stage{stage_idx}-{stage_mode}"
        )

        self.cfg.tokens_per_stage = tokens_cap
        self.cfg.thinking_insertion_prob = insertion
        self.cfg.thinking_secondary_insertion_prob = secondary
        self.cfg.thinking_prompt_tokens = prompt_tokens
        self.cfg.reinforce_prob_threshold = reinforce_threshold
        self.cfg.reinforce_max_eval_length = reinforce_max_len
        self.cfg.fusion_alpha = fusion_alpha
        self.cfg.fusion_top_p = fusion_top_p
        self.cfg.fusion_temperature = fusion_temperature
        self.cfg.current_stage_label = label
        self.cfg.current_stage_mode = stage_mode
        self.cfg.current_stage_index = stage_idx

        self.stage_idx = stage_idx
        self.current_mode = stage_mode
        self.current_label = label

        return label, stage_mode, tokens_cap, insertion

    def update_for_epoch(self, epoch_idx: int) -> bool:
        desired = self.stage_for_epoch(epoch_idx)
        if desired != self.stage_idx:
            self.apply(desired)
            return True
        return False

    def describe(self) -> str:
        desc = (
            f"{self.current_label} (mode={self.current_mode}, "
            f"tokens={self.cfg.tokens_per_stage}, prob={self.cfg.thinking_insertion_prob}"
        )
        if self.current_mode == "soft_fusion":
            desc += f", alpha={self.cfg.fusion_alpha:.2f}, top_p={self.cfg.fusion_top_p}"
        desc += ")"
        return desc