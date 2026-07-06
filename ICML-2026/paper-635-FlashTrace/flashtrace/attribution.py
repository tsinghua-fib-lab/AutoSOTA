import matplotlib
import matplotlib.cm as mpl_cm
from matplotlib import pyplot as plt
import numpy as np
import torch

if not hasattr(mpl_cm, "register_cmap"):
    from matplotlib import colors as _mpl_colors

    def _register_cmap(name=None, cmap=None, data=None, lut=None, *, force=False):
        """Compatibility wrapper for Matplotlib >=3.10 where register_cmap moved."""
        if cmap is not None and data is not None:
            raise ValueError("Cannot specify both `cmap` and `data` when registering a colormap.")
        if data is not None:
            if name is None:
                raise ValueError("Must supply a name when registering colormap data.")
            cmap = _mpl_colors.LinearSegmentedColormap(name, data, lut=lut)
        elif cmap is None:
            raise ValueError("Must supply `cmap` or `data` when registering a colormap.")

        if isinstance(cmap, str):
            cmap = mpl_cm.get_cmap(cmap)

        name = name or cmap.name
        copied = cmap.copy()
        copied.name = name
        mpl_cm._colormaps.register(copied, name=name, force=force)

    def _unregister_cmap(name):
        mpl_cm._colormaps.unregister(name)

    mpl_cm.register_cmap = _register_cmap
    mpl_cm.unregister_cmap = _unregister_cmap

import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple, Sequence
import textwrap
from transformers import LongformerTokenizer, LongformerForMaskedLM
import networkx as nx
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from wordfreq import zipf_frequency

from dataclasses import dataclass
from typing import Literal

from .core import (
    IFRParameters,
    ModelMetadata,
    attach_hooks,
    build_weight_pack,
    compute_ifr_for_all_positions,
    compute_ifr_sentence_aggregate,
    compute_multi_hop_ifr,
    extract_model_metadata,
)
from .qwen35 import build_layer_inputs, is_hybrid_stack


@dataclass
class AttnLRPSpanAggregate:
    """Span-wise aggregated AttnLRP result (single vector), analogous to IFRAggregate.

    This dataclass stores the result of span-wise AttnLRP aggregation computed
    in O(N) time using a single forward + backward pass.

    Attributes
    ----------
    token_importance_total : torch.Tensor
        1D float32 CPU tensor of length (user_prompt_len + gen_len) after
        chat-template stripping, aligned with `all_tokens`.
    all_tokens : List[str]
        All tokens (user prompt + generation)
    user_prompt_tokens : List[str]
        User prompt tokens only
    generation_tokens : List[str]
        Generation tokens only
    sink_range : Tuple[int, int]
        [sink_start, sink_end] in generation-token indices
    sink_weights : Optional[torch.Tensor]
        Weights used for aggregation (if any)
    metadata : Dict[str, Any]
        Additional metadata about the computation
    """
    token_importance_total: torch.Tensor
    all_tokens: List[str]
    user_prompt_tokens: List[str]
    generation_tokens: List[str]
    sink_range: Tuple[int, int]
    sink_weights: Optional[torch.Tensor]
    metadata: Dict[str, Any]


@dataclass
class MultiHopAttnLRPResult:
    """Multi-hop AttnLRP attribution result, analogous to MultiHopIFRResult.

    This dataclass stores the result of multi-hop AttnLRP computation where
    attribution is recursively propagated from output → thinking → input.

    Attributes
    ----------
    raw_attributions : List[AttnLRPSpanAggregate]
        List of per-hop attribution results. Index 0 is the base (output→all),
        subsequent indices are hop 1, 2, etc. (thinking→all with weights).
    thinking_ratios : List[float]
        Fraction of attribution mass on the thinking span at each hop.
        Useful for understanding how much attribution "stays" in reasoning.
    observation : Dict[str, torch.Tensor]
        Dictionary containing:
        - "mask": observation mask (1 for observable tokens, 0 for thinking/sink)
        - "base": base attribution masked to observable tokens
        - "per_hop": list of per-hop attributions masked to observable tokens
        - "sum": cumulative sum of all per-hop observations
        - "avg": average of per-hop observations
    """
    raw_attributions: List[AttnLRPSpanAggregate]
    thinking_ratios: List[float]
    observation: Dict[str, torch.Tensor]


from .shared_utils import (
    DEFAULT_GENERATE_KWARGS,
    create_sentences,
    create_sentence_masks,
)

from .lrp_patches import lrp_context, detect_model_type

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

class LLMAttribution():
    def __init__(
        self, 
        model: Any, 
        tokenizer: Any, 
        generate_kwargs: Optional[Dict[str, Any]] = None,
        *,
        use_chat_template: bool = False,
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.use_chat_template = bool(use_chat_template)

        self.prompt = None
        self.prompt_ids = None
        self.prompt_tokens = None
        self.chat_prompt_indices = None

        self.user_prompt = None
        self.user_prompt_ids = None
        self.user_prompt_tokens = None
        self.user_prompt_indices = None

        self.generation = None
        self.generation_ids = None
        self.generation_tokens = None

        self.model.eval()
    
    def decode_text_into_tokens(self, text) -> list[str]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        text_tokens = []
        offsets = list(offsets)
        for i in range(len(ids)):
            span = offsets.pop(0)
            start, end = span
            actual_text = text[start:end]
            text_tokens.append(actual_text)

        return text_tokens

    def extract_user_prompt_attributions(self, input, attribution) -> list[str]:   
        # Extract all attributions to be kept (gen -> user prompt and gen -> gen attributions)
        user_prompt_attr_idx = torch.tensor(self.user_prompt_indices)
        gen_attr_idx = torch.arange(len(input), attribution.shape[1])
        all_keep_idx = torch.cat((user_prompt_attr_idx, gen_attr_idx), dim = 0)

        return attribution[:, all_keep_idx]
    
    # Takes a torch tensor of size [N, M] and extends it to [N, target_length] with a padding value
    def pad_vector(self, vector, target_length, padding_value = 0) -> torch.Tensor:
        current_length = vector.shape[1]
        if current_length >= target_length:
            return vector
        padding_size = target_length - current_length
        padded_vector = F.pad(vector, (0, padding_size), value=padding_value)
        return padded_vector
    
    def format_prompt(self, prompt) -> str:
        if not self.use_chat_template:
            return prompt

        formatted_prompt = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        return formatted_prompt

    def _locate_user_prompt_indices(self) -> list[int]:
        prompt_ids = [int(token_id) for token_id in self.prompt_ids[0].detach().cpu().tolist()]
        user_prompt_ids = [int(token_id) for token_id in self.user_prompt_ids[0].detach().cpu().tolist()]
        if not user_prompt_ids:
            raise ValueError("User prompt token subsequence is empty.")

        window_len = len(user_prompt_ids)
        max_start = len(prompt_ids) - window_len
        for start in range(max_start + 1):
            if prompt_ids[start : start + window_len] == user_prompt_ids:
                return list(range(start, start + window_len))

        raise ValueError(
            "User prompt token subsequence was not found in the formatted prompt."
        )

    # Query the model for its generation
    # This internally saves the input and generated token ids for attribution target
    def response(self, prompt) -> str:
        self.user_prompt = prompt
        self.prompt = self.format_prompt(self.user_prompt)

        # these are the ids for the user supplied prompt
        self.user_prompt_ids = self.tokenizer(self.user_prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids
        # this is the tokenization of the chat prompt
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids

        with torch.no_grad():
            outputs = self.model.generate(self.prompt_ids, **self.generate_kwargs) # [1, num_prompt_tokens + num_generations]

        # Get only the generated tokens (excluding the prompt)
        self.generation_ids = outputs[:, self.prompt_ids.shape[1]:] # [1, num_generations]
        self.generation = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = True)
        gen_with_eos = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = False, clean_up_tokenization_spaces = False)

        # Track the exact user prompt subsequence inside the formatted prompt.
        m = len(self.prompt_ids[0])
        self.user_prompt_indices = self._locate_user_prompt_indices()

        # make a list of indices which are all prompt tokens 
        # (chat prompt formatting) that are not the user prompt tokens
        self.chat_prompt_indices = [idx for idx in range(0, m) if idx < self.user_prompt_indices[0] or idx > self.user_prompt_indices[-1]]

        # get the full prompt, user prompt, and generation as tokenized words
        self.prompt_tokens = self.decode_text_into_tokens(self.prompt)
        # print(self.prompt_tokens)
        self.user_prompt_tokens = self.decode_text_into_tokens(self.user_prompt)
        # print(self.user_prompt_tokens)
        self.generation_tokens = self.decode_text_into_tokens(gen_with_eos)
        # print(self.generation_tokens)
    
        return self.generation
    
    # nearly identical to response(), but we do not actually query the model
    # we assume generation = target, and generate all the class variables as done in response()
    def _strip_single_trailing_eos_text(self, target: str) -> str:
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and target.endswith(eos):
            return target[: -len(eos)]
        return target

    def target_response(self, prompt, target) -> str:
        self.user_prompt = prompt
        self.prompt = self.format_prompt(self.user_prompt)

        # these are the ids for the user supplied prompt
        self.user_prompt_ids = self.tokenizer(self.user_prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids
        # this is the tokenization of the chat prompt
        self.prompt_ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids # [1, num_prompt_tokens]
        target = self._strip_single_trailing_eos_text(target)
        # Tokenize the target generation and append EOS by ID so the target text is
        # never merged with the EOS token by tokenizer pre-tokenization.
        self.generation_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens = False).to(self.device).input_ids # [1, num_generations]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_id = torch.tensor([[int(eos_token_id)]], dtype=self.generation_ids.dtype, device=self.generation_ids.device)
            self.generation_ids = torch.cat([self.generation_ids, eos_id], dim=1)
        self.generation = target
        gen_with_eos = self.tokenizer.decode(self.generation_ids[0], skip_special_tokens = False, clean_up_tokenization_spaces = False)

        # Track the exact user prompt subsequence inside the formatted prompt.
        m = len(self.prompt_ids[0])
        self.user_prompt_indices = self._locate_user_prompt_indices()

        # make a list of indices which are all prompt tokens 
        # (chat prompt formatting) that are not the user prompt tokens
        self.chat_prompt_indices = [idx for idx in range(0, m) if idx < self.user_prompt_indices[0] or idx > self.user_prompt_indices[-1]]

        # get the full prompt, user prompt, and generation as tokenized words
        self.prompt_tokens = self.decode_text_into_tokens(self.prompt)
        self.user_prompt_tokens = self.decode_text_into_tokens(self.user_prompt)
        self.generation_tokens = self.decode_text_into_tokens(gen_with_eos)

        return self.generation

class LLMAttributionResult():
    def __init__(
        self,
        tokenizer: Any, 
        attribution_matrix: torch.Tensor, 
        prompt_tokens: list[str],
        generation_tokens: list[str],
        all_tokens: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.tokenizer = tokenizer

        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.all_tokens = all_tokens
        if self.all_tokens is not None:
            self.all_tokens = self.all_tokens

        self.attribution_matrix = attribution_matrix.detach().cpu()
        self.metadata = metadata

    # normalize rows of a matrix to sum to 1
    def normalize_sum_to_one(self, attriubtion_matrix) -> torch.Tensor:
        # we use nans for visualization, but they must be removed (set to 0) for this function
        attribution_no_nan = torch.nan_to_num(attriubtion_matrix, nan=0.0)
        # we do not want to include negative attributions, clamp all to 0
        attribution_no_nan = attribution_no_nan.clamp(0)
        # first, normalize the rows of the attribution matrix to sum to one
        attribution_row_sums = attribution_no_nan.sum(1, keepdim=True) + 1e-8
        # perform normalization
        return attribution_no_nan / attribution_row_sums
    
    def remove_nan(self, attriubtion_matrix) -> torch.Tensor:
        # we use nans for visualization, but they must be removed (set to 0) for this function
        attribution_no_nan = torch.nan_to_num(attriubtion_matrix, nan=0.0)
        # we do not want to include negative attributions, clamp all to 0
        attribution_no_nan = attribution_no_nan.clamp(0)
        return attribution_no_nan

    # normalize the max of a vector to 1
    def normalize_max(self, attribution_vector) -> torch.Tensor:
        if attribution_vector.max() > 0:
            attribution_vector = attribution_vector / attribution_vector.max()
        elif attribution_vector.max() <= 0:
            attribution_vector = - attribution_vector / attribution_vector.min()
        
        return attribution_vector

    ########################################## token attr ##########################################

    def compute_CAGE_token_attr(
        self,
        token_to_explain: int,
        *,
        clear_values: bool = True,
        token_attr_matrix_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token-level CAGE-style recursive attribution over a token attribution matrix.

        token_to_explain is a generation-token index in [0, G).
        """
        attr_norm = (
            token_attr_matrix_norm
            if token_attr_matrix_norm is not None
            else self.normalize_sum_to_one(self.attribution_matrix)
        )

        prompt_len = len(self.prompt_tokens)
        gen_len = len(self.generation_tokens)
        expected_cols = prompt_len + gen_len
        if attr_norm.ndim != 2 or attr_norm.shape[0] != gen_len or attr_norm.shape[1] != expected_cols:
            raise TypeError(
                "Expected token attribution matrix of shape [G, P+G] where "
                f"G={gen_len} and P={prompt_len}, got {tuple(attr_norm.shape)}."
            )
        if token_to_explain < 0 or token_to_explain >= gen_len:
            raise IndexError(f"token_to_explain out of range: {token_to_explain} not in [0, {gen_len}).")

        r = attr_norm[token_to_explain, :].clone()
        for k in range(token_to_explain - 1, -1, -1):
            alpha = r[prompt_len + k]
            if alpha != 0:
                r += attr_norm[k, :] * alpha
            if clear_values:
                r[prompt_len + k] = 0
        return r

    def get_all_token_attrs(self, indices_to_explain: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return token-level (seq, row, rec) attributions.

        indices_to_explain must be a generation-token span [start_tok, end_tok]
        (closed interval), typically pointing to the \\box{...} inner answer span.
        """
        if not isinstance(indices_to_explain, list) or len(indices_to_explain) != 2:
            raise ValueError(
                "indices_to_explain must be a token span [start_tok, end_tok] "
                f"(got {indices_to_explain!r})."
            )
        start_tok, end_tok = indices_to_explain
        if not isinstance(start_tok, int) or not isinstance(end_tok, int):
            raise TypeError(f"indices_to_explain elements must be ints (got {indices_to_explain!r}).")
        if start_tok < 0 or end_tok < 0 or start_tok > end_tok:
            raise ValueError(f"Invalid token span: {indices_to_explain!r}.")

        attr_norm = self.normalize_sum_to_one(self.attribution_matrix)
        gen_len = int(attr_norm.shape[0])
        if end_tok >= gen_len:
            raise IndexError(f"end_tok out of range: {end_tok} >= G={gen_len}.")

        seq_attr = attr_norm
        row_attr = attr_norm[start_tok : end_tok + 1, :].sum(dim=0, keepdim=True)

        rec_sum = torch.zeros_like(row_attr)
        for t in range(start_tok, end_tok + 1):
            rec_sum += self.compute_CAGE_token_attr(
                t,
                clear_values=True,
                token_attr_matrix_norm=attr_norm,
            ).reshape(1, -1)
        rec_attr = rec_sum

        return seq_attr, row_attr, rec_attr

    ########################################## sentence attr ##########################################

    # This converts any token attribution to a sentence attribution
    def compute_sentence_attr(self, norm = True) -> None:
        raise RuntimeError("Sentence-level aggregation has been removed; use token-level get_all_token_attrs().")

        # Legacy implementation (kept for reference)
        # create the prompt ang generation sentences
        self.prompt_sentences = create_sentences("".join(self.prompt_tokens), self.tokenizer)
        self.generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        self.all_sentences = self.prompt_sentences + self.generation_sentences

        # create a mask that tracks the tokens used in each sentence of the generation
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, self.generation_sentences)
        # create a mask that tracks the tokens used in each sentence of the prompt and the generation
        sentence_masks_all = create_sentence_masks(self.prompt_tokens + self.generation_tokens, self.all_sentences)

        num_inp_sent = len(self.prompt_sentences)
        num_gen_sent = len(self.generation_sentences)
        num_all_sent = len(self.all_sentences)

        # Now we want to turn our attribution which is over tokens into an attribution over sentences
        # attribution rows = gen sentences
        # attribution columns = prompt sentences + gen sentences
        self.sentence_attr = torch.full((num_gen_sent, num_all_sent), torch.nan)
        for i in range(num_gen_sent):
            # Select the rows (sentence) of the matrix which are attributed to the inputs (cols)
            # A whole sentence is selected at once
            row_indices = torch.where(sentence_masks_generation[i] == 1)[0]
            attr_rows = self.attribution_matrix[row_indices, :]

            for j in range(num_all_sent):
                # we do not attribute a generation to itself or any
                # future generations so we can skip those here 
                if j > i + num_inp_sent - 1:
                    continue

                # now we select the columns
                col_indices = torch.where(sentence_masks_all[j] == 1)[0]

                # now select a whole sentence of cols from these rows
                attr = attr_rows[:, col_indices]

                # Find which of these indices are NaN
                nan_mask = torch.isnan(attr)
                # Replace NaNs with 0
                attr[nan_mask] = 0.0

                # take sum of this 2d attr and place it in the correct 
                # spot of the sentence attribution
                self.sentence_attr[i, j] = torch.sum(attr)
        
        if norm:
            self.sentence_attr = self.normalize_sum_to_one(self.sentence_attr)
        else:
            self.sentence_attr = self.remove_nan(self.sentence_attr)

        return

    def plot_attr_table_sentence(self, height = None) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        width = 15
        wrapped_sentences_x = []
        for sentence in self.all_sentences:
            wrapped_sentences_x.append(textwrap.fill(sentence, width=width))
        wrapped_sentences_y = []
        max_num_lines = 0
        for sentence in self.generation_sentences:
            sentence = textwrap.fill(sentence, width=width)
            num_lines = len(sentence.split("\n"))
            max_num_lines = num_lines if num_lines > max_num_lines else max_num_lines
            wrapped_sentences_y.append(sentence)


        fig_width = (len(self.all_sentences) * width / 10) if (len(self.all_sentences) * width / 10) > 10 else 10
        if height is None:
            fig_height = (len(self.generation_sentences) * max_num_lines / 8) if (len(self.generation_sentences) * max_num_lines / 8) > 8 else 8
        else:
            fig_height = 5
            
        fig, axs = plt.subplots(1, 1, figsize = (fig_width, fig_height))

        # use a positive only heatmap cmap
        if np.nanmin(self.sentence_attr) >= 0:
            sns.heatmap(self.sentence_attr, annot=False, xticklabels=wrapped_sentences_x, yticklabels=wrapped_sentences_y, cmap="Blues", ax = axs)
        # use a postitive and negative heatmap cmap
        else:
            # set vmax vmin such that 0 is center value of color map
            max_abs_attr_val = np.nanmax(self.sentence_attr.abs())
            sns.heatmap(self.sentence_attr, annot=False, xticklabels=wrapped_sentences_x, yticklabels=wrapped_sentences_y, vmax=max_abs_attr_val, vmin=-max_abs_attr_val, cmap="Blues", ax = axs)

        axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=200 / len(self.all_sentences))
        plt.yticks(rotation=45) 
        plt.xticks(rotation=45) 
        plt.show()

    def plot_context_attr_sentence(self, title) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        wrapped_sentences = []
        width = 20
        for sentence in self.prompt_sentences:
            wrapped_sentences.append(textwrap.fill(sentence, width=width))

        fig_width = len(wrapped_sentences) * (width / 10) 
        fig_height = len(wrapped_sentences) / 2 if len(wrapped_sentences) / 2 > 3 else 3

        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(np.arange(len(wrapped_sentences)), self.normalize_max(torch.nansum(self.sentence_attr[:, :len(self.prompt_sentences)].cpu().detach(), dim = 0)))
        plt.xticks(range(len(wrapped_sentences)), wrapped_sentences, rotation=0)
        plt.ylabel("Influence")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def save_context_attr_sentence(self, prompt_sentences, path) -> None:
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return
        
        wrapped_sentences = []
        width = 20
        for sentence in prompt_sentences:
            wrapped_sentences.append(textwrap.fill(sentence, width=width))

        fig_width = len(wrapped_sentences) * (width / 10) 
        fig_height = len(wrapped_sentences) / 2 if len(wrapped_sentences) / 2 > 3 else 3

        fig, axs = plt.subplots(1, 1, figsize = (fig_width, fig_height))
        plt.bar(np.arange(len(wrapped_sentences)), self.normalize_max(torch.nansum(self.sentence_attr[:, :len(prompt_sentences)].cpu().detach(), dim = 0)))
        plt.xticks(range(len(wrapped_sentences)), wrapped_sentences, rotation=0)
        plt.ylabel("Influence")
        plt.tight_layout()
        plt.savefig(path + ".png", bbox_inches='tight', transparent = "False")
        fig.clear()
        plt.close(fig)


    def draw_graph(self, cmap = plt.cm.Blues, wrap_width=20, thresh = 0.10, spacing = 4, arrow_mod = 1, rad = 0.3):
        """
        Simplified one-row attribution graph:
        - All tokens (prompts + generations) drawn in one horizontal row
        - Directed weighted edges: generation -> input
        """

        grad_array = self.sentence_attr
        outputs = self.all_sentences
        generated = self.generation_sentences

        grad_array = grad_array.permute((1, 0))  # -> [outputs, generated]
        attr_np = grad_array.cpu().numpy() if hasattr(grad_array, "cpu") else grad_array
        attr_np = np.nan_to_num(attr_np, nan=0.0)

        G = nx.DiGraph()
        prompt_len = len(outputs) - len(generated)
        n_gen = len(generated)

        # Node ids
        prompt_ids = [f"p_{i}" for i in range(prompt_len)]
        gen_ids = [f"g_{j}" for j in range(n_gen)]

        # Add nodes
        def add_node(node_id, label, ntype):
            wrapped = textwrap.fill(label, width=wrap_width)
            wrap_height = len(wrapped.split('\n'))
            G.add_node(node_id, label=wrapped, type=ntype)
            return wrap_height

        max_wrap_height = 0
        for i in range(prompt_len):
            wrap_height = add_node(prompt_ids[i], outputs[i], "prompt")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height
        for j in range(n_gen):
            wrap_height = add_node(gen_ids[j], generated[j], "generated")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height

        def out_i_to_node(i):
            return prompt_ids[i] if i < prompt_len else gen_ids[i - prompt_len]

        # Add edges gen -> output
        for j in range(n_gen):
            src = gen_ids[j]
            for i in range(len(outputs)):
                w = attr_np[i, j] if (i < attr_np.shape[0] and j < attr_np.shape[1]) else 0.0
                if w != 0.0:
                    G.add_edge(src, out_i_to_node(i), weight=w)


        # --- layout: single row ---
        y_row = 0.0
        pos = {}
        all_nodes = prompt_ids + gen_ids
        for idx, nid in enumerate(all_nodes):
            pos[nid] = (idx * spacing, y_row)

        # --- figure ---
        ncols = len(all_nodes)
        fig_width = max(10, ncols * (spacing * 0.6))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi = 300)

        # prune edges
        edges = list(G.edges(data=True))
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size > 0:
            threshold = thresh * np.max(np.abs(weights))  # keep edges ≥ 5% of max weight
            for (u, v, edata) in list(edges):  # iterate over a copy
                if abs(edata["weight"]) < threshold:
                    G.remove_edge(u, v)

        # visualization
        edges = G.edges(data=True)  # refresh edges after pruning
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size == 0:
            weights = np.array([1])  # fallback if everything pruned
        max_w = np.max(np.abs(weights))
        norm = mpl.colors.TwoSlopeNorm(vmin=-max_w, vcenter=0, vmax=max_w) \
            if np.min(weights) < 0 else mpl.colors.Normalize(vmin=0, vmax=max_w)
        
        # Draw nodes (larger font + padding)
        for nid, (x, y) in pos.items():
            lbl = G.nodes[nid]["label"]
            ntype = G.nodes[nid]["type"]
            box_color = "#d4c1ffc8" if ntype == "prompt" else "#cfffcc" #cfe8ff
            ax.annotate(
                lbl, xy=(x, y), xytext=(x, y),
                ha="center", va="center", fontsize=12, zorder=3,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color,
                        edgecolor="gray", linewidth=1.2, alpha=1),
            )

        box_height = max_wrap_height / 4
        # Draw edges with curved arrows
        for (u, v, edata) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            start = (x1, y1 - box_height)
            end   = (x2, y2 - box_height)

            w = edata["weight"]
            color = cmap(norm(w))
            width = (1.5 * arrow_mod) + 5.0 * (abs(w) / max_w)

            arrow_rad = rad if x1 <= x2 else -rad
            arrow = FancyArrowPatch(
                (start), (end),
                connectionstyle=f"arc3,rad={arrow_rad}",
                # arrowstyle=f"-|>,head_length={2*arrow_mod},head_width={arrow_mod}",
                arrowstyle=f"<|-,head_length={2*arrow_mod},head_width={arrow_mod}",
                linewidth=width, color=color, alpha=1, zorder=2,
                shrinkA=16, shrinkB=16, mutation_scale=12,
                clip_on=False
            )
            ax.add_patch(arrow)

        ax.set_xlim(-spacing, (ncols - 1) * spacing + spacing)
        ax.set_ylim(-3, 3)
        ax.axis("off")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


    def save_graph(self, all_sentences, generation_sentences, path, cmap = plt.cm.Blues, wrap_width=20, thresh = 0.10, spacing = 4, arrow_mod = 1, rad = 0.3):
        """
        Simplified one-row attribution graph:
        - All tokens (prompts + generations) drawn in one horizontal row
        - Directed weighted edges: generation -> input
        """

        grad_array = self.sentence_attr
        outputs = all_sentences
        generated = generation_sentences

        grad_array = grad_array.permute((1, 0))  # -> [outputs, generated]
        attr_np = grad_array.cpu().numpy() if hasattr(grad_array, "cpu") else grad_array
        attr_np = np.nan_to_num(attr_np, nan=0.0)

        G = nx.DiGraph()
        prompt_len = len(outputs) - len(generated)
        n_gen = len(generated)

        # Node ids
        prompt_ids = [f"p_{i}" for i in range(prompt_len)]
        gen_ids = [f"g_{j}" for j in range(n_gen)]

        # Add nodes
        def add_node(node_id, label, ntype):
            wrapped = textwrap.fill(label, width=wrap_width)
            wrap_height = len(wrapped.split('\n'))
            G.add_node(node_id, label=wrapped, type=ntype)
            return wrap_height

        max_wrap_height = 0
        for i in range(prompt_len):
            wrap_height = add_node(prompt_ids[i], outputs[i], "prompt")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height
        for j in range(n_gen):
            wrap_height = add_node(gen_ids[j], generated[j], "generated")
            if wrap_height > max_wrap_height:
                max_wrap_height = wrap_height

        def out_i_to_node(i):
            return prompt_ids[i] if i < prompt_len else gen_ids[i - prompt_len]

        # Add edges gen -> output
        for j in range(n_gen):
            src = gen_ids[j]
            for i in range(len(outputs)):
                w = attr_np[i, j] if (i < attr_np.shape[0] and j < attr_np.shape[1]) else 0.0
                if w != 0.0:
                    G.add_edge(src, out_i_to_node(i), weight=w)


        # --- layout: single row ---
        y_row = 0.0
        pos = {}
        all_nodes = prompt_ids + gen_ids
        for idx, nid in enumerate(all_nodes):
            pos[nid] = (idx * spacing, y_row)

        # --- figure ---
        ncols = len(all_nodes)
        fig_width = max(10, ncols * (spacing * 0.6))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi = 300)

        # prune edges
        edges = list(G.edges(data=True))
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size > 0:
            threshold = thresh * np.max(np.abs(weights))  # keep edges ≥ 5% of max weight
            for (u, v, edata) in list(edges):  # iterate over a copy
                if abs(edata["weight"]) < threshold:
                    G.remove_edge(u, v)

        # visualization
        edges = G.edges(data=True)  # refresh edges after pruning
        weights = np.array([edata["weight"] for _, _, edata in edges])
        if weights.size == 0:
            weights = np.array([1])  # fallback if everything pruned
        max_w = np.max(np.abs(weights))
        norm = mpl.colors.TwoSlopeNorm(vmin=-max_w, vcenter=0, vmax=max_w) \
            if np.min(weights) < 0 else mpl.colors.Normalize(vmin=0, vmax=max_w)
        
        # Draw nodes (larger font + padding)
        for nid, (x, y) in pos.items():
            lbl = G.nodes[nid]["label"]
            ntype = G.nodes[nid]["type"]
            box_color = "#d4c1ffc8" if ntype == "prompt" else "#cfffcc" #cfe8ff
            ax.annotate(
                lbl, xy=(x, y), xytext=(x, y),
                ha="center", va="center", fontsize=12, zorder=3,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color,
                        edgecolor="gray", linewidth=1.2, alpha=1),
            )

        box_height = max_wrap_height / 4
        # Draw edges with curved arrows
        for (u, v, edata) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            start = (x1, y1 - box_height)
            end   = (x2, y2 - box_height)

            w = edata["weight"]
            color = cmap(norm(w))
            width = (1.5 * arrow_mod) + 5.0 * (abs(w) / max_w)

            arrow_rad = rad if x1 <= x2 else -rad
            arrow = FancyArrowPatch(
                (start), (end),
                connectionstyle=f"arc3,rad={arrow_rad}",
                # arrowstyle=f"-|>,head_length={2*arrow_mod},head_width={arrow_mod}",
                arrowstyle=f"<|-,head_length={2*arrow_mod},head_width={arrow_mod}",
                linewidth=width, color=color, alpha=1, zorder=2,
                shrinkA=16, shrinkB=16, mutation_scale=12,
                clip_on=False
            )
            ax.add_patch(arrow)

        ax.set_xlim(-spacing, (ncols - 1) * spacing + spacing)
        ax.set_ylim(-3, 3)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(path + ".png", dpi=500, transparent = "False")
        fig.clear()
        plt.close(fig)



    ########################################## recursive sentence attr ##########################################

    # this function is identical to compute_recursive_attr except for var names
    # see that function for details
    def compute_CAGE_sentence_attr(self, sentence_to_explain = -1, clear_values = True) -> None:
        raise RuntimeError("Sentence-level CAGE has been removed; use token-level compute_CAGE_token_attr().")

        # Legacy implementation (kept for reference)
        if self.sentence_attr is None:
            print(
                '''The sentence attribution has not been computed.
                Call LLMAttributionResult.compute_sentence_attr() first.
                '''
            )
            return

        if self.sentence_attr.shape[1] != len(self.all_sentences):
            raise TypeError(
                """This attribution object is of shape [generations, prompt]. 
                This function only operates on attributions of shape 
                [generations, prompt + generations]"""
            )
        
        self.CAGE_sentence_attr = self.sentence_attr[sentence_to_explain].clone()
        gen_row_indices_to_collapse = list(range(0, len(self.generation_sentences[:sentence_to_explain])))[::-1]
        prompt_sentences_length = len(self.prompt_sentences)
        for index in gen_row_indices_to_collapse:
            biased_row = self.sentence_attr[index] * self.CAGE_sentence_attr[prompt_sentences_length + index]
            if clear_values:
                self.CAGE_sentence_attr[prompt_sentences_length + index] = 0
            self.CAGE_sentence_attr += biased_row

        return
        

    ########################################## Multi Sentence Attr ##########################################

    # this function returns a tuple containing a sentence attribution matrix,
    # the sum of all rows of that matrix, the sum of indices_to_explain rows of that matrix, and a CAGE attribution over the indices_to_explain
    def get_all_sentence_attrs(self, indices_to_explain) -> tuple:
        raise RuntimeError("Sentence-level attribution outputs have been removed; use get_all_token_attrs([start_tok, end_tok]).")

        # Legacy implementation (kept for reference)
        self.compute_sentence_attr(norm = True)

        attr = self.sentence_attr

        row_attr = 0
        for index in indices_to_explain:
            row_attr += attr[index, :]
        row_attr = row_attr.reshape(1, -1)

        rec_attr = 0
        for index in indices_to_explain:
            self.compute_CAGE_sentence_attr(index)
            rec_attr += self.CAGE_sentence_attr
        rec_attr = rec_attr.reshape(1, -1)

        return attr, row_attr, rec_attr
    
class LLMBasicAttribution(LLMAttribution):
    def __init__(self, model, tokenizer, language: str = "en") -> None:
        super().__init__(model, tokenizer)
        self.zipf_language = language

    def calculate_basic_attribution(self, prompt: str, target: Optional[str] = None) -> LLMAttributionResult:
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        prompt_length = len(self.user_prompt_tokens)
        generation_length = len(self.generation_tokens)
        total_length = prompt_length + generation_length

        score_array = torch.zeros((generation_length, total_length), dtype=torch.float32)

        if generation_length == 0:
            all_tokens = self.user_prompt_tokens + self.generation_tokens
            return LLMAttributionResult(
                self.tokenizer,
                score_array,
                self.user_prompt_tokens,
                self.generation_tokens,
                all_tokens=all_tokens,
            )

        if generation_length > 0 and prompt_length > 0:
            normalized_prompt_tokens = [token.strip() for token in self.user_prompt_tokens]

            for gen_idx, gen_token in enumerate(self.generation_tokens):
                normalized_gen_token = gen_token.strip()

                if not normalized_gen_token:
                    continue

                weight = float(zipf_frequency(normalized_gen_token, self.zipf_language))
                if weight <= 0.0:
                    continue

                for prompt_idx, prompt_token in enumerate(normalized_prompt_tokens):
                    if prompt_token == normalized_gen_token:
                        score_array[gen_idx, prompt_idx] = weight

            row_sums = score_array.sum(dim=1, keepdim=True)
            nonzero_rows = row_sums.squeeze(1) > 0
            if torch.any(nonzero_rows):
                score_array[nonzero_rows] = score_array[nonzero_rows] / row_sums[nonzero_rows]

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
        )


class LLMIFRAttribution(LLMAttribution):
    """Information Flow Routes attribution integrated with the LLMAttribution API."""

    def __init__(
        self,
        model,
        tokenizer,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        *,
        chunk_tokens: int = 128,
        sink_chunk_tokens: int = 32,
        renorm_threshold_default: float = 0.0,
        show_progress: bool = True,
        recompute_attention: bool = False,
        use_chat_template: bool = False,
    ) -> None:
        super().__init__(model, tokenizer, generate_kwargs, use_chat_template=use_chat_template)
        self.chunk_tokens = int(chunk_tokens)
        self.sink_chunk_tokens = int(sink_chunk_tokens)
        self.renorm_threshold_default = float(renorm_threshold_default)
        self.show_progress = bool(show_progress)
        self.recompute_attention = bool(recompute_attention)

    @property
    def _model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _ensure_generation(self, prompt: str, target: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        input_ids_all = torch.cat([self.prompt_ids, self.generation_ids], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids_all)
        return input_ids_all, attention_mask, prompt_len, gen_len

    def _capture_model_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recompute_attention: bool = False,
    ) -> Tuple[Dict[str, List[Optional[torch.Tensor]]], Optional[Sequence[torch.Tensor]], ModelMetadata, List[Dict[str, torch.Tensor | nn.Module]]]:
        metadata = extract_model_metadata(self.model)
        hybrid = is_hybrid_stack(metadata)
        # Hybrid stacks (Qwen3.5) have linear-attention layers that expose no
        # softmax weights to recompute; they always use stored attention.
        capture_attentions = hybrid or not recompute_attention
        cache, hooks = attach_hooks(metadata.layers, self._model_dtype)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=capture_attentions,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in hooks:
                try:
                    handle.remove()
                except Exception:
                    pass

        attentions = outputs.attentions if capture_attentions else None
        weight_pack = build_weight_pack(metadata, self._model_dtype)

        if hybrid:
            self._hybrid_layer_inputs = build_layer_inputs(
                metadata, cache, attentions, self._model_dtype
            )
        else:
            self._hybrid_layer_inputs = None

        return cache, attentions, metadata, weight_pack

    def _build_ifr_params(self, metadata: ModelMetadata, sequence_length: int) -> IFRParameters:
        return IFRParameters(
            n_layers=metadata.n_layers,
            n_heads_q=metadata.n_heads_q,
            n_kv_heads=metadata.n_kv_heads,
            head_dim=metadata.head_dim,
            group_size=metadata.group_size,
            d_model=metadata.d_model,
            sequence_length=sequence_length,
            model_dtype=self._model_dtype,
            chunk_tokens=self.chunk_tokens,
            sink_chunk_tokens=self.sink_chunk_tokens,
            layer_inputs=getattr(self, "_hybrid_layer_inputs", None),
        )

    def _finalize_result(self, score_array: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> LLMAttributionResult:
        if score_array.ndim == 1:
            score_array = score_array.unsqueeze(0)
        score_array = score_array.detach().cpu()

        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)
        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata=metadata,
        )

    def _project_vector(self, vector: torch.Tensor) -> torch.Tensor:
        matrix = vector.detach().cpu().view(1, -1)
        projected = self.extract_user_prompt_attributions(self.prompt_tokens, matrix)
        return projected[0]

    @torch.no_grad()
    def calculate_ifr_for_all_positions(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        renorm_threshold: Optional[float] = None,
    ) -> LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])
        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "all_positions",
                    "sink_indices": [],
                    "renorm_threshold": renorm_threshold,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        sink_range = (prompt_len, prompt_len + gen_len - 1)
        all_positions = compute_ifr_for_all_positions(
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            sink_range=sink_range,
            return_layerwise=False,
            rotary_emb=metadata.rotary_emb,
        )

        meta = {
            "ifr": {
                "type": "all_positions",
                "sink_indices": all_positions.sink_indices,
                "renorm_threshold": renorm,
            }
        }
        return self._finalize_result(all_positions.token_importance_matrix, metadata=meta)

    @torch.no_grad()
    def calculate_ifr_for_all_positions_output_only(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        renorm_threshold: Optional[float] = None,
    ) -> LLMAttributionResult:
        """Compute IFR for sink positions restricted to an output span.

        This mirrors calculate_ifr_for_all_positions but only computes per-token IFR
        rows for sink positions in sink_span (generation-token indices). All other
        generation rows are left as NaN (becoming 0 after normalization).
        """
        input_ids_all, attn_mask, prompt_len, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])
        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "all_positions_output_only",
                    "sink_span_generation": None,
                    "sink_indices": [],
                    "renorm_threshold": renorm_threshold,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        note = ""
        if sink_span is None:
            sink_span = (0, gen_len - 1)
            note = "sink_span not provided; fell back to full generation."
        span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end >= gen_len:
            raise ValueError(f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len}.")

        sink_start_abs = prompt_len + span_start
        sink_end_abs = prompt_len + span_end

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        sink_range = (sink_start_abs, sink_end_abs)
        all_positions = compute_ifr_for_all_positions(
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            sink_range=sink_range,
            return_layerwise=False,
            rotary_emb=metadata.rotary_emb,
        )

        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)
        score_array[span_start : span_end + 1, :] = all_positions.token_importance_matrix

        meta = {
            "ifr": {
                "type": "all_positions_output_only",
                "sink_span_generation": (span_start, span_end),
                "sink_span_absolute": (sink_start_abs, sink_end_abs),
                "sink_indices": all_positions.sink_indices,
                "renorm_threshold": renorm,
                "note": note,
            }
        }
        return self._finalize_result(score_array, metadata=meta)

    @torch.no_grad()
    def calculate_ifr_span(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        span: Optional[Tuple[int, int]] = None,
        renorm_threshold: Optional[float] = None,
    ) -> LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "sentence_aggregate",
                    "sink_span_generation": None,
                    "renorm_threshold": renorm_threshold,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        span_start, span_end = span if span is not None else (0, gen_len - 1)
        if span_start < 0 or span_end < span_start or span_end >= gen_len:
            raise ValueError(
                f"Invalid span ({span_start}, {span_end}) for generation length {gen_len}."
            )

        sink_start_abs = prompt_len + span_start
        sink_end_abs = prompt_len + span_end

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        aggregate = compute_ifr_sentence_aggregate(
            sink_start=sink_start_abs,
            sink_end=sink_end_abs,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            rotary_emb=metadata.rotary_emb,
        )

        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)
        for offset in range(span_start, span_end + 1):
            score_array[offset] = aggregate.token_importance_total

        meta = {
            "ifr": {
                "type": "sentence_aggregate",
                "sink_span_generation": (span_start, span_end),
                "sink_span_absolute": (sink_start_abs, sink_end_abs),
                "renorm_threshold": renorm,
                "aggregate": aggregate,
            }
        }
        return self._finalize_result(score_array, metadata=meta)

    @torch.no_grad()
    def calculate_ifr_multi_hop(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        renorm_threshold: Optional[float] = None,
        observation_mask: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> LLMAttributionResult:
        input_ids_all, attn_mask, prompt_len, gen_len = self._ensure_generation(prompt, target)
        total_len = int(input_ids_all.shape[1])

        if gen_len == 0:
            empty = torch.zeros((0, total_len), dtype=torch.float32)
            metadata = {
                "ifr": {
                    "type": "multi_hop",
                    "sink_span_generation": sink_span,
                    "thinking_span_generation": thinking_span,
                    "renorm_threshold": renorm_threshold,
                    "note": "No generation tokens; returning empty attribution matrix.",
                }
            }
            return self._finalize_result(empty, metadata=metadata)

        if sink_span is None:
            sink_span = (0, gen_len - 1)
        span_start, span_end = sink_span
        if span_start < 0 or span_end < span_start or span_end >= gen_len:
            raise ValueError(
                f"Invalid sink_span ({span_start}, {span_end}) for generation length {gen_len}."
            )
        if thinking_span is None:
            thinking_span = sink_span
        think_start, think_end = thinking_span
        if think_start < 0 or think_end < think_start or think_end >= gen_len:
            raise ValueError(
                f"Invalid thinking_span ({think_start}, {think_end}) for generation length {gen_len}."
            )

        sink_start_abs = prompt_len + span_start
        sink_end_abs = prompt_len + span_end
        think_start_abs = prompt_len + think_start
        think_end_abs = prompt_len + think_end

        obs_mask_tensor: Optional[torch.Tensor] = None
        if observation_mask is not None:
            obs_mask_tensor = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_tensor.ndim != 1:
                raise ValueError("observation_mask must be a 1D tensor or sequence.")
            if obs_mask_tensor.numel() == gen_len:
                mask_full = torch.zeros(total_len, dtype=torch.float32)
                mask_full[prompt_len : prompt_len + gen_len] = obs_mask_tensor
                obs_mask_tensor = mask_full
            elif obs_mask_tensor.numel() != total_len:
                raise ValueError(
                    f"observation_mask must have length {gen_len} (generation) or {total_len} (full sequence)."
                )

        cache, attentions, metadata, weight_pack = self._capture_model_state(
            input_ids_all, attn_mask, recompute_attention=self.recompute_attention,
        )
        params = self._build_ifr_params(metadata, total_len)
        renorm = self.renorm_threshold_default if renorm_threshold is None else float(renorm_threshold)

        multi_hop = compute_multi_hop_ifr(
            sink_start=sink_start_abs,
            sink_end=sink_end_abs,
            thinking_span=(think_start_abs, think_end_abs),
            n_hops=int(n_hops),
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm,
            observation_mask=obs_mask_tensor,
            rotary_emb=metadata.rotary_emb,
        )

        eval_vector = multi_hop.observation["sum"]
        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)
        for offset in range(span_start, span_end + 1):
            score_array[offset] = eval_vector

        projected_per_hop = [
            self._project_vector(result.token_importance_total) for result in multi_hop.raw_attributions
        ]
        obs = multi_hop.observation
        observation_projected = {
            "mask": self.extract_user_prompt_attributions(
                self.prompt_tokens, obs["mask"].view(1, -1)
            )[0],
            "base": self._project_vector(obs["base"]),
            "sum": self._project_vector(obs["sum"]),
            "avg": self._project_vector(obs["avg"]),
            "per_hop": [self._project_vector(vec) for vec in obs["per_hop"]],
        }

        meta: Dict[str, Any] = {
            "ifr": {
                "type": "multi_hop",
                "sink_span_generation": (span_start, span_end),
                "sink_span_absolute": (sink_start_abs, sink_end_abs),
                "thinking_span_generation": (think_start, think_end),
                "thinking_span_absolute": (think_start_abs, think_end_abs),
                "renorm_threshold": renorm,
                "n_hops": int(n_hops),
                "thinking_ratios": multi_hop.thinking_ratios,
                "per_hop_projected": projected_per_hop,
                "observation_projected": observation_projected,
                "raw": multi_hop,
            }
        }

        return self._finalize_result(score_array, metadata=meta)

class LLMGradientAttribtion(LLMAttribution):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    # if captum version = True, interpolation only performed over prompt tokens
    # else interpolation over prompt tokens and all intermediate generations
    def calculate_IG_per_generation(self, prompt, steps, baseline, batch_size = 1, captum_version = False, target = None) -> LLMAttributionResult:
        # run the model so we can access the input ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
                
        # Make a copy of the input ids
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input, generated, and total tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length

        # instantiate a matrix which will track the attribution of every generated token to the input
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.empty((generation_length, total_length))

        # grads must be measured to the embedding layer
        embedding_layer = self.model.get_input_embeddings()

        # check batch size
        batch_size = steps if steps < batch_size else batch_size

        # create alphas and set step size trapezoidal riemann sum integral estimation
        alphas = torch.linspace(0, 1, steps, dtype=torch.float32).to(self.device)
        step_sizes = torch.full_like((alphas), 1 / steps).to(self.device)
        step_sizes[0] /= 2
        step_sizes[-1] /= 2
        
        # this is used for precision casting in case the model is not loaded in fp32
        model_dtype = next(self.model.parameters()).dtype

        # we calculate the gradients of predicting self.generation_ids[step] 
        # by updating the input to be prompt + self.generation_ids[:step]
        # for step in tqdm(range(generation_length)):
        for step in range(generation_length):
            # take inputs off of the graph to avoid gradient accumulation across steps
            input_ids_all = input_ids_all.detach()

            # Capture the input embeddings and force require grad
            input_embeds_orig = embedding_layer(input_ids_all).float()
            # The baseline value is a token id. Commonly employed as 0 (for llama that is the token '!')
            # also used is tokenizer.eos_token_id or tokenizer.pad_token_id
            baseline_embeds = embedding_layer(torch.full_like(input_ids_all, baseline)).float()

            # set target as next known generated token
            target_token = self.generation_ids[0, step].item()
   
            # # Make a tensor to store the gradients over all IG steps 
            # # each individual gradient will be [batch_size, seq_len, embedding_dim]
            # IG_grads = torch.zeros((steps, input_embeds_orig.shape[1], input_embeds_orig.shape[2])).to(self.device)

            # Make a tensor to store the sum of the gradients across the IG steps 
            IG_sum = torch.zeros(input_embeds_orig.shape[1], input_embeds_orig.shape[2], device=self.device)

            # perform IG (gradients of interpolated inputs)
            for batch_start in range(0, steps, batch_size):
                # grab a batch of alphas and step sizes
                batch_end = min(batch_start + batch_size, steps)
                alphas_batch = alphas[batch_start : batch_end].view(-1, 1, 1).float()
                step_sizes_batch = step_sizes[batch_start : batch_end].view(-1, 1, 1)

                # interpolate the batch of embeddings
                # captum does not interpolate over the current generated tokens
                # as a result, the generation -> generation gradients are mostly ignored
                if captum_version == True:
                    scaled_embeds_batch = baseline_embeds[:, :input_length] + alphas_batch * (input_embeds_orig[:, :input_length] - baseline_embeds[:, :input_length]) 
                    input_embeds_batch = input_embeds_orig.detach().clone().repeat(batch_end - batch_start, 1, 1)
                    input_embeds_batch[:, :input_length] = scaled_embeds_batch          
                # We do interpolate over the prompt and current generation
                # This allows generation -> generation attributions to be captured
                else:
                    input_embeds_batch = baseline_embeds + alphas_batch * (input_embeds_orig - baseline_embeds) # [batch_size, seq_len, embedding_dim]

                # set requires grad on input embeds
                input_embeds_batch = input_embeds_batch.to(model_dtype).detach().clone().requires_grad_(True)
                # perform inference
                logits = self.model(inputs_embeds=input_embeds_batch).logits # [batch_size, seq_len, vocab_size]
                # evaluate the probability of the target token's generation
                probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1) # [batch_size, vocab_size]
                losses = probs[:, target_token] # [batch_size]

                # clear grads 
                self.model.zero_grad(set_to_none=True)
                if input_embeds_batch.grad is not None:
                    input_embeds_batch.grad.zero_()

                # gather the gradients wrt these probabilities across batch
                losses.sum().backward()

                # perform (x - x') * grad * step_size 
                # baseline_diff = (input_embeds_orig - baseline_embeds) 
                # IG_grads[batch_start : batch_end] = baseline_diff * input_embeds_batch.grad.detach().clone() * step_sizes_batch # [batch_size, seq_len, embedding_dim]

                # perform (x - x') * grad * step_size 
                baseline_diff = (input_embeds_orig - baseline_embeds) 
                grads_batch = baseline_diff * input_embeds_batch.grad.detach().clone() * step_sizes_batch# [batch_size, seq_len, embedding_dim]
                # Sum over batch
                IG_sum += (grads_batch).sum(dim=0) # [seq_len, embedding_dim]

                # Free memory
                del input_embeds_batch, logits, probs, grads_batch
                torch.cuda.empty_cache()

                # del input_embeds_batch, logits, probs, losses

            # # This is a sum over the number of IG steps. Finishes IG result    
            # IG_grads = IG_grads.sum(0) # From [steps, seq_len, embed_dim] to [seq_len, embed_dim]
            # # take the sum over the embedding_dim
            # IG_grads = IG_grads.sum(-1) # [seq_len]

            # Sum across embedding dimension
            IG_grads = IG_sum.sum(-1).detach().cpu() 

            # pad these grads with nan since they must fit into score_array with all other token attributions
            score_array[step] = self.pad_vector(IG_grads.view(1, -1), total_length, torch.nan) # [1, total_length]

            # clean up before the next loop
            # del input_embeds_batch, logits, probs, losses
            # torch.cuda.empty_cache()

            # Append next token to input for next step generation and attribution
            input_ids_all = torch.cat([input_ids_all, torch.tensor([[target_token]]).to(self.device)], dim=1)
            input_ids_all = input_ids_all.detach().clone()

        # remove from the attribution all values associated with thechat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)

class LLMPerturbationAttribution(LLMAttribution):

    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)

        self.mlm_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.mlm_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(self.device)



    #  we want to evaluate the probability of producing a reponse given a prompt
    def compute_logprob_response_given_prompt(self, prompt_ids, response_ids) -> torch.Tensor:
        """
        Compute log-probabilities of `response_ids` given `prompt_ids`.

        prompt_ids: [B, N]
        response_ids: [B, M]
        Returns: [B, M]
        """
        # concat prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)   # [B, N+M]
        attention_mask = torch.ones_like(input_ids)

        # Get model outputs
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, seq_len, vocab_size]

        # Compute log-probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

        # Only consider response tokens
        response_start = prompt_ids.shape[1]

        # Align logits to predict each y_t from y_{<t}
        logits_for_response = log_probs[:, response_start - 1: -1, :]  # [B, M, vocab]

        # Gather log-probs for the actual response tokens
        gathered = logits_for_response.gather(2, response_ids.unsqueeze(-1))  # [B, M, 1]

        return gathered.squeeze(-1)  # [B, M]


    def compute_kl_response_given_prompt(self, prompt_ids, response_ids) -> torch.Tensor:
        """
        Compute KL divergence scores for each token in `response_ids` given `prompt_ids`.
        Mimics run_probing(metrics="kl_div") but only for the full sequence.

        Args:
            model: HuggingFace autoregressive model.
            prompt_ids: [B, N] tensor of prompt token IDs.
            response_ids: [B, M] tensor of response token IDs.

        Returns:
            KL-divergence scores: [B, M] tensor.
        """
        device = prompt_ids.device
        prompt_ids = prompt_ids.to(device)
        response_ids = response_ids.to(device)

        # Concatenate prompt + response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, N+M]
        attention_mask = torch.ones_like(input_ids, device=device)

        # Compute logits
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, N+M, V]
        logits = logits.to(torch.float32)  # avoid float16 overflow
        log_probs = F.log_softmax(logits, dim=-1)  # [B, N+M, V]

        # Align: y_t predicted from x_{<t}
        B, N = prompt_ids.shape
        M = response_ids.shape[1]
        response_positions = torch.arange(N, N + M, device=device)
        log_probs_response = log_probs[:, response_positions - 1, :]  # [B, M, V]

        # Gather log-probs for actual response tokens
        log_p = log_probs_response.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # [B, M]

        # KL-divergence: assume uniform / null-context q (like run_probing)
        log_p_minus_log_q = -log_probs_response + log_p.unsqueeze(-1)  # [B, M, V]
        p = log_p.exp()  # [B, M]

        kl_scores = (log_p_minus_log_q * p.unsqueeze(-1)).sum(dim=-1)  # [B, M]

        # print(self.tokenizer.decode(response_ids[0]))
        # print(kl_scores)

        return kl_scores


    def calculate_feature_ablation_sentences(self, prompt, baseline, measure = "log_loss", target = None) -> LLMAttributionResult:   
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the prompt ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input tokens and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length


        # given the text user prompt, create a mask over the tokens of each sentence
        user_prompt_sentences = create_sentences("".join(self.user_prompt_tokens), self.tokenizer, show=True)
        sentence_masks_prompt = create_sentence_masks(self.user_prompt_tokens, user_prompt_sentences, show=True)

        # mask prompt sentences and generated sentences
        # given the generation, create a mask over the tokens of each sentence
        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        # find the total sizes of the masks we need
        l    = len(self.chat_prompt_indices)     # input formating tokens
        n, m = sentence_masks_prompt.shape       # (user prompt sentences, user prompt tokens)
        o, p = sentence_masks_generation.shape   # (generation sentences + EOS, generation tokens + EOS)

        # Create a tensor that can fit all masks diagonally
        masks = torch.zeros((l + n + o, l + m + p))

        # we never mask the chat_prompt_indices, leave as 0
        # prompt indices cover:
        #   0 : start of sentence_masks_prompt
        #   end of sentence_masks_prompt : start of sentence_masks_generation

        # input sentence masks only cover the user prompt
        user_prompt_start_idx = self.user_prompt_indices[0]
        masks[user_prompt_start_idx : user_prompt_start_idx + n, user_prompt_start_idx : user_prompt_start_idx + m] = sentence_masks_prompt

        # gen sentence masks only cover the generations
        masks[l + n:, l + m:] = sentence_masks_generation
        
        num_input_masks = masks.shape[0]

        # instantiate a matrix which will track the attribution of every generated token to intermediate generations
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.full((generation_length, total_length), torch.nan)

        for step in range(len(sentence_masks_generation)):
        # for step in range(len(sentence_masks_generation) + 1):
            input_ids_all = input_ids_all.detach()

            # assume the we are generating a sentence of the generation_ids and we find the
            # prob of generating this sentence from the current input_ids (prompt + any current generations)
            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0] # [len(target_tokens)]
            gen_tokens = self.generation_ids[:, gen_token_indices] # [1, len(target_tokens)]

            if measure == "log_loss":
                original_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]
            elif measure == "KL":
                original_probs = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

            # perturb each sentence of the input and current generation 
            # and measure how the probs of predicitng gen_tokens changes
            for i in range(num_input_masks - len(sentence_masks_generation) + step):
                # find the input tokens to be masked
                tokens_to_mask = torch.where(masks[i] == 1)[0]

                # if we don't want to mask anything just continue
                if len(tokens_to_mask) == 0:
                    continue

                # save the original token values for unmasking
                original_token_value = input_ids_all[:, tokens_to_mask].clone()
                # mask the values
                input_ids_all[:, tokens_to_mask] = baseline

                if measure == "log_loss":
                    # prob of generating a token from a perturbation of the input_ids (prompt + current generations)
                    perturbed_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]
                elif measure == "KL":
                    perturbed_probs = self.compute_kl_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

                # change from original generation prob
                score_delta = original_probs - perturbed_probs # [1, len(target_tokens)]

                # since scores are for each output token over the set of input tokens [tokens_to_mask],
                # we expand them to be over all these tokens
                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing = "ij")
                score_array[rows, cols] = score_delta.reshape(-1, 1).repeat((1, len(tokens_to_mask))).to(score_array.dtype) # [len(target_tokens), len(tokens_to_mask)]
                    
                # un-ablate the input
                input_ids_all[:, tokens_to_mask] = original_token_value

            # Append generated tokens to input for next step
            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim = 1)
    
        # remove from the attribution all values associated with the chat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)

    def mlm_mask_indices(self, input_ids, tokens_to_mask):
        """
        Replace masked positions in a LLaMA token sequence using LED.
        """

        # 1. Convert input_ids to tokens
        new_text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 2. Mask only selected tokens
        for idx in tokens_to_mask:
            new_text_tokens[idx] = self.mlm_tokenizer.mask_token

        # 3. Convert tokens back to string
        new_text = self.tokenizer.convert_tokens_to_string(new_text_tokens)

        # 4. Encode for Longformer MLM
        inputs = self.mlm_tokenizer(new_text, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 5. Find masked positions
        masked_positions = (inputs["input_ids"] == self.mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # 6. Add global attention on masked positions
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[0, masked_positions] = 1
        inputs["global_attention_mask"] = global_attention_mask

        # 7. Predict all masked positions in one forward pass
        with torch.no_grad():
            logits = self.mlm_model(**inputs).logits  # [batch, seq_len, vocab_size]
            predicted_ids = logits[0, masked_positions, :].argmax(dim=-1)
            replacement_ids = inputs["input_ids"].clone()
            replacement_ids[0, masked_positions] = predicted_ids

        # 8. Convert predicted tokens to string
        regenerated_tokens = [replacement_ids[0, idx] for idx in masked_positions]
        regenerated_text = self.mlm_tokenizer.decode(predicted_ids, skip_special_tokens=True)
        if regenerated_text and regenerated_text[0] != ' ':
            regenerated_text = ' ' + regenerated_text
            
        # 8. Re-tokenize with LLaMA tokenizer
        replacement_input_ids = self.tokenizer(regenerated_text, return_tensors='pt').input_ids

        # 9. Pad/truncate to match original masked length
        original_len = len(tokens_to_mask)
        new_len = replacement_input_ids.shape[1]

        if new_len > original_len:
            replacement_input_ids = replacement_input_ids[:, :original_len]
        elif new_len < original_len:
            remainder = torch.full((1, original_len - new_len), self.tokenizer.eos_token_id, dtype=torch.long)
            replacement_input_ids = torch.cat((replacement_input_ids, remainder), dim=1)

        if replacement_input_ids.dtype != torch.int64:
            replacement_input_ids = replacement_input_ids.to(torch.int64)

        return replacement_input_ids.to(self.device)

    def calculate_feature_ablation_sentences_mlm(self, prompt, target = None) -> LLMAttributionResult:   
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the prompt ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input tokens and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length


        # given the text user prompt, create a mask over the tokens of each sentence
        user_prompt_sentences = create_sentences("".join(self.user_prompt_tokens), self.tokenizer, show=True)
        sentence_masks_prompt = create_sentence_masks(self.user_prompt_tokens, user_prompt_sentences, show=True)

        # mask prompt sentences and generated sentences
        # given the generation, create a mask over the tokens of each sentence
        generation_sentences = create_sentences("".join(self.generation_tokens), self.tokenizer)
        sentence_masks_generation = create_sentence_masks(self.generation_tokens, generation_sentences)

        # find the total sizes of the masks we need
        l    = len(self.chat_prompt_indices)     # input formating tokens
        n, m = sentence_masks_prompt.shape       # (user prompt sentences, user prompt tokens)
        o, p = sentence_masks_generation.shape   # (generation sentences + EOS, generation tokens + EOS)

        # Create a tensor that can fit all masks diagonally
        masks = torch.zeros((l + n + o, l + m + p))

        # we never mask the chat_prompt_indices, leave as 0
        # prompt indices cover:
        #   0 : start of sentence_masks_prompt
        #   end of sentence_masks_prompt : start of sentence_masks_generation

        # input sentence masks only cover the user prompt
        user_prompt_start_idx = self.user_prompt_indices[0]
        masks[user_prompt_start_idx : user_prompt_start_idx + n, user_prompt_start_idx : user_prompt_start_idx + m] = sentence_masks_prompt

        # gen sentence masks only cover the generations
        masks[l + n:, l + m:] = sentence_masks_generation
        
        num_input_masks = masks.shape[0]

        # instantiate a matrix which will track the attribution of every generated token to intermediate generations
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.full((generation_length, total_length), torch.nan)

        for step in range(len(sentence_masks_generation)):
        # for step in range(len(sentence_masks_generation) + 1):
            input_ids_all = input_ids_all.detach()

            # assume the we are generating a sentence of the generation_ids and we find the
            # prob of generating this sentence from the current input_ids (prompt + any current generations)
            gen_token_indices = torch.where(sentence_masks_generation[step] == 1)[0] # [len(target_tokens)]
            gen_tokens = self.generation_ids[:, gen_token_indices] # [1, len(target_tokens)]

            original_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

            # perturb each sentence of the input and current generation 
            # and measure how the probs of predicitng gen_tokens changes
            for i in range(num_input_masks - len(sentence_masks_generation) + step):
                # find the input tokens to be masked
                tokens_to_mask = torch.where(masks[i] == 1)[0]

                # if we don't want to mask anything just continue
                if len(tokens_to_mask) == 0:
                    continue

                # save the original token values for unmasking
                # original_token_value = input_ids_all.clone()
                original_token_value = input_ids_all[:, tokens_to_mask].clone()

                # we need replace the tokens_to_mask with roberta predicted words and turn them back into tokens
                new_ids = self.mlm_mask_indices(input_ids_all, tokens_to_mask)
                try:
                    input_ids_all[:, tokens_to_mask] = new_ids
                except:
                    print(new_ids)

                # prob of generating a token from a perturbation of the input_ids (prompt + current generations)
                perturbed_probs = self.compute_logprob_response_given_prompt(input_ids_all, gen_tokens).detach().cpu() # [1, len(target_tokens)]

                # change from original generation prob
                score_delta = original_probs - perturbed_probs # [1, len(target_tokens)]

                # since scores are for each output token over the set of input tokens [tokens_to_mask],
                # we expand them to be over all these tokens
                rows, cols = torch.meshgrid(gen_token_indices, tokens_to_mask, indexing = "ij")
                score_array[rows, cols] = score_delta.reshape(-1, 1).repeat((1, len(tokens_to_mask))).to(score_array.dtype) # [len(target_tokens), len(tokens_to_mask)]
                    
                # un-ablate the input
                # input_ids_all = original_token_value
                input_ids_all[:, tokens_to_mask] = original_token_value

            # Append generated tokens to input for next step
            input_ids_all = torch.cat([input_ids_all, gen_tokens], dim = 1)
    
        # remove from the attribution all values associated with the chat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)


class LLMAttentionAttribution(LLMAttribution):
    def __init__(self, model, tokenizer, generate_kwargs = None):
        super().__init__(model, tokenizer, generate_kwargs)

    def calculate_attention_attribution(self, prompt, target = None) -> LLMAttributionResult:
        # run the model so we can access the prompt ids and generated token ids
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)
        
        # Make a copy of the input ids 
        # We will expand the original prompt by each generated token
        input_ids_all = self.prompt_ids.clone()

        # we want to know how many input and generated tokens there are
        input_length = self.prompt_ids.shape[1]
        generation_length = self.generation_ids.shape[1]
        total_length = input_length + generation_length

        # instantiate a matrix which will track the attribution of every generated token to the input
        # cols = total_length because we will capture generation -> previous generation attributions
        score_array = torch.empty((generation_length, total_length))

        with torch.no_grad():
            # for step in tqdm(range(generation_length)):
            for step in range(generation_length):
                input_ids_all = input_ids_all.detach()

                target_token = self.generation_ids[0, step]

                # perform inference
                outputs = self.model(input_ids_all, output_attentions = True)
                
                # get attention weights (mean over layers, heads, and attention rows)
                attentions = torch.stack(outputs.attentions, 0).mean(0).mean(1).mean(1) # [batch, seq_length]
                attentions = torch.stack(outputs.attentions, 0)[-1].mean(1).mean(1) # [batch, seq_length]
                # attentions = torch.stack(outputs.attentions, 0)[-1].mean(1).mean(1) # [batch, seq_length]
                # pad the scores with nan since they must fit into score_array with all other token attributions
                score_array[step] = self.pad_vector(attentions.detach().cpu(), total_length, torch.nan)

                # Append generated token to input for next step
                input_ids_all = torch.cat([input_ids_all, torch.tensor([[target_token]]).to(self.device)], dim=1)

        # remove from the attribution all values associated with thechat prompt
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(self.tokenizer, score_array, self.user_prompt_tokens, self.generation_tokens, all_tokens = all_tokens)
   
    def rollout(self, attentions):
        num_blocks = attentions.shape[0]
        batch_size = attentions.shape[1]
        num_tokens = attentions.shape[2]
        eye = torch.eye(num_tokens).expand(num_blocks, batch_size, num_tokens, num_tokens).to(attentions[0].device)

        matrices_aug = attentions + eye

        # normalize all the matrices, making residual connection addition equal to 0.5*A + 0.5*I
        matrices_aug = matrices_aug / matrices_aug.sum(dim=-1, keepdim=True)

        # perform rollout
        joint_attention = matrices_aug[0]
        for i in range(0 + 1, num_blocks):
            joint_attention = matrices_aug[i].bmm(joint_attention)

        return joint_attention


class LLMLRPAttribution(LLMAttribution):
    """AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers.

    This class implements AttnLRP, a gradient-based attribution method that
    leverages Layer-wise Relevance Propagation (LRP) rules adapted for
    transformer architectures.

    AttnLRP achieves O(1) time complexity (single backward pass) while
    providing theoretically grounded attributions with proven faithfulness.

    Reference:
        AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers
        ICML 2024. https://arxiv.org/abs/2402.05602

    Parameters
    ----------
    model : transformers model
        The language model to compute attributions for
    tokenizer : transformers tokenizer
        The tokenizer for the model
    model_type : str, optional
        The model architecture type. If None, will be auto-detected.
        Supported: 'qwen3', 'qwen2', 'llama'
    generate_kwargs : dict, optional
        Keyword arguments for model.generate()

    Example
    -------
    >>> attr = LLMLRPAttribution(model, tokenizer)
    >>> result = attr.calculate_attnlrp(
    ...     prompt="Context: Mount Everest is 8848m. Question: How high?",
    ...     target="8848 meters"
    ... )
    >>> seq_attr, row_attr, rec_attr = result.get_all_token_attrs([0, len(result.generation_tokens) - 1])
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_type: Optional[str] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model, tokenizer, generate_kwargs)

        # Auto-detect or validate model type
        if model_type is None:
            self.model_type = detect_model_type(model)
        else:
            self.model_type = model_type

    def _resolve_score_mode(
        self,
        score_mode: Optional[Literal["max", "generated"]],
        target: Optional[str],
    ) -> Literal["max", "generated"]:
        if score_mode is None:
            return "generated" if target is not None else "max"
        return score_mode

    def calculate_attnlrp(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> LLMAttributionResult:
        """Calculate AttnLRP attribution for a prompt-response pair.

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each position (original behavior).
            "generated": use the logit of the generated/target token at each position.
            Default: auto ("generated" if target is provided, else "max").

        Returns
        -------
        LLMAttributionResult
            Attribution result with score matrix of shape [gen_len, prompt_len + gen_len]
        """
        # Get the generation (either from model or from target)
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        score_mode = self._resolve_score_mode(score_mode, target)

        # Get lengths
        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        total_len = prompt_len + gen_len

        # Handle empty generation
        if gen_len == 0:
            empty_scores = torch.zeros((0, total_len), dtype=torch.float32)
            return self._finalize_result(empty_scores)

        # Concatenate prompt and generation ids
        input_ids = torch.cat([self.prompt_ids, self.generation_ids], dim=1)

        # Get the embedding layer
        embedding_layer = self.model.get_input_embeddings()

        # Get model dtype for proper precision handling
        model_dtype = next(self.model.parameters()).dtype

        # Initialize score array
        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)

        # Apply LRP patches and compute attributions
        with lrp_context(self.model, self.model_type):
            # Get input embeddings with gradient tracking
            input_embeds = embedding_layer(input_ids).float()
            input_embeds = input_embeds.detach().clone().requires_grad_(True)

            # Forward pass with LRP-patched model
            output_logits = self.model(
                inputs_embeds=input_embeds.to(model_dtype),
                use_cache=False,
            ).logits

            # Compute attribution for each generation position
            for step in range(gen_len):
                gen_pos = prompt_len + step

                if score_mode == "max":
                    score_logit = output_logits[0, gen_pos - 1, :].max()
                elif score_mode == "generated":
                    token_id = self.generation_ids[0, step]
                    score_logit = output_logits[0, gen_pos - 1, token_id]
                else:
                    raise ValueError(f"Unsupported score_mode={score_mode}")

                # Backward pass - this computes LRP through the patched layers
                if input_embeds.grad is not None:
                    input_embeds.grad.zero_()

                score_logit.backward(retain_graph=(step < gen_len - 1))

                # Compute relevance: Input * Gradient, summed over embedding dimension
                # Cast to float32 for numerical stability before summing
                relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0]

                # Store in score array, padded appropriately
                score_array[step, :gen_pos] = relevance[:gen_pos]

        return self._finalize_result(score_array)

    def calculate_attnlrp_batched(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> LLMAttributionResult:
        """Calculate AttnLRP attribution using batched computation.

        This is a memory-efficient version that computes attribution for
        all generation positions in a single forward pass, but requires
        more careful handling of gradients.

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each position (original behavior).
            "generated": use the logit of the generated/target token at each position.
            Default: auto ("generated" if target is provided, else "max").

        Returns
        -------
        LLMAttributionResult
            Attribution result with score matrix
        """
        # Get the generation
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        score_mode = self._resolve_score_mode(score_mode, target)

        # Get lengths
        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        total_len = prompt_len + gen_len

        if gen_len == 0:
            empty_scores = torch.zeros((0, total_len), dtype=torch.float32)
            return self._finalize_result(empty_scores)

        # Concatenate prompt and generation ids
        input_ids = torch.cat([self.prompt_ids, self.generation_ids], dim=1)

        # Get embedding layer and model dtype
        embedding_layer = self.model.get_input_embeddings()
        model_dtype = next(self.model.parameters()).dtype

        # Initialize score array
        score_array = torch.full((gen_len, total_len), torch.nan, dtype=torch.float32)

        with lrp_context(self.model, self.model_type):
            # Get input embeddings
            input_embeds = embedding_layer(input_ids).float()
            input_embeds = input_embeds.detach().clone().requires_grad_(True)

            # Single forward pass
            output_logits = self.model(
                inputs_embeds=input_embeds.to(model_dtype),
                use_cache=False,
            ).logits

            # Get scoring logits for all generation positions
            gen_positions = list(range(prompt_len - 1, prompt_len + gen_len - 1))
            if score_mode == "max":
                score_logits = torch.stack([output_logits[0, pos, :].max() for pos in gen_positions])
            elif score_mode == "generated":
                positions = torch.as_tensor(gen_positions, device=output_logits.device)
                token_ids = self.generation_ids[0].to(device=output_logits.device)
                score_logits = output_logits[0, positions, :].gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError(f"Unsupported score_mode={score_mode}")

            # Backward from sum of all scoring logits
            # This gives us the total relevance across all positions
            if input_embeds.grad is not None:
                input_embeds.grad.zero_()

            score_logits.sum().backward()

            # Compute aggregated relevance
            relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0]

            # For batched version, we use the same relevance for all generation positions
            # This is an approximation but much faster
            for step in range(gen_len):
                gen_pos = prompt_len + step
                score_array[step, :gen_pos] = relevance[:gen_pos]

        return self._finalize_result(score_array)

    def _finalize_result(
        self,
        score_array: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMAttributionResult:
        """Finalize the attribution result.

        Extracts user prompt attributions and creates the result object.

        Parameters
        ----------
        score_array : torch.Tensor
            Raw score array of shape [gen_len, total_len]
        metadata : dict, optional
            Additional metadata to include

        Returns
        -------
        LLMAttributionResult
            The finalized attribution result
        """
        if score_array.ndim == 1:
            score_array = score_array.unsqueeze(0)
        score_array = score_array.detach().cpu()

        # Extract only user prompt attributions (remove chat template tokens)
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        if metadata is None:
            metadata = {}
        metadata["method"] = "attnlrp"
        metadata["model_type"] = self.model_type

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata=metadata,
        )

    def calculate_attnlrp_span_aggregate(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_start: int = 0,
        sink_end: Optional[int] = None,
        sink_weights: Optional[torch.Tensor] = None,
        normalize_weights: bool = True,
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> AttnLRPSpanAggregate:
        """Compute span-wise (multi-token) aggregated AttnLRP in ONE forward + ONE backward.

        This returns a single attribution vector over the whole context (prompt + generation),
        equal to the weighted sum/avg of per-token AttnLRP attributions over the sink span.

        The key insight is that backward propagation is linear with respect to the objective,
        and the LRP patches (divide_gradient, stop_gradient, identity_rule_implicit) are all
        linear transformations on the incoming gradient. Therefore:

            R_F = x ⊙ ∂F/∂x = x ⊙ Σ_g w_g ∂f_g/∂x = Σ_g w_g (x ⊙ ∂f_g/∂x) = Σ_g w_g R_{f_g}

        This means computing attribution for the aggregated objective F = Σ w_g f_g in one
        backward pass is mathematically equivalent to computing per-token attributions and
        summing them with weights.

        Complexity: O(N) instead of O(M×N) for the naive per-token approach.

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        sink_start : int
            Start of sink span in generation token indices (inclusive). Default: 0
        sink_end : int, optional
            End of sink span in generation token indices (inclusive).
            Default: None (uses gen_len - 1, i.e., full generation)
        sink_weights : torch.Tensor, optional
            Optional tensor of shape [span_len], weighting each sink position.
            Default: None (uniform weights)
        normalize_weights : bool
            If True, weights are normalized to sum to 1 (weighted average).
            If False, computes weighted sum. Default: True
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each sink position (matches existing calculate_attnlrp)
            "generated": use the logit of the actually generated token id at each position
            Default: auto ("generated" if target is provided, else "max")

        Returns
        -------
        AttnLRPSpanAggregate
            Aggregated attribution result with token_importance_total vector
        """
        # 1) Get generation (either from model or from target)
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        score_mode = self._resolve_score_mode(score_mode, target)

        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        total_len = prompt_len + gen_len

        # Handle empty generation
        if gen_len == 0:
            empty = torch.zeros((0,), dtype=torch.float32)
            return AttnLRPSpanAggregate(
                token_importance_total=empty,
                all_tokens=[],
                user_prompt_tokens=[],
                generation_tokens=[],
                sink_range=(0, -1),
                sink_weights=None,
                metadata={"method": "attnlrp_span_aggregate", "note": "empty_generation"},
            )

        if prompt_len <= 0:
            raise ValueError("prompt_len must be > 0 for causal LM attribution.")

        # Set default sink_end to full generation
        if sink_end is None:
            sink_end = gen_len - 1

        sink_start = int(sink_start)
        sink_end = int(sink_end)

        if not (0 <= sink_start <= sink_end < gen_len):
            raise ValueError(f"Invalid sink span [{sink_start}, {sink_end}] for gen_len={gen_len}.")

        span_len = sink_end - sink_start + 1

        # 2) Build input ids and embeddings
        input_ids = torch.cat([self.prompt_ids, self.generation_ids], dim=1)
        embedding_layer = self.model.get_input_embeddings()
        model_dtype = next(self.model.parameters()).dtype

        # 3) Forward with LRP patches, then single backward from aggregated scalar objective
        with lrp_context(self.model, self.model_type):
            input_embeds = embedding_layer(input_ids).float()
            input_embeds = input_embeds.detach().clone().requires_grad_(True)

            output_logits = self.model(
                inputs_embeds=input_embeds.to(model_dtype),
                use_cache=False,
            ).logits  # [1, total_len, vocab]

            device = output_logits.device
            logits_dtype = output_logits.dtype

            # Positions in logits corresponding to generation indices g:
            # g=0 -> pos = prompt_len - 1  (logits at position i predict token i+1)
            # g=k -> pos = prompt_len + k - 1
            pos_start = prompt_len + sink_start - 1
            pos_end = prompt_len + sink_end - 1
            positions = torch.arange(pos_start, pos_end + 1, device=device)

            # Build weights tensor
            if sink_weights is None:
                w = torch.ones((span_len,), device=device, dtype=logits_dtype)
                if normalize_weights:
                    w = w / float(span_len)
            else:
                w = sink_weights.to(device=device, dtype=logits_dtype)
                if w.numel() != span_len:
                    raise ValueError("sink_weights length must equal (sink_end - sink_start + 1).")
                if normalize_weights:
                    w = w / (w.sum() + 1e-12)

            # Per-position scalar targets f_g
            if score_mode == "max":
                # Vectorized max over vocab for each selected position
                per_pos = output_logits[0, positions, :].max(dim=-1).values  # [span_len]
            elif score_mode == "generated":
                # Logit of actually generated token id at each position
                token_ids = self.generation_ids[0, sink_start:sink_end + 1].to(device=device)  # [span_len]
                per_pos = output_logits[0, positions, :].gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError(f"Unsupported score_mode={score_mode}")

            # Aggregated scalar objective: F = Σ w_g * f_g
            objective = (w * per_pos).sum()

            if input_embeds.grad is not None:
                input_embeds.grad.zero_()

            objective.backward()

            # 4) Gradient*Input relevance over embedding dim -> per-token relevance
            relevance_full = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0]  # [total_len]
            relevance_with_chat_template = relevance_full.to(torch.float32).clone()

        # 5) Strip chat template tokens (extract only user prompt + full generation tokens)
        score_array = relevance_full.unsqueeze(0)  # [1, total_len]
        score_array = self.extract_user_prompt_attributions(self.prompt_tokens, score_array)
        token_importance_total = score_array[0].to(torch.float32).cpu()

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        metadata = {
            "method": "attnlrp_span_aggregate",
            "base_method": "attnlrp",
            "model_type": self.model_type,
            "sink_range_gen": (sink_start, sink_end),
            "normalize_weights": normalize_weights,
            "score_mode": score_mode,
            # Debug/analysis: token-level relevance aligned to the FULL tokenization
            # (chat template prompt tokens + generation tokens). This does not affect
            # the returned token_importance_total (which is trimmed for evaluation).
            "token_importance_total_with_chat_template": relevance_with_chat_template,
            "prompt_tokens_with_chat_template": list(self.prompt_tokens or []),
            "user_prompt_indices": list(self.user_prompt_indices or []),
            "chat_prompt_indices": list(self.chat_prompt_indices or []),
        }

        return AttnLRPSpanAggregate(
            token_importance_total=token_importance_total,
            all_tokens=all_tokens,
            user_prompt_tokens=self.user_prompt_tokens,
            generation_tokens=self.generation_tokens,
            sink_range=(sink_start, sink_end),
            sink_weights=(sink_weights.detach().cpu() if sink_weights is not None else None),
            metadata=metadata,
        )

    def calculate_attnlrp_aggregated(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> LLMAttributionResult:
        """Calculate aggregated AttnLRP attribution using span aggregation.

        This method provides an O(N) alternative to the naive O(M×N) per-token
        AttnLRP computation. It computes attribution over the full generation span
        in a single forward + backward pass.

        The resulting attribution matrix uses the same aggregated attribution
        vector for all generation rows (since we're computing the combined
        importance of all generation tokens at once).

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each position (original behavior).
            "generated": use the logit of the generated/target token at each position.
            Default: auto ("generated" if target is provided, else "max").

        Returns
        -------
        LLMAttributionResult
            Attribution result compatible with the standard evaluation pipeline
        """
        # Get the generation
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        score_mode = self._resolve_score_mode(score_mode, target)

        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        total_len = prompt_len + gen_len

        # Handle empty generation
        if gen_len == 0:
            empty_scores = torch.zeros((0, total_len), dtype=torch.float32)
            return self._finalize_result(empty_scores, metadata={
                "method": "attnlrp_aggregated",
                "note": "empty_generation"
            })

        # Compute span aggregate over full generation
        aggregate = self.calculate_attnlrp_span_aggregate(
            prompt,
            target=target,
            sink_start=0,
            sink_end=gen_len - 1,
            normalize_weights=True,
            score_mode=score_mode,
        )

        # Build score array: replicate the aggregated vector for each generation row
        # We need to reconstruct the full-length vector before extraction
        relevance_vector = aggregate.token_importance_total

        # The aggregate already has chat tokens stripped; we need to match the format
        # expected by _finalize_result which also strips, so we create a padded version
        user_prompt_len = len(self.user_prompt_tokens)
        gen_token_len = len(self.generation_tokens)
        expected_len = user_prompt_len + gen_token_len

        # Build score matrix
        score_array = torch.full((gen_len, expected_len), torch.nan, dtype=torch.float32)

        # For each generation position, set the attribution up to that position
        for step in range(gen_len):
            gen_pos = user_prompt_len + step
            score_array[step, :gen_pos] = relevance_vector[:gen_pos]

        metadata = {
            "method": "attnlrp_aggregated",
            "model_type": self.model_type,
            "aggregate": aggregate,
        }

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata=metadata,
        )

    def calculate_attnlrp_ft_hop0(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        neg_handling: Literal["drop", "abs"] = "drop",
        norm_mode: Literal["norm", "no_norm"] = "norm",
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> LLMAttributionResult:
        """Return AttnLRP hop0 from the FT multi-hop path as a token-level matrix."""
        multi_hop = self.calculate_attnlrp_multi_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=0,
            neg_handling=neg_handling,
            norm_mode=norm_mode,
            score_mode=score_mode,
        )
        raw_attributions = getattr(multi_hop, "raw_attributions", None) or []
        base_attr = raw_attributions[0] if raw_attributions else None
        if base_attr is None or not hasattr(base_attr, "token_importance_total"):
            raise RuntimeError("AttnLRP hop0 missing from multi-hop result.")

        hop0_vec = torch.as_tensor(getattr(base_attr, "token_importance_total"), dtype=torch.float32).detach().cpu()
        if hop0_vec.numel() <= 0:
            raise RuntimeError("Empty generation for AttnLRP (hop0).")

        user_prompt_len = len(self.user_prompt_tokens)
        gen_len = len(self.generation_tokens)
        gen_len_ids = int(self.generation_ids.shape[1]) if self.generation_ids is not None else gen_len
        if gen_len != gen_len_ids:
            raise RuntimeError(
                "AttnLRP generation length mismatch between decoded tokens and token ids: "
                f"len(generation_tokens)={gen_len} vs generation_ids.shape[1]={gen_len_ids}."
            )
        expected_len = user_prompt_len + gen_len
        if int(hop0_vec.numel()) != expected_len:
            raise RuntimeError("Unexpected AttnLRP hop0 vector length; cannot package into attribution matrix.")

        score_array = torch.full((gen_len, expected_len), torch.nan, dtype=torch.float32)
        for step in range(gen_len):
            gen_pos = user_prompt_len + step
            score_array[step, :gen_pos] = hop0_vec[:gen_pos]

        metadata = {
            "method": "attnlrp_ft_hop0",
            "sink_span": tuple(getattr(base_attr, "sink_range", (0, max(0, gen_len - 1)))),
            "thinking_span": thinking_span,
            "n_hops": 0,
            "neg_handling": neg_handling,
            "norm_mode": norm_mode,
            "ratio_enabled": norm_mode == "norm",
            "multi_hop_result": multi_hop,
        }

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata=metadata,
        )

    def calculate_attnlrp_multi_hop(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        neg_handling: Literal["drop", "abs"] = "drop",
        norm_mode: Literal["norm", "no_norm"] = "norm",
        score_mode: Optional[Literal["max", "generated"]] = None,
        observation_mask: Optional[torch.Tensor | List[float]] = None,
    ) -> MultiHopAttnLRPResult:
        """Compute multi-hop AttnLRP attribution recursively through thinking span.

        This method implements recursive attribution propagation analogous to
        compute_multi_hop_ifr:

        1. Base hop (hop 0): Compute attribution from sink_span (output) to all tokens
        2. For each subsequent hop:
           - Use attribution scores on thinking_span as weights
           - Compute weighted attribution from thinking_span to all tokens
           - Track "observation" (attribution to input tokens, excluding thinking/sink)
           - Update weights for next hop

        The key insight is that attribution mass flowing through the thinking span
        can be "unrolled" by recursively attributing from that span back to earlier
        tokens, weighted by how much each thinking token contributed.

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        sink_span : Tuple[int, int], optional
            (start, end) indices in generation tokens for the output span.
            Default: full generation (0, gen_len-1)
        thinking_span : Tuple[int, int], optional
            (start, end) indices in generation tokens for the reasoning span.
            Default: same as sink_span
        n_hops : int
            Number of recursive hops. Default: 1
        neg_handling : Literal["drop", "abs"]
            How to enforce non-negativity after each hop output.
            "drop": clamp negative values to 0; "abs": take absolute value.
        norm_mode : Literal["norm", "no_norm"]
            "norm": per-hop global normalize + thinking-span normalize + enable hop ratios.
            "no_norm": disable global normalize, thinking normalize, and hop ratios.
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each position (original behavior).
            "generated": use the logit of the generated/target token at each position.
            Default: auto ("generated" if target is provided, else "max").
        observation_mask : torch.Tensor or List[float], optional
            Custom mask for observable tokens. Shape: (gen_len,) or (total_len,).
            1 = observable (input), 0 = not observable (thinking/output).
            Default: auto-generated based on spans.

        Returns
        -------
        MultiHopAttnLRPResult
            Contains raw_attributions, thinking_ratios, and observation dict.
        """
        # Get the generation
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        score_mode = self._resolve_score_mode(score_mode, target)

        prompt_len = int(self.prompt_ids.shape[1])
        gen_len = int(self.generation_ids.shape[1])
        total_len = prompt_len + gen_len

        # Handle empty generation
        if gen_len == 0:
            empty_aggregate = AttnLRPSpanAggregate(
                token_importance_total=torch.zeros((0,), dtype=torch.float32),
                all_tokens=[],
                user_prompt_tokens=[],
                generation_tokens=[],
                sink_range=(0, -1),
                sink_weights=None,
                metadata={"method": "attnlrp_multi_hop", "note": "empty_generation"},
            )
            return MultiHopAttnLRPResult(
                raw_attributions=[empty_aggregate],
                thinking_ratios=[0.0],
                observation={"mask": torch.tensor([]), "base": torch.tensor([]),
                            "per_hop": [], "sum": torch.tensor([]), "avg": torch.tensor([])},
            )

        # Validate and set default spans
        if sink_span is None:
            sink_span = (0, gen_len - 1)
        sink_start, sink_end = sink_span
        if sink_start < 0 or sink_end < sink_start or sink_end >= gen_len:
            raise ValueError(f"Invalid sink_span ({sink_start}, {sink_end}) for gen_len={gen_len}.")

        if thinking_span is None:
            thinking_span = sink_span
        think_start, think_end = thinking_span
        if think_start < 0 or think_end < think_start or think_end >= gen_len:
            raise ValueError(f"Invalid thinking_span ({think_start}, {think_end}) for gen_len={gen_len}.")

        hop_count = max(0, int(n_hops))
        ratio_enabled = norm_mode == "norm"
        if neg_handling not in ("drop", "abs"):
            raise ValueError("neg_handling must be 'drop' or 'abs'.")
        if norm_mode not in ("norm", "no_norm"):
            raise ValueError("norm_mode must be 'norm' or 'no_norm'.")

        # Compute base attribution from sink_span
        base_attr = self.calculate_attnlrp_span_aggregate(
            prompt,
            target=target,
            sink_start=sink_start,
            sink_end=sink_end,
            sink_weights=None,
            normalize_weights=ratio_enabled,
            score_mode=score_mode,
        )

        def _postprocess_hop_vector(v: torch.Tensor) -> torch.Tensor:
            v = torch.nan_to_num(v.to(dtype=torch.float32), nan=0.0)
            if neg_handling == "drop":
                v = v.clamp(min=0.0)
            else:
                v = v.abs()
            if ratio_enabled:
                denom = float(v.sum().item())
                if denom > 0.0:
                    v = v / (denom + 1e-12)
                else:
                    v = torch.zeros_like(v)
            return v

        token_total = _postprocess_hop_vector(base_attr.token_importance_total)
        base_attr.token_importance_total = token_total
        base_attr.metadata = dict(base_attr.metadata or {})
        base_attr.metadata.update(
            {
                "neg_handling": neg_handling,
                "norm_mode": norm_mode,
                "ratio_enabled": ratio_enabled,
            }
        )

        raw_attributions: List[AttnLRPSpanAggregate] = [base_attr]

        # Get the stripped token importance vector (user_prompt + generation tokens)
        T = token_total.shape[0]  # This is user_prompt_len + gen_len after stripping
        user_prompt_len = len(self.user_prompt_tokens)

        # Build observation mask (in stripped token space)
        # think_start/think_end are in generation-token indices
        # In stripped space: thinking is at user_prompt_len + think_start : user_prompt_len + think_end + 1
        # sink is at user_prompt_len + sink_start : user_prompt_len + sink_end + 1
        if observation_mask is None:
            obs_mask = torch.ones((T,), dtype=torch.float32)
            # Mask out thinking span
            think_start_stripped = user_prompt_len + think_start
            think_end_stripped = user_prompt_len + think_end
            obs_mask[think_start_stripped:min(think_end_stripped + 1, T)] = 0.0
            # Mask out sink span
            sink_start_stripped = user_prompt_len + sink_start
            sink_end_stripped = user_prompt_len + sink_end
            obs_mask[sink_start_stripped:min(sink_end_stripped + 1, T)] = 0.0
            # Mask out anything after thinking span (future tokens)
            if think_end_stripped + 1 < T:
                obs_mask[think_end_stripped + 1:] = 0.0
        else:
            obs_mask_input = torch.as_tensor(observation_mask, dtype=torch.float32)
            if obs_mask_input.numel() == gen_len:
                # Expand to full stripped length
                obs_mask = torch.ones((T,), dtype=torch.float32)
                obs_mask[user_prompt_len:user_prompt_len + gen_len] = obs_mask_input
                # Keep input tokens as 1 by default
            elif obs_mask_input.numel() == T:
                obs_mask = obs_mask_input.clone()
            else:
                raise ValueError(f"observation_mask must have length {gen_len} or {T}.")

        # Compute base observation
        base_obs = token_total.clone() * obs_mask
        obs_accum = base_obs.clone()
        per_hop_obs: List[torch.Tensor] = []

        # Extract thinking slice weights for next hop
        think_start_stripped = user_prompt_len + think_start
        think_end_stripped = user_prompt_len + think_end
        thinking_slice = token_total[think_start_stripped:think_end_stripped + 1].detach().clone()
        if ratio_enabled:
            thinking_mass = float(thinking_slice.sum().item())
            if thinking_mass > 0.0:
                w_thinking = thinking_slice / (thinking_mass + 1e-12)
            else:
                w_thinking = torch.zeros_like(thinking_slice)
            total_mass = float(token_total.sum().item())
            current_ratio = thinking_mass / (total_mass + 1e-12) if total_mass > 0 else 0.0
            ratios: List[float] = [current_ratio]
        else:
            w_thinking = thinking_slice
            current_ratio = 1.0
            ratios = []

        # Multi-hop iterations
        for hop in range(1, hop_count + 1):
            # Compute attribution from thinking span with weights from previous hop
            hop_attr = self.calculate_attnlrp_span_aggregate(
                prompt,
                target=target,
                sink_start=think_start,
                sink_end=think_end,
                sink_weights=w_thinking,
                normalize_weights=False,
                score_mode=score_mode,
            )

            hop_total = _postprocess_hop_vector(hop_attr.token_importance_total)
            hop_attr.token_importance_total = hop_total
            hop_attr.metadata = dict(hop_attr.metadata or {})
            hop_attr.metadata.update(
                {
                    "neg_handling": neg_handling,
                    "norm_mode": norm_mode,
                    "ratio_enabled": ratio_enabled,
                }
            )
            raw_attributions.append(hop_attr)

            # Compute observation for this hop (masked and weighted by current_ratio)
            obs_only = hop_total * obs_mask * (current_ratio if ratio_enabled else 1.0)
            obs_accum += obs_only
            per_hop_obs.append(obs_only)

            # Update weights for next hop
            thinking_slice = hop_total[think_start_stripped:think_end_stripped + 1].detach().clone()
            if ratio_enabled:
                thinking_mass = float(thinking_slice.sum().item())
                if thinking_mass > 0.0:
                    w_thinking = thinking_slice / (thinking_mass + 1e-12)
                else:
                    w_thinking = torch.zeros_like(thinking_slice)
                hop_total_mass = float(hop_total.sum().item())
                if hop_total_mass <= 0.0:
                    current_ratio = 0.0
                else:
                    current_ratio *= thinking_mass / (hop_total_mass + 1e-12)
                ratios.append(current_ratio)
            else:
                w_thinking = thinking_slice

        # Compute average observation
        obs_avg = obs_accum / float(max(1, hop_count)) if hop_count > 0 else obs_accum

        observation = {
            "mask": obs_mask,
            "base": base_obs,
            "per_hop": per_hop_obs,
            "sum": obs_accum,
            "avg": obs_avg,
        }

        return MultiHopAttnLRPResult(
            raw_attributions=raw_attributions,
            thinking_ratios=ratios,
            observation=observation,
        )

    def calculate_attnlrp_aggregated_multi_hop(
        self,
        prompt: str,
        target: Optional[str] = None,
        *,
        sink_span: Optional[Tuple[int, int]] = None,
        thinking_span: Optional[Tuple[int, int]] = None,
        n_hops: int = 1,
        neg_handling: Literal["drop", "abs"] = "drop",
        norm_mode: Literal["norm", "no_norm"] = "norm",
        score_mode: Optional[Literal["max", "generated"]] = None,
    ) -> LLMAttributionResult:
        """Calculate multi-hop aggregated AttnLRP attribution.

        This is a convenience wrapper around calculate_attnlrp_multi_hop that
        returns an LLMAttributionResult compatible with the evaluation pipeline.

        The returned attribution uses the observation["sum"] vector which
        accumulates attribution to input tokens across all hops.

        Parameters
        ----------
        prompt : str
            The input prompt text
        target : str, optional
            The target response text. If None, the model generates a response.
        sink_span : Tuple[int, int], optional
            (start, end) indices in generation tokens for the output span.
        thinking_span : Tuple[int, int], optional
            (start, end) indices in generation tokens for the reasoning span.
        n_hops : int
            Number of recursive hops. Default: 1
        neg_handling : Literal["drop", "abs"]
            How to enforce non-negativity after each hop output.
        norm_mode : Literal["norm", "no_norm"]
            "norm": per-hop global normalize + thinking-span normalize + enable hop ratios.
            "no_norm": disable global normalize, thinking normalize, and hop ratios.
        score_mode : Literal["max", "generated"], optional
            "max": use max logit at each position (original behavior).
            "generated": use the logit of the generated/target token at each position.
            Default: auto ("generated" if target is provided, else "max").

        Returns
        -------
        LLMAttributionResult
            Attribution result compatible with the standard evaluation pipeline
        """
        # Get the generation first to set up tokens
        if target is None:
            self.response(prompt)
        else:
            self.target_response(prompt, target)

        gen_len = int(self.generation_ids.shape[1])

        # Handle empty generation
        if gen_len == 0:
            empty_scores = torch.zeros((0, len(self.user_prompt_tokens)), dtype=torch.float32)
            return LLMAttributionResult(
                self.tokenizer,
                empty_scores,
                self.user_prompt_tokens,
                self.generation_tokens,
                all_tokens=self.user_prompt_tokens + self.generation_tokens,
                metadata={"method": "attnlrp_aggregated_multi_hop", "note": "empty_generation"},
            )

        # Compute multi-hop attribution
        multi_hop = self.calculate_attnlrp_multi_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=n_hops,
            neg_handling=neg_handling,
            norm_mode=norm_mode,
            score_mode=score_mode,
        )

        # Use the accumulated observation as the relevance vector
        # This gives attribution to input tokens, accumulated across hops
        relevance_vector = multi_hop.observation["sum"]

        user_prompt_len = len(self.user_prompt_tokens)
        gen_token_len = len(self.generation_tokens)
        expected_len = user_prompt_len + gen_token_len

        # Build score matrix
        score_array = torch.full((gen_len, expected_len), torch.nan, dtype=torch.float32)

        # For each generation position, set the attribution
        for step in range(gen_len):
            gen_pos = user_prompt_len + step
            score_array[step, :gen_pos] = relevance_vector[:gen_pos]

        metadata = {
            "method": "attnlrp_aggregated_multi_hop",
            "model_type": self.model_type,
            "n_hops": n_hops,
            "sink_span": sink_span,
            "thinking_span": thinking_span,
            "neg_handling": neg_handling,
            "norm_mode": norm_mode,
            "ratio_enabled": norm_mode == "norm",
            "thinking_ratios": multi_hop.thinking_ratios,
            "multi_hop_result": multi_hop,
        }

        all_tokens = self.user_prompt_tokens + self.generation_tokens

        return LLMAttributionResult(
            self.tokenizer,
            score_array,
            self.user_prompt_tokens,
            self.generation_tokens,
            all_tokens=all_tokens,
            metadata=metadata,
        )
