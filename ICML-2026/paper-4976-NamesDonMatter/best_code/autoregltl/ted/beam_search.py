"""
Beam Search implementation adapted from
https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/beam_search.py

The primary difference is that this implementation uses negative infinities instead of small negative numbers.
Using small negative numbers was causing the beam search to produce unexpected results.
"""

import torch
import torch.nn.functional as F

def map_structure(func, nested):
    if isinstance(nested, dict):
        return {key: map_structure(func, value) for key, value in nested.items()}
    elif isinstance(nested, (list, tuple)):
        return type(nested)(map_structure(func, item) for item in nested)
    else:
        return func(nested)

def flatten_beam_dim(tensor):
    shape = list(tensor.shape)
    shape[0] *= shape[1]
    shape.pop(1)
    return tensor.view(shape)

def expand_to_same_rank(tensor, target):
    if tensor.ndim is None:
        raise ValueError('')
    if target.ndim is None:
        raise ValueError('')
    diff_rank = target.ndim - tensor.ndim
    for _ in range(diff_rank):
        tensor = tensor.unsqueeze(-1)
    return tensor

class StateKeys:
    """Keys of the dictionary that stores the beam search state"""
    CUR_INDEX = 'cur_index'
    ALIVE_SEQ = 'alive_seq'
    ALIVE_LOG_PROBS = 'alive_log_probs'
    ALIVE_CACHE = 'alive_cache'
    FINISHED_SEQ = 'finished_seq'
    FINISHED_SCORES = 'finished_scores'
    FINISHED_FLAGS = 'finished_flags'

class BeamSearch:

    def __init__(self, logits_fn, batch_size, device, syntax_enforcer,
        max_decode_length, start_id, eos_id, vocab_size, alpha, beam_size, dtype=torch.float32):
        """
        Args:
            logits_fn: Interface to decoder
            batch_size: Integer, batch size
            device: torch.device, device on which to run computations
        """
        self.logits_fn = logits_fn
        self.batch_size = batch_size
        self.device = device
        self.syntax_enforcer = syntax_enforcer
        self.alpha = alpha
        self.beam_size = beam_size
        self.dtype = dtype
        self.max_decode_length = max_decode_length
        self.start_id = start_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size

    def search(self, initial_ids, initial_cache):
        """
        Args:
            initial_ids: Initial input IDs
            initial_cache: Dictionary storing cached values to be passed into logits_fn
        Returns:
            top decoded sequences with shape (batch_size, beam_size, max_decode_length)
            scores of top sequences with shape (batch_size, beam_size)
        """
        # Get initial state
        state = self.get_initial_state(initial_ids, initial_cache)

        while self.condition(state):
            # Detach the gradients to mimic tf.stop_gradient
            state = map_structure(lambda t: t.detach(), self.step(state))

        finished_state = state

        alive_seq = finished_state[StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[StateKeys.FINISHED_FLAGS]

        finished_cond = torch.any(finished_flags, dim=1)
        seq_cond = expand_to_same_rank(finished_cond, finished_seq)
        score_cond = expand_to_same_rank(finished_cond, finished_scores)

        # If there are no finished sequences for a batch item, return alive sequences
        finished_seq = torch.where(seq_cond, finished_seq, alive_seq)
        finished_scores = torch.where(score_cond, finished_scores, alive_log_probs)

        return finished_seq, finished_scores

    def get_initial_state(self, initial_ids, initial_cache):
        """
        Args:
            initial_ids: Initial input IDs
            initial_cache: Dictionary storing cached values to be passed into the logits_fn
        Returns:
            Initial state
        """
        cur_index = torch.tensor(0, device=self.device)

        # Create alive sequence with shape [batch_size, beam_size, 1]
        alive_seq = self.expand_to_beam_size(initial_ids)
        alive_seq = alive_seq.unsqueeze(2)

        # Create tensor for storing initial log probabilities.
        # Assume initial_ids are prob 1.0
        initial_log_probs = torch.tensor([[0.] + [float("-inf")] * (self.beam_size - 1)], dtype=self.dtype, device=self.device)
        alive_log_probs = initial_log_probs.repeat(self.batch_size, 1)

        # Expand all values stored in the dictionary to the beam size, so that each beam has a separate cache.
        alive_cache = map_structure(lambda t: self.expand_to_beam_size(t), initial_cache)

        finished_seq = torch.zeros(self.batch_size, self.beam_size, 1, dtype=torch.int32, device=self.device)
        finished_scores = torch.ones(self.batch_size, self.beam_size, dtype=self.dtype, device=self.device) * float("-inf")
        finished_flags = torch.zeros(self.batch_size, self.beam_size, dtype=torch.bool, device=self.device)

        state = {
            StateKeys.CUR_INDEX: cur_index,
            StateKeys.ALIVE_SEQ: alive_seq,
            StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            StateKeys.ALIVE_CACHE: alive_cache,
            StateKeys.FINISHED_SEQ: finished_seq,
            StateKeys.FINISHED_FLAGS: finished_flags,
            StateKeys.FINISHED_SCORES: finished_scores
        }

        return state

    def condition(self, state):
        """
        Args:
            state: current state
        Returns:
            bool tensor, whether beam search should be continued or not
        """
        # check whether maximum decode length has been reached
        cur_index = state[StateKeys.CUR_INDEX]
        not_at_max_decode_length = torch.lt(cur_index, self.max_decode_length)

        # check whether worst score in finished sequences is better than the best score in alive sequences

        alive_log_probs = state[StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[StateKeys.FINISHED_SCORES]
        finished_flags = state[StateKeys.FINISHED_FLAGS]

        # get best scores in alive sequences
        max_length_norm = self.length_normalization(self.max_decode_length)
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # Get worst scores in finished sequences
        # Set filler scores to zero
        finished_scores = torch.where(finished_flags, finished_scores, 0.0)
        worst_finished_scores = torch.min(finished_scores, dim=1)[0]  # use [0] to extract scores as a tensor
        finished_batches = torch.any(finished_flags, dim=1)
        # Set to negative infinity if no finished sequences
        worst_finished_scores += torch.where(finished_batches, 0.0, float("-inf"))

        worst_finished_better_than_best_alive = torch.all(worst_finished_scores > best_alive_scores)

        return torch.logical_and(not_at_max_decode_length, torch.logical_not(worst_finished_better_than_best_alive))

    def step(self, state):
        """
        Args:
            state: Current state
        Returns:
            New state
        """
        # Grow alive sequences by one step each
        new_alive_seq, new_alive_log_probs, top_ids, new_alive_cache = self.grow_alive_seq(state)

        new_finished_flags = torch.eq(top_ids, self.eos_id)

        # Get new alive and finished state
        alive_state = self.get_new_alive_state(new_alive_seq, new_alive_log_probs, new_finished_flags, new_alive_cache)
        finished_state = self.get_new_finished_state(state, new_alive_seq, new_alive_log_probs, new_finished_flags)

        # Construct new state
        new_state = {StateKeys.CUR_INDEX: state[StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        return new_state

    def grow_alive_seq(self, state):
        """
        Args:
            state: Current state
        Returns:
            Top sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            Scores of top sequences with shape (batch_size, 2 * beam_size)
            New cache of the top sequences
        """

        cur_index = state[StateKeys.CUR_INDEX]
        alive_seq = state[StateKeys.ALIVE_SEQ]

        alive_log_probs = state[StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[StateKeys.ALIVE_CACHE]

        flat_ids = alive_seq.view(self.batch_size * self.beam_size, -1)
        flat_cache = map_structure(flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.logits_fn(flat_ids, cur_index, flat_cache)

        if self.syntax_enforcer is not None:
            flat_logits = self.syntax_enforcer.enforce(flat_ids, flat_logits)

        logits = flat_logits.view(self.batch_size, self.beam_size, -1)

        new_cache = map_structure(lambda t: self.unflatten_beam_dim(t), flat_cache)

        # Convert logits to normalized log probs
        candidate_log_probs = logits.log_softmax(dim=-1)

        log_probs = candidate_log_probs + alive_log_probs.unsqueeze(2)  # (batch_size, beam_size, vocab_size)

        # Get the 2 * beam_size candidates with the highest probabilities
        flat_log_probs = log_probs.view(-1, self.beam_size * self.vocab_size)  # (batch_size, beam_size * vocab_size)

        topk_log_probs, topk_indices = torch.topk(flat_log_probs, 2 * self.beam_size, dim=-1)

        # Extract alive sequences with highest log probabilities
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = self.gather_beams([alive_seq, new_cache], topk_beam_indices, 2 * self.beam_size)
        topk_ids = topk_indices % self.vocab_size
        topk_seq = torch.cat([topk_seq, topk_ids.unsqueeze(2)], dim=2)

        return topk_seq, topk_log_probs, topk_ids, new_cache

    def get_new_alive_state(self, new_alive_seq, new_alive_log_probs, new_finished_flags, new_alive_cache):
        """
        Args:
            new_alive_seq: Int32 tensor, new grown sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            new_alive_log_probs: dtype tensor, log probabilities of new sequences with shape (batch_size, 2 * beam_size)
            new_finished_flags: Bool tensor, indicates which sequences are alive
            new_alive_cache: Dictionary, new cache of sequences
        Returns:
            New partial state containing:
                Top sequences that are still alive with shape (batch_size, beam_size, cur_index + 1)
                Log probabilities of top alive sequences with shape (batch_size, beam_size)
                Cache of top alive sequences
        """
        # Set finished sequences to negative infinity
        new_alive_log_probs = torch.where(new_finished_flags, float("-inf"), new_alive_log_probs)

        top_alive_seq, top_alive_log_probs, top_alive_cache = self.gather_top_beams([new_alive_seq, new_alive_log_probs, new_alive_cache], new_alive_log_probs, self.beam_size)
        # Debug for NaNs if it doesn't work correctly
        # assert torch.all(top_alive_log_probs.isfinite()), "All alive log probs should be finite"

        return {
            StateKeys.ALIVE_SEQ: top_alive_seq,
            StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def get_new_finished_state(self, state, new_alive_seq, new_alive_log_probs, new_finished_flags):
        """
        Args:
            state: Dictionary, current state
            new_alive_seq: Int32 tensor, new grown sequences with shape (batch_size, 2 * beam_size, cur_index + 1)
            new_alive_log_probs: Dtype tensor, log probabilities of new sequences with shape (batch_size, 2 * beam_size)
            new_finished_flags: Bool tensor, indicates which sequences are alive
        Returns:
            New partial state containing:
                Top finished sequences with shape (batch_size, beam_size, cur_index + 1)
                Finished scores of top finished sequences with shape (batch_size, beam_size)
                Finished flags of finished sequences with shape (batch_size, beam_size)
        """
        cur_index = state[StateKeys.CUR_INDEX]

        finished_seq = state[StateKeys.FINISHED_SEQ]
        finished_scores = state[StateKeys.FINISHED_SCORES]
        finished_flags = state[StateKeys.FINISHED_FLAGS]

        # Append a column of zeros to finished_seq to increment length
        finished_seq = torch.cat([finished_seq, torch.zeros(self.batch_size, self.beam_size, 1, dtype=torch.int32, device=self.device)], dim=2)

        # Calculate new scores from log probabilities
        new_scores = new_alive_log_probs / self.length_normalization(cur_index + 1)
        # Set the scores of the still-alive seq in new_seq to negative infinity.
        new_scores += torch.where(new_finished_flags, 0.0, float("-inf"))

        finished_seq = torch.cat([finished_seq, new_alive_seq], dim=1)
        finished_scores = torch.cat([finished_scores, new_scores], dim=1)
        finished_flags = torch.cat([finished_flags, new_finished_flags], dim=1)

        top_finished_seq, top_finished_scores, top_finished_flags = self.gather_top_beams([finished_seq, finished_scores, finished_flags], finished_scores, self.beam_size)

        return {
            StateKeys.FINISHED_SEQ: top_finished_seq,
            StateKeys.FINISHED_SCORES: top_finished_scores,
            StateKeys.FINISHED_FLAGS: top_finished_flags
        }

    def gather_beams(self, nested, beam_indices, new_beam_size):
        """
        Args:
            nested: Nested structure (tensor, list, tuple or dict) containing tensors with shape (batch_size, beam_size, ...)
            beam_indices: Tensor with shape (batch_size, new_beam_size) specifying beams that are gathered
            new_beam_size: Number of beams pulled from nested tensors
        Returns:
            Nested structure containing tensors with shape (batch_size, new_beam_size, ...)
        """
        batch_pos = torch.arange(self.batch_size * new_beam_size) // new_beam_size
        batch_pos = batch_pos.view(self.batch_size, new_beam_size)

        # Creating a tensor with shape (batch_size, beam_size, 2) where the last dimension contains gathering coordinates (i, j)
        # coordinates = torch.stack([batch_pos, beam_indices], dim=2)
        # map tf.gather_nd(state, coordinates)
        return map_structure(lambda state: state[batch_pos, beam_indices], nested)

    def gather_top_beams(self, nested, log_probs, beam_size):
        _, top_indices = torch.topk(log_probs, k=beam_size, dim=-1)
        return self.gather_beams(nested, top_indices, beam_size)

    def length_normalization(self, length):
        """
        Calculate the length normalization divisor.
        """
        # Check if length is a torch Tensor
        if isinstance(length, torch.Tensor):
            length = length.to(self.dtype)
        else:
            length = torch.tensor(length, dtype=self.dtype, device=self.device)
        # Length normalization in https://arxiv.org/abs/1609.08144
        # return torch.pow((length + 5.0) / 6.0, self.alpha)

        # Length normalization in huggingface transformers
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
        return torch.pow(length, self.alpha)

    def expand_to_beam_size(self, tensor):
        """Tiles a given tensor by beam_size.

        Args:
            tensor: tensor to tile [batch_size, ...]

        Returns:
            Tiled tensor [batch_size, beam_size, ...]
        """
        tensor = tensor.unsqueeze(1)
        tile_dims = [1] * tensor.ndim
        tile_dims[1] = self.beam_size
        return tensor.repeat(tile_dims)

    def unflatten_beam_dim(self, tensor):
        shape = list(tensor.shape)
        new_shape = [self.batch_size, self.beam_size] + shape[1:]
        return tensor.view(new_shape)