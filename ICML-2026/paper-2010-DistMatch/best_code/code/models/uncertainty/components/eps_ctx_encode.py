import logging
import math
from typing import Tuple

import torch
from torch import nn

from utils.calc_torch import unfold_window

LOGGER = logging.getLogger(__name__)


class FcModel(nn.Module):

    def __init__(self, input_dim, out_dim, hidden: Tuple, dropout: float = 0, dropout_at_first=False,
                 dropout_after_last=False, dropout_intermediate=False, relu_after_last=False) -> None:
        nn.Module.__init__(self)
        self._out_dim = out_dim
        if dropout is None:
            dropout = 0
        hidden_layers = []
        if len(hidden) > 0:
            for idx, layer in enumerate(hidden):
                hidden_layers.append(nn.ReLU())
                if dropout > 0 and dropout_intermediate:
                    hidden_layers.append(nn.Dropout(p=dropout))
                hidden_layers.append(nn.Linear(layer, hidden[idx+1] if idx < (len(hidden) - 1) else self._out_dim))
            stack = [nn.Linear(input_dim, hidden[0])] + hidden_layers
        else:
            stack = [nn.Linear(input_dim, self._out_dim)]

        if dropout > 0 and dropout_at_first:
            stack = [nn.Dropout(p=dropout)] + stack
        if relu_after_last:
            stack.append(nn.ReLU())
        if dropout > 0 and dropout_after_last:
            stack.append(nn.Dropout(p=dropout))
        self.linear_stack = nn.Sequential(*stack)

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, context, **kwargs):
        return self.linear_stack(context)


class PositionEncoding(nn.Module):
    OPS = {
        'abs-transformer': ("_abs_transformer", True),
        'abs-transformer-cat': ("_abs_transformer", False),
        'rel-transformer': ("_rel_transformer", True),
        'rel-transformer-cat': ("_rel_transformer", False),
        'rel-simple': ("_rel_simple", False),
        'rel-linear': ("_rel_linear", False),
        'rel-decay': ("_rel_decay", False),
    }

    def __init__(self, mode: str, **kwargs) -> None:
        nn.Module.__init__(self)
        assert mode.split("_")[0] in self.OPS
        self._op_mode = mode.split("_")[0]
        self._op, self._add_to_feature = self.OPS[self._op_mode]
        self._op = getattr(self, self._op)
        if not self._add_to_feature:
            self._add_dim = (int(mode.split("_")[1]) if len(mode.split("_")) > 1 else 1)
        if self._op_mode == "rel-linear":
            self._pos_back_window = torch.nn.Parameter(
                data=torch.floor(torch.linspace(start=kwargs['mem_size'], end=10, steps=kwargs['dim'])),
                requires_grad=False)
            self._add_dim = kwargs['dim']
        if self._op_mode == "rel-decay":
            # 1000/100/10 steps back -> DecayBase 0.003/0.03/0.3
            self._pos_decays = torch.nn.Parameter(
                data=1 - (3 * torch.pow(torch.linspace(start=kwargs['mem_size'], end=10, steps=kwargs['dim']), -1)),
                requires_grad=False)
            self._add_dim = kwargs['dim']

    def forward(self, context_enc, step_no, min_step=None, max_step=None, ref_step=None):
        """
        :param context_enc: [batch, seq_len, ctx_enc_size] or [seq_len, ctx_enc_size]
        :param step_no: [batch, seq_len, 1] or [seq_len, 1]
        :param min_step: int
        :param max_step: int
        :param ref_step: int
        :return:
        """
        device = context_enc.device
        if self._add_to_feature:
            return context_enc + self._encoding(step_no=step_no, dim=context_enc.shape[-1], min_step=min_step,
                                                max_step=max_step, ref_step=ref_step, device=device)
        else:
            return torch.cat((context_enc, self._encoding(step_no=step_no, dim=self._add_dim, min_step=min_step,
                                                          max_step=max_step,  ref_step=ref_step, device=device)),
                             dim=(1 if len(context_enc.shape) == 2 else 2))

    @property
    def additional_dim(self):
        return 0 if self._add_to_feature else self._add_dim

    def _encoding(self, step_no, dim, min_step, max_step, ref_step, device):
        if len(step_no.shape) == 3:
            batches, sequence_len = step_no.shape[0], step_no.shape[1]
        else:
            batches, sequence_len = None, step_no.shape[0]
        step_no = step_no.reshape(-1, 1)
        # TODO Refstep/MaxStep/MinStep is not  batchwise
        enc = self._op(step_no=step_no, dim=dim, min_step=min_step, max_step=max_step, ref_step=ref_step, device=device)
        if batches is not None:
            return enc.reshape(batches, sequence_len, -1)
        else:
            return enc.reshape(sequence_len, -1)

    def _abs_transformer(self, step_no, dim, device, **_):
        pe_matrix = torch.zeros(step_no.shape[0], dim, device=device)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe_matrix[:, 0::2] = torch.sin(step_no * div_term)
        pe_matrix[:, 1::2] = torch.cos(step_no * div_term)
        return pe_matrix

    def _rel_transformer(self, step_no, dim, min_step, device, **_):
        return self._abs_transformer(step_no=step_no - min_step, dim=dim, device=device)

    def _rel_simple(self, step_no, min_step, max_step, **_):
        assert self._add_dim == 1
        steps = max_step - min_step
        return (step_no - min_step) / steps

    def _rel_linear(self, step_no, ref_step, **_):
        relative_step = step_no - ref_step
        return torch.nn.functional.relu(relative_step + self._pos_back_window) / self._pos_back_window

    def _rel_decay(self, step_no, ref_step, **_):
        relative_step = torch.abs(step_no - ref_step)
        return torch.pow(self._pos_decays.unsqueeze(0).expand(step_no.shape[0], -1), relative_step)


class HistoryCompression(nn.Module):

    # Tuple: PoolMode, useReset
    OPS = {
        'max-reset': ("_max_pool", True),
        'sum-reset': ("_sum_pool", True),
        'mean-reset': ("_mean_pool", True),
        'decay-reset': ("_decay_pool", True),
        'max-pool': ("_max_pool",  False),
        'sum-pool': ("_sum_pool", False),
        'mean-pool': ("_mean_pool", False),
        'decay-pool': ("_decay_pool", False),
    }

    def __init__(self, ctx_enc_size, mode: str, window) -> None:
        """
        :param mode:
        :param window: maximal backwards window considered in the history (if None full history considered)
        """
        nn.Module.__init__(self)
        assert mode in self.OPS
        self._op_mode = mode
        self._op, self._with_reset = self.OPS[self._op_mode]
        self._window = int(window) if window is not None else None
        self._op = getattr(self, self._op)
        self._ctx_enc_size = ctx_enc_size
        if self._with_reset:
            self._reset = nn.Sequential(nn.Linear(self._ctx_enc_size * 2, self._ctx_enc_size), nn.Sigmoid())

    def forward(self, context_enc, history_enc, history_state):
        """
        Training:
        Either only context_enc [seq_len, ctx_enc_size] (latest element last!)
        or
        Inference:
        context_enc [1, ctx_enc_size]          # Current Step
        history_enc [hist_len, ctx_enc_size]   # Memory (latest element last!)
        history_state: reset state (in case reset is used)
        """
        if history_enc is not None:  # In Inference or mix mini-batch
            if history_state is None:
                history_state = self._init_state(context_enc.device)
            assert context_enc.shape[0] == 1
            if self._window is None:
                history_compression = self._op(torch.cat((history_enc, context_enc), dim=0), only_last=True)
            else:
                history_compression = self._op(torch.cat((history_enc[-self._window:], context_enc)), only_last=True)
        elif self._window is None:  # Whole Batch in Training no window
            history_state = self._init_state(context_enc.device)
            history_compression = self._op(context_enc, only_last=False)
        else:  # Whole Batch with limited History Window
            assert self._window < context_enc.shape[0] + 1
            history_state = self._init_state(context_enc.device)
            if self._op_mode.startswith("mean"):
                # Avg Pool does not work pecause only half kernel size is allowed to pad
                #history_compression = torch.avg_pool1d(context_enc, kernel_size=self._window, stride=1,
                #                                       padding=self._window-1, count_include_pad=False)[:-self._window]
                # For the first window_size elements we calc the mean manually
                first_part = self._mean_pool(context_enc[:self._window-1], only_last=False)
                windowed_ctx = unfold_window(M=context_enc, window_len=self._window)
                history_compression = self._op(windowed_ctx.transpose(0, 1), only_last=True).squeeze(0)
                history_compression = torch.cat((first_part, history_compression))
            else:
                # Split in windows and use onlyLast
                windowed_ctx = unfold_window(M=context_enc, window_len=self._window,
                                             M_past=self._get_padding(context_enc.shape[1], context_enc.device,))
                history_compression = self._op(windowed_ctx.transpose(0, 1), only_last=True).squeeze(0)

        if self._with_reset:
            reset_history = torch.empty_like(history_compression)
            for idx, ctx_step in enumerate(context_enc):
                reset_in = torch.cat((history_state, ctx_step)).unsqueeze(0)
                history_state = history_compression[idx] * self._reset(reset_in).squeeze(0)
                reset_history[idx] = history_state
        else:
            reset_history = history_compression
        return reset_history, history_state

    @property
    def additional_dim(self):
        return self._ctx_enc_size

    def _init_state(self, device):
        if self._op_mode.endswith("reset"):
            state = torch.full((self._ctx_enc_size,), fill_value=-1.0, device=device)  # ToDo Which value to set?
        else:
            state = None
        return state

    def _get_padding(self, n_feat, device):
        if self._op_mode.startswith("sum"):
            return torch.zeros((self._window-1, n_feat), device=device)
        elif self._op_mode.startswith("max"):
            return torch.full((self._window-1, n_feat), fill_value=-float('inf'), device=device)
        else:
            raise ValueError("Invalid mode for padding!")

    def _sum_pool(self, input, only_last, **kwargs):
        if only_last:
            return torch.sum(input, dim=0, keepdim=True)
        else:
            return torch.cumsum(input, dim=0)

    def _max_pool(self, input, only_last, **kwargs):
        if only_last:
            return torch.max(input, dim=0, keepdim=True).values
        else:
            return torch.cummax(input, dim=0).values

    def _mean_pool(self, input, only_last, **kwargs):
        if only_last:
            return torch.sum(input, dim=0, keepdim=True) / input.shape[0]
        else:
            num = torch.cumsum(input, dim=0)
            denom = torch.arange(start=1, end=input.shape[0]+1, device=input.device).unsqueeze(1).expand_as(num)
            return torch.div(num, denom)

    def _decay_pool(self, input, only_last, **kwargs):
        assert self._window is None  # Not Supported (Decay automatically creates "natural window")
        exp_base = torch.tensor([0.995], dtype=torch.float, device=input.device)  # 0.995 -> 13% after 400 steps
        if only_last:
            decay = torch.pow(exp_base, torch.arange(start=0, end=input.shape[0], device=input.device))
            decay_sum = torch.sum(decay)
            decay = decay / decay_sum
            r = torch.sum(input * decay.unsqueeze(1), dim=0, keepdim=True)
            return r
        else:
            seq_len = input.shape[0]
            arrange_1 = torch.arange(start=0, end=seq_len, device=input.device)
            decay = torch.pow(exp_base, arrange_1).unsqueeze(1)
            decay = decay.repeat(1, seq_len)
            shifts = arrange_1
            arrange_2 = (arrange_1.unsqueeze(1).repeat(1, seq_len) - shifts.unsqueeze(0)) % seq_len
            decay = torch.gather(decay, 0, arrange_2)
            decay = torch.tril(decay)
            input = input.unsqueeze(1).repeat(1, seq_len, 1)
            decay_sum = torch.sum(decay, dim=0, keepdim=True)
            decay = decay / decay_sum
            decayed = input * decay.unsqueeze(2)
            decayed = torch.sum(decayed, dim=0)
            return decayed


class ContextEncodeModule(nn.Module):

    def __init__(self, ctx_input_dim, ctx_out_dim, ctx_enc_hidden, pos_encode=None, history_compression=None,
                 dropout=None, relu_after_last=False) -> None:
        nn.Module.__init__(self)
        assert pos_encode is None  # Pos encode has to be separate!
        self._base_encode = FcModel(input_dim=ctx_input_dim, out_dim=ctx_out_dim, hidden=ctx_enc_hidden,
                                    dropout=dropout, relu_after_last=relu_after_last,
                                    dropout_intermediate=True, dropout_after_last=True)
        # Init Compression
        if history_compression is not None:
            if isinstance(history_compression['mode'], list):
                modes = history_compression['mode']
                if isinstance(history_compression['window'], list):
                    windows = history_compression['window']
                    assert len(modes) == len(windows)
                else:
                    windows = [history_compression['window'] for _ in range(len(modes))]

                self._history_compression = nn.ModuleList(
                    [HistoryCompression(ctx_enc_size=ctx_out_dim, mode=mode, window=windows[idx])
                     for idx, mode in enumerate(modes)])

            else:
                self._history_compression = nn.ModuleList(
                    [HistoryCompression(ctx_enc_size=ctx_out_dim, **history_compression)])
        else:
            self._history_compression = None

    @property
    def output_dim(self):
        return self._base_encode.output_dim + \
               (sum([compress.additional_dim for compress in self._history_compression])
                if self._history_compression is not None else 0)

    @property
    def has_history_comp(self):
        return self._history_compression is not None

    def forward(self, context, step_no, context_past, context_past_state,
                past_pre_encoded=False, past_has_history=False, past_real_len=None):
        """
        !!
        Latest must be last element in sequence dimension
        !!
        :param step_no:
        :param context: [batch, seq_len, ctx_size] or [seq_len, ctx_size]
        :param step_no: [batch, seq_len, 1] or [seq_len, 1]
        :param context_past: Optional([batch, hist_seq_len, ctx_size] or [seq_len, hist_seq_len, ctx_size]
        :param context_past_state: Optional(list(Any)) - Only without batched context!
        :param past_pre_encoded: If true past context already has base encoding
        :param past_has_history: If true past context has already history
        :param past_real_len: Optional[batch] - Integer defining the real length of padded history

        """
        # 1) Check Batching an do basic pre-encode
        if len(context.shape) == 3:
            batches, seq_len, ctx_size = context.shape
            assert context_past_state is None
            context = self._base_encode(context.view(-1, ctx_size)).view(batches, seq_len, -1)
        else:
            batches, seq_len, ctx_size = None, context.shape[0], context.shape[1]
            context = self._base_encode(context)
            context = context.unsqueeze(0)

        # 2) Create History Compression
        if self._history_compression is not None:
            # Base Encode/Selection History Context
            # ONLY relevant fo NO batches
            if context_past is not None:
                if not past_pre_encoded:
                    if batches is None:
                        context_past = self._base_encode(context_past).unsqueeze(0)
                    else:
                        assert context_past.shape[0] == batches
                        assert context_past.shape[2] == ctx_size
                        context_past = self._base_encode(context_past.view(-1, ctx_size)).view(batches, -1, ctx_size)
                elif past_has_history:
                    assert batches is None
                    context_past = context_past[:, 0:context_past.shape[1] // (len(self._history_compression) + 1)].unsqueeze(0)
                else:
                    assert batches is None
                    context_past = context_past.unsqueeze(0)
            else:
                context_past = None

            batches_with_history = []
            for batch_no, batch_ctx in enumerate(context):
                history_compressed = []
                history_state = []
                hist_len = past_real_len[batch_no] if past_real_len is not None else \
                    (context_past.shape[1] if context_past is not None else None)
                for idx, compression in enumerate(self._history_compression):
                    compressed, state = compression(
                        context_enc=batch_ctx,
                        history_enc=context_past[batch_no][-hist_len:] if context_past is not None else None,
                        history_state=context_past_state[idx] if context_past_state is not None else None)
                    history_compressed.append(compressed)
                    history_state.append(state)

                history_compressed.append(batch_ctx)
                history_compressed.reverse()
                batches_with_history.append(torch.cat(history_compressed, dim=1))

            batches_with_history = torch.stack(batches_with_history, dim=0)

            if batches is None:
                assert batches_with_history.shape[0] == 1
                return batches_with_history.squeeze(0), history_state
            else:
                return batches_with_history, None
        else:
            if batches is None:
                return context.squeeze(0), None
            else:
                return context, None
