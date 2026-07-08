import collections
import logging
from abc import abstractmethod
from typing import List, Optional, Tuple

import hydra
import torch
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader, default_collate
from torchmetrics import MetricCollection

from models.uncertainty.components.data_mixing import SubGroupMixSampler
from models.uncertainty.components.eps_memory import FiFoMemory
from models.uncertainty.pi_base import PICalibData, PICalibArtifacts
from models.uncertainty.score_service import score
from utils.calc_torch import calc_residuals, unfold_window
from utils.loss import pinball_loss, width_loss, coverage_loss, chung_calib_loss, mse_loss
from utils.metrics import WinklerScore, MissCoverage, PIWidth, DummyMetric, CoverageDiff

LOGGER = logging.getLogger(__name__)


class EpsSelTrainerDataset(Dataset):

    def __init__(self, eps_context: List[torch.Tensor], Y: List[torch.Tensor], Y_hat: List[torch.Tensor],
                 alpha: List[float], step_no: List[torch.Tensor],
                 ts_id: Optional[List[str]] = None,
                 score_param=None,
                 real_history_size: Optional[List[int]] = None,
                 add_history_ctx: Optional[List[torch.Tensor]] = None,
                 add_history_step_no: Optional[List[torch.Tensor]] = None,
                 has_history=True):
        super().__init__()
        assert len(eps_context) == len(Y) == len(Y_hat) == len(step_no) == len(alpha)
        self._eps_context = eps_context
        self._Y = Y
        self._Y_hat = Y_hat
        self._step_no = step_no
        self._alpha = alpha
        self._ts_id = ts_id
        if has_history:
            self._has_history = True
            self._real_history_size = real_history_size
            self._add_history_ctx = add_history_ctx
            self._add_history_step_no = add_history_step_no
        else:
            self._has_history = False
            self._score_param = score_param

    def __getitem__(self, index):
        if not self._has_history:
            return self._eps_context[index], self._Y[index], self._Y_hat[index], self._alpha[index], self._step_no[index], \
                self._ts_id[index], self._score_param[index]
        else:
            return self._eps_context[index], self._Y[index], self._Y_hat[index], self._alpha[index], self._step_no[index],\
                   self._ts_id[index], self._real_history_size[index],\
                   self._add_history_ctx[index], self._add_history_step_no[index]

    def __len__(self):
        return len(self._eps_context)

    def _get_mask(self, batch_steps):
        return torch.ones((batch_steps, batch_steps), dtype=torch.bool)


BATCH_MODE_ONE_TS = "one_ts"
BATCH_MODE_NAIVE_MIX = "naive_mix"

LOSS_MODE_RES = "residuals"
LOSS_MODE_ABS = "absolute"
LOSS_MODE_MSE = "mse"
LOSS_MODE_MIX = "mix_lh"
LOSS_MODE_EPS_CDF = "eps_cdf"


class PadCollate:
    PAD_VALUE = -1  # DO NOT CHANGE THIS VALUE!

    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_collate(self, batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            # find the longest sequence and check if size differes
            lenghts = list(map(lambda x: x.shape[self.dim], batch))
            if lenghts.count(lenghts[0]) == len(lenghts):
                return default_collate(batch)
            else:
                max_len = max(lenghts)
                # pad according to max_len
                batch = list(map(lambda x: self.pad_tensor(x, pad=max_len, dim=self.dim), batch))
                # stack all
                return torch.stack(batch, dim=0)
        # Pytorch Code Start
        elif isinstance(elem, collections.abc.Sequence):
            elem_type = type(elem)
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self.pad_collate(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self.pad_collate(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.pad_collate(samples) for samples in transposed]
        # Pytorch Code END
        else:
            return default_collate(batch)

    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.full(size=pad_size, fill_value=self.PAD_VALUE)], dim=dim)


class CalibTrainerMixin:

    def __init__(self, batch_mode, with_loss_weight, coverage_loss_weight, chung_loss_weight=0,
                 batch_size=None, batch_mix_count=None, all_alpha_in_one_batch=False,
                 split_in_subsequence_of_size=None, subsequence_stride=1,
                 loss_mode=LOSS_MODE_RES, pub_training=False) -> None:
        self._batch_mode = batch_mode
        self._batch_mix_count = batch_mix_count
        self._width_loss_weight = with_loss_weight
        self._coverage_loss_weight = coverage_loss_weight
        self._chung_loss_weight = chung_loss_weight
        self._all_alpha_in_one_batch = all_alpha_in_one_batch
        self._batch_size = batch_size
        self._sub_sequence = split_in_subsequence_of_size
        self._sub_sequence_stride = subsequence_stride
        self._loss_mode = loss_mode
        self._pub_training = pub_training
        self._validate()

    def _validate(self):
        assert self._batch_mode in [BATCH_MODE_ONE_TS, BATCH_MODE_NAIVE_MIX]
        assert self._loss_mode in [LOSS_MODE_ABS, LOSS_MODE_RES, LOSS_MODE_MIX, LOSS_MODE_MSE, LOSS_MODE_EPS_CDF]
        if self._loss_mode == LOSS_MODE_ABS:
            assert self._coverage_loss_weight == 0
            assert self._chung_loss_weight == 0
        elif self._loss_mode == LOSS_MODE_MSE:
            assert self._chung_loss_weight == 0
            assert self._width_loss_weight == 0
        else:
            assert self._chung_loss_weight == 0
            assert self._width_loss_weight == 0
            assert self._coverage_loss_weight == 0
        if self._batch_mode == BATCH_MODE_NAIVE_MIX:
            assert self._batch_mix_count is not None
        else:
            assert self._batch_mix_count is None
            assert not self._pub_training

    def _no_train_alpha_needed(self):
        return self._loss_mode in [LOSS_MODE_MIX, LOSS_MODE_MSE, LOSS_MODE_EPS_CDF]

    def _train_model(self, calib_data: [PICalibData], Y_hat: List, fc_state_step: List,
                     alphas: List[float], experiment_config, trainer_config, history_size=None):

        metrics = MetricCollection(WinklerScore(coverage_values=alphas),
                                   MissCoverage(coverage_values=alphas),
                                   CoverageDiff(coverage_values=alphas),
                                   PIWidth(coverage_values=alphas))

        if self._batch_mode == BATCH_MODE_ONE_TS:
            data_loader = lambda num_worker=0: self.get_data_loader(calib_data=calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step,
                                                       alphas=alphas, num_worker=num_worker)
            map_to_model = lambda batch_data:\
                dict(ctx_data=batch_data[0], Y=batch_data[1], Y_hat=batch_data[2], alpha=batch_data[3],
                     step_no=batch_data[4], score_param=batch_data[6])
            move_batch_to_device = lambda batch_data, device:\
                (batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device),
                 batch_data[4].to(device), batch_data[5].to(device), batch_data[6].to(device))
            map_to_loss_in = lambda model_out, batch_data: dict(Y=batch_data[1], base_alphas=batch_data[3], **model_out)
            map_to_metrics_in = (lambda model_out, batch_data: dict(Y=batch_data[1], alpha=batch_data[3], **model_out))\
                if not self._no_train_alpha_needed() else\
                (lambda model_out, batch_data: dict(Y=batch_data[1],  **model_out))
        else:
            data_loader = lambda num_worker=0:\
                self.get_data_loader_naive_mix(calib_data=calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step,
                                               alphas=alphas, history_size=history_size, num_worker=num_worker)
            if history_size is not None and history_size > 0:
                assert score.mode is None
                map_to_model = lambda batch_data:\
                    dict(ctx_data=batch_data[0], Y=batch_data[1], Y_hat=batch_data[2], alpha=batch_data[3],
                         step_no=batch_data[4], ts_id=batch_data[5], real_hist_size=batch_data[6], ctx_hist=batch_data[7],
                         step_hist=batch_data[8])
                move_batch_to_device = lambda batch_data, device: \
                    (batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device),
                     batch_data[4].to(device), batch_data[5].to(device), batch_data[6].to(device), batch_data[7].to(device),
                     batch_data[8].to(device))
            else:
                map_to_model = lambda batch_data:\
                    dict(ctx_data=batch_data[0], Y=batch_data[1], Y_hat=batch_data[2], alpha=batch_data[3],
                         step_no=batch_data[4], ts_id=batch_data[5], score_param=batch_data[6])
                move_batch_to_device = lambda batch_data, device: \
                    (batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device),
                     batch_data[3].to(device), batch_data[4].to(device), batch_data[5].to(device),
                     batch_data[6].to(device))


            map_to_loss_in = lambda model_out, batch_data: dict(**model_out)
            map_to_metrics_in = lambda model_out, batch_data: dict(**model_out)

        trainer = hydra.utils.instantiate(
            trainer_config,
            experiment_config=experiment_config,
            model=self,
            get_data_loader=data_loader,
            move_batch_to_device=move_batch_to_device,
            map_to_model_in=map_to_model,
            loss_func=get_loss_func(self._width_loss_weight, self._coverage_loss_weight,
                                    self._chung_loss_weight, loss_mode=self._loss_mode),
            map_to_loss_in=map_to_loss_in,
            val_metrics=metrics,
            train_metrics=metrics if not self._no_train_alpha_needed() else MetricCollection(DummyMetric()),
            map_to_metrics_in=map_to_metrics_in
        )
        trainer.train()

    @abstractmethod
    def _get_calib_ctx(self, calib_data: PICalibData, Y_hat, fc_state_step=None) -> Tuple[torch.tensor, int, int]:
        """
        :return: context of calib data, window_offset, ts_id_enc
        """
        pass

    def get_data_loader(self, calib_data: [PICalibData], Y_hat: List, fc_state_step: List,
                        alphas: List[float], num_worker=0):
        splits = [], []
        for idx, c_data in enumerate(calib_data):
            # Prepare Context
            ctx_data, window_offset, ts_id_enc = self._get_calib_ctx(calib_data=c_data, Y_hat=Y_hat[idx],
                                                                     fc_state_step=fc_state_step[idx])
            step_no = torch.arange(start=c_data.step_offset, end=c_data.step_offset + c_data.Y_calib.shape[0],
                                   dtype=torch.long)
            # Split in Train/Val
            assert ctx_data.shape[0] == c_data.Y_calib[window_offset:].shape[0] == Y_hat[idx][window_offset:].shape[0] == step_no[window_offset:].shape[0]
            split_size = (ctx_data.shape[0] // 2) + ctx_data.shape[0] % 2
            # ToDo Split in train/val in a more advanced way
            ctx_data_split = torch.split(ctx_data, split_size, dim=0)
            Y_split = torch.split(c_data.Y_calib[window_offset:], split_size, dim=0)
            Y_hat_split = torch.split(Y_hat[idx][window_offset:], split_size, dim=0)
            step_no_split = torch.split(step_no[window_offset:], split_size, dim=0)
            step_no_split = step_no_split[0].unsqueeze(1), step_no_split[1].unsqueeze(1)

            # Create Subsequences if used (Either because many-to-one or limitation to mem size)
            if self._sub_sequence is not None and self._sub_sequence < ctx_data_split[0].shape[0]:
                split_subsequence = lambda split: tuple(
                    map(lambda x: self._split_in_subsequences(x, self._sub_sequence, self._sub_sequence_stride), split))
                ctx_data_split = split_subsequence(ctx_data_split)
                Y_split = split_subsequence(Y_split)
                Y_hat_split = split_subsequence(Y_hat_split)
                step_no_split = split_subsequence(step_no_split)
            else:
                split_unsqueeze = lambda split: tuple(
                    map(lambda x: x.unsqueeze(0), split))
                ctx_data_split = split_unsqueeze(ctx_data_split)
                Y_split = split_unsqueeze(Y_split)
                Y_hat_split = split_unsqueeze(Y_hat_split)
                step_no_split = split_unsqueeze(step_no_split)

            # Generate Samples
            no_of_samples = tuple([s.shape[0] for s in ctx_data_split])
            score_param = self._get_score_param(c_data)
            pack_tuple = lambda split_idx, _idx, alpha_t:\
                (ctx_data_split[split_idx][_idx],
                 Y_split[split_idx][_idx],
                 Y_hat_split[split_idx][_idx],
                 alpha_t,
                 step_no_split[split_idx][_idx],
                 torch.tensor([ts_id_enc], dtype=torch.int),
                 score_param)
            self._pack_data_in_splits(splits, pack_tuple, no_of_samples, alphas)

        return DataLoader(EpsSelTrainerDataset(*zip(*splits[0]), has_history=False),
                          batch_size=self._batch_size, shuffle=True, collate_fn=PadCollate(), num_workers=num_worker), \
               DataLoader(EpsSelTrainerDataset(*zip(*splits[1]), has_history=False),
                          batch_size=self._batch_size, shuffle=True,
                          collate_fn=PadCollate(), num_workers=num_worker)

    def get_data_loader_naive_mix(self, calib_data: [PICalibData], Y_hat: List, fc_state_step: List,
                                  alphas: List[float], history_size, num_worker=0):
        splits = [], []

        for idx, c_data in enumerate(calib_data):
            # Prepare Context
            ctx_data, window_offset, ts_id_enc = self._get_calib_ctx(calib_data=c_data, Y_hat=Y_hat[idx],
                                                                     fc_state_step=fc_state_step[idx])
            step_no = torch.arange(start=c_data.step_offset, end=c_data.step_offset + c_data.Y_calib.shape[0],
                                   dtype=torch.long)
            # Split in Train/Val
            assert ctx_data.shape[0] == c_data.Y_calib[window_offset:].shape[0] == Y_hat[idx][window_offset:].shape[0] == step_no[window_offset:].shape[0]
            split_size = (ctx_data.shape[0] // 2) + 1
            # ToDo Split in train/val in a more advanced way
            ctx_data_split = torch.split(ctx_data, split_size, dim=0)
            Y_split = torch.split(c_data.Y_calib[window_offset:], split_size, dim=0)
            Y_hat_split = torch.split(Y_hat[idx][window_offset:], split_size, dim=0)
            step_no_split = torch.split(step_no[window_offset:], split_size, dim=0)
            step_no_split = step_no_split[0].unsqueeze(1), step_no_split[1].unsqueeze(1)

            # Create Subsequences if used
            split_subsequence = lambda split: tuple(map(lambda x: self._split_in_subsequences(
                x, window_len=min(history_size, x.shape[0])+1, stride=1,
                front_padding=torch.zeros((min(history_size, x.shape[0]), x.shape[1]), dtype=x.dtype)), split))

            ctx_data_split = split_subsequence(ctx_data_split)
            Y_split = split_subsequence(Y_split)
            Y_hat_split = split_subsequence(Y_hat_split)
            step_no_split = split_subsequence(step_no_split)

            # Generate Samples
            no_of_samples = tuple([s.shape[0] for s in ctx_data_split])
            score_param = self._get_score_param(c_data)

            pack_tuple = lambda split_idx, _b_idx, alpha_t:\
                (ctx_data_split[split_idx][_b_idx][-1:],
                 Y_split[split_idx][_b_idx][-1:], Y_hat_split[split_idx][_b_idx][-1:],
                 alpha_t,
                 step_no_split[split_idx][_b_idx][-1:],
                 torch.tensor([ts_id_enc], dtype=torch.int),
                 score_param,
                 torch.tensor([_b_idx], dtype=torch.int),
                 ctx_data_split[split_idx][_b_idx][:-1],
                 step_no_split[split_idx][_b_idx][:-1])
            self._pack_data_in_splits(splits, pack_tuple, no_of_samples, alphas)

        train_data = EpsSelTrainerDataset(*zip(*splits[0]), has_history=history_size is not None and history_size > 0)
        val_data = EpsSelTrainerDataset(*zip(*splits[1]), has_history=history_size is not None and history_size > 0)
        train_sampler = SubGroupMixSampler(train_data._ts_id, mix_count=self._batch_mix_count, batch_size=self._batch_size)
        val_sampler = SubGroupMixSampler(val_data._ts_id, mix_count=self._batch_mix_count, batch_size=self._batch_size)
        return DataLoader(train_data, batch_sampler=train_sampler, num_workers=num_worker), \
               DataLoader(val_data, batch_sampler=val_sampler, num_workers=num_worker)

    def _get_score_param(self, calib_data: PICalibData):
        score_param = calib_data.score_param
        assert len(score_param) <= 1  # For know only 1 score param is allowed
        tmp = next(iter(score_param.values()), None)
        if tmp is None:
            return False
        else:
            return torch.tensor([tmp], dtype=torch.float)

    def _pack_data_in_splits(self, splits, pack_func, no_of_samples, alphas):
        for split_idx, split_part in enumerate(splits):
            for _idx in range(no_of_samples[split_idx]):
                if self._no_train_alpha_needed():
                    dummy_alpha = torch.tensor([0.5], dtype=torch.float)
                    split_part.append(pack_func(split_idx, _idx, dummy_alpha))
                elif self._all_alpha_in_one_batch:
                    alphas_batch = torch.tensor(alphas, dtype=torch.float)
                    split_part.append(pack_func(split_idx, _idx, alphas_batch))
                else:
                    for alpha in alphas:
                        alpha = torch.tensor([alpha], dtype=torch.float)
                        split_part.append(pack_func(split_idx, _idx, alpha))

    @staticmethod
    def _split_in_subsequences(sequence, window_len, stride, front_padding=None):
        windows = unfold_window(sequence, window_len=window_len, stride=stride, M_past=front_padding)
        return windows


class EpsCtxMemoryMixin:
    def __init__(self, mem_size: int, keep_calib_eps: bool, store_step_no=False, mix_data_count=None,
                 memory_in_ram=False, pub_inference=False) -> None:
        self._memory = FiFoMemory(mem_size, store_step_no=store_step_no)
        self._mem_size = mem_size
        self._keep_calib_eps = keep_calib_eps
        self._store_step_no = store_step_no
        self._current_device = torch.device('cpu')
        self._mix_data_count = int(mix_data_count) if mix_data_count is not None else 0
        self._memory_in_ram = memory_in_ram
        self._pub_inference = pub_inference
        if self._mix_data_count > 0:
            self._mix_mem_dict = dict()
            self._mix_data_memory = [FiFoMemory(mem_size, store_step_no=store_step_no) for _ in range(self._mix_data_count)]
        else:
            self._mix_data_memory = []
            assert not self._pub_inference

    @property
    def max_mem_size(self):
        return self._mem_size

    def _fill_memory(self, calib_data: PICalibData, calib_artifacts: PICalibArtifacts,
                     mix_calib_data: List[PICalibData] = None, mix_calib_artifacts: List[PICalibArtifacts] = None):
        self._memory.clear()
        if not self._pub_inference:
            ctx_encoded, eps, step_no, history_state = self._encode_calib_data(
                c_data=calib_data, Y_hat=calib_artifacts.fc_Y_hat,
                fc_state_step=calib_artifacts.fc_state_step, calib_artifacts=calib_artifacts)
            eps = eps.to(self._current_device)
            if self._keep_calib_eps:
                self._memory.add_freezed(ctx_encoded, eps, step_no=step_no)
            else:
                self._memory.add_transient(ctx_encoded, eps, step_no=step_no)
        else:
            history_state = None

        if self._mix_data_count > 0:
            assert mix_calib_data is not None and mix_calib_artifacts is not None
            for m in self._mix_data_memory:
                m.clear()
            self._mix_mem_dict.clear()
            assert len(mix_calib_data) == len(mix_calib_artifacts)
            extra_ctx_history_states = []  # state for the optional history compression
            for idx, c_data in enumerate(mix_calib_data):
                self._mix_mem_dict[c_data.ts_id] = len(self._mix_mem_dict)
                ctx_encoded, eps, step_no, h_state = self._encode_calib_data(
                    c_data=c_data, Y_hat=mix_calib_artifacts[idx].fc_Y_hat,
                    fc_state_step=mix_calib_artifacts[idx].fc_state_step, calib_artifacts=mix_calib_artifacts[idx])
                extra_ctx_history_states.append(h_state)
                eps = eps.to(self._current_device)
                if self._keep_calib_eps:
                    self._mix_data_memory[self._resolve_mix_ts_id(c_data.ts_id)]\
                        .add_freezed(ctx_encoded, eps, step_no=step_no)
                else:
                    self._mix_data_memory[self._resolve_mix_ts_id(c_data.ts_id)]\
                        .add_transient(ctx_encoded, eps, step_no=step_no)

                #if mem_ctx is None:
                #    mem_ctx = ctx_encoded
                #    mem_eps = eps.to(self._current_device)
                #    mem_step = step_no
                #else:
                #    mem_ctx = torch.cat((mem_ctx, ctx_encoded), dim=0)
                #    mem_eps = torch.cat((mem_eps, eps.to(self._current_device)), dim=0)
                #    mem_step = torch.cat((mem_step, step_no), dim=0) if mem_step is not None else None
        else:
            extra_ctx_history_states = None

        return history_state, extra_ctx_history_states

    def _encode_calib_data(self, c_data: PICalibData, Y_hat, fc_state_step, calib_artifacts: PICalibArtifacts):
        if self._store_step_no:
            step_no = torch.arange(start=c_data.step_offset, end=c_data.step_offset + c_data.Y_calib.shape[0],
                                   dtype=torch.long, device=self._current_device)
        else:
            step_no = None
        ctx_data, window_offset, ts_id_enc = self._get_calib_ctx(calib_data=c_data, Y_hat=Y_hat,
                                                                 fc_state_step=fc_state_step)  # [calib_size, ctx_size]
        ctx_encoded, ctx_history_state = self._encode_ctx(context=ctx_data.to(self._current_device),
                                                          step_no=step_no)  # [calib_size, ctx_emb_size]
        eps = score.get(Y=c_data.Y_calib, Y_hat=Y_hat, **c_data.score_param)  # [calib_size, *]
        calib_artifacts.eps = eps
        return ctx_encoded, eps, step_no, ctx_history_state

    def _add_step_to_mem(self, ctx, eps, step):
        if not self._pub_inference:
            self._memory.add_transient(ctx.to(self._current_device), eps.to(self._current_device), step.to(self._current_device))

    def _get_data_with_mix_mem_ctx(self, selected_mix_ts=None, selected_subsets=None, selected_mem_samples=None):
        base_mem = [self._memory.ctx] if not self._pub_inference else []
        if selected_mix_ts is None:
            m = torch.cat(base_mem + [m.ctx for m in self._mix_data_memory], dim=0)
        elif selected_subsets is not None:
            m = torch.cat(base_mem + [m.ctx[selected_subsets[idx], :] for idx, m in enumerate(self._mix_data_memory)], dim=0)
        else:
            m = torch.cat(base_mem + [m.ctx for m in self._get_mix_mem_sel(selected_mix_ts)], dim=0)
        if selected_mem_samples is not None:
            return m[selected_mem_samples]
        else:
            return m

    def _get_data_with_mix_mem_eps(self, selected_mix_ts=None, selected_subsets=None, selected_mem_samples=None):
        base_mem = [self._memory.eps] if not self._pub_inference else []
        if selected_mix_ts is None:
            m = torch.cat(base_mem + [m.eps for m in self._mix_data_memory], dim=0)
        elif selected_subsets is not None:
            m = torch.cat(base_mem + [m.eps[selected_subsets[idx], :] for idx, m in enumerate(self._mix_data_memory)], dim=0)
        else:
            m = torch.cat(base_mem + [m.eps for m in self._get_mix_mem_sel(selected_mix_ts)], dim=0)
        if selected_mem_samples is not None:
            return m[selected_mem_samples]
        else:
            return m

    def _get_data_with_mix_mem_step(self, selected_mix_ts=None, selected_subsets=None, selected_mem_samples=None):
        base_mem = [self._memory.step_no] if not self._pub_inference else []
        if selected_mix_ts is None:
            m = torch.cat(base_mem + [m.step_no for m in self._mix_data_memory], dim=0)
        elif selected_subsets is not None:
            m = torch.cat(base_mem + [m.step_no[selected_subsets[idx], :] for idx, m in enumerate(self._mix_data_memory)], dim=0)
        else:
            m = torch.cat(base_mem + [m.step_no for m in self._get_mix_mem_sel(selected_mix_ts)], dim=0)
        if selected_mem_samples is not None:
            return m[selected_mem_samples]
        else:
            return m

    def _get_mix_mem_sel(self, ts_ids: List):
        return [self._get_mix_mem(_id) for _id in ts_ids]

    def _get_mix_mem(self, ts_id):
        return self._mix_data_memory[self._resolve_mix_ts_id(ts_id)]

    def _add_step_to_mix_mem(self, ts_id, ctx, eps, step):
        self._mix_data_memory[self._resolve_mix_ts_id(ts_id)]\
            .add_transient(ctx.to(self._current_device), eps.to(self._current_device), step.to(self._current_device))

    def _resolve_mix_ts_id(self, ts_id):
        return self._mix_mem_dict[ts_id]

    def to_device(self, device):
        if self._memory_in_ram:
            device = torch.device('cpu')
        self._memory.to(device=device)
        if self._mix_data_count > 0:
            for m in self._mix_data_memory:
                m.to(device=device)
        self._current_device = device

    @abstractmethod
    def _encode_ctx(self, context, step_no) -> Tuple[torch.tensor, torch.tensor]:
        pass

    @abstractmethod
    def _get_calib_ctx(self, calib_data, Y_hat, fc_state_step=None) -> Tuple[torch.tensor, int, int]:
        """
        :return: context of calib data, window_offset
        """
        pass


def get_loss_func(width_loss_weight, coverage_loss_weight, chung_weight, loss_mode):

    def get_loss(Y, q_low, q_high, low_alpha, high_alpha, base_alphas, **kwargs):
        """
        :param Y:            [batch_size, 1] or [batch, batch_size, 1]
        :param q_low:        [(batch_size or 1) * batches * no_alphas , 1]
        :param q_high:       [(batch_size or 1) * batches * no_alphas, 1]
        :param low_alpha:    [1] or [batch_size, 1]
        :param high_alpha:   [1] or [batch_size, 1]
        :param base_alphas   [1] or List or [batch, no_alphas]
        :return: Tuple:
                  1) Tensor[float] Overall Loss (backprop here)
                  2) Dict[str, Tensor[float]] individual losses
        """
        low_alpha = low_alpha.repeat(q_low.shape[0], 1) if torch.numel(low_alpha) == 1 else low_alpha
        high_alpha = high_alpha.repeat(q_high.shape[0], 1) if torch.numel(high_alpha) == 1 else high_alpha

        if len(base_alphas.shape) == 1:  # One Batch  (different alpha results a concated)
            no_alphas = base_alphas.shape[0]
            assert q_high.shape[0] % no_alphas == 0
            eval_len = q_high.shape[0] // no_alphas
            per_alpha_len = q_high.shape[0] // no_alphas
            assert Y.shape[0] == eval_len
            Y = Y[-eval_len:]   # In case interval prediction has offset at begging
        else:  # Multi Batch    (first different batches then different alphas after another)
            no_alphas = base_alphas.shape[1]
            batches = base_alphas.shape[0]
            assert q_high.shape[0] % (no_alphas * batches) == 0
            eval_len = q_high.shape[0] // (no_alphas * batches)
            per_alpha_len = q_high.shape[0] // no_alphas
            Y = Y[:, -eval_len:, :]     # In case interval prediction has offset at begging
            Y = Y.reshape(-1, 1)
        Y = Y.repeat(no_alphas, 1)      # Repeat for each alpha
        loss_mask = kwargs.get('loss_mask', None)
        loss_low = pinball_loss(q_low, Y, low_alpha, mask=loss_mask).sum()
        loss_high = pinball_loss(q_high, Y, high_alpha, mask=loss_mask).sum()
        if chung_weight > 0:
            assert loss_mask is None
            loss_chung_low = chung_weight * chung_calib_loss(q_low, Y, low_alpha).sum()
            loss_chung_high = chung_weight * chung_calib_loss(q_high, Y, high_alpha).sum()
        else:
            loss_chung_low = loss_chung_high = torch.tensor([0], dtype=torch.float, device=Y.device)
        if width_loss_weight > 0:
            loss_width = width_loss_weight * width_loss(q_low, q_high, mask=loss_mask).sum()
        else:
            loss_width = torch.tensor([0], dtype=torch.float, device=Y.device)
        loss_coverage = torch.tensor([0], dtype=torch.float, device=Y.device)
        if coverage_loss_weight > 0:
            assert loss_mask is None
            Y_c, q_low_c, q_high_c = torch.split(Y, per_alpha_len, dim=0), torch.split(q_low, per_alpha_len, dim=0),\
                                     torch.split(q_high, per_alpha_len, dim=0)
            for idx, alpha in enumerate(base_alphas[0]):
                loss_coverage += coverage_loss_weight * coverage_loss(Y_c[idx], q_low_c[idx], q_high_c[idx], alpha).mean()
        return loss_low + loss_high + loss_width + loss_coverage + loss_chung_low + loss_chung_high, \
               {'loss_low_pinball': loss_low, 'loss_high_pinball': loss_high, 'loss_width': loss_width,
                'coverage_loss': loss_coverage, 'loss_chung_low': loss_chung_low, 'loss_chung_high': loss_chung_high}

    def get_loss_abs(Y, Y_hat, q_high, high_alpha, base_alphas, **kwargs):
        """
        :param Y:            [batch_size, 1] or [batch, batch_size, 1]
        :param q_high:       [(batch_size or 1) * batches * no_alphas, 1]
        :param high_alpha:   [1] or [batch_size, 1]
        :param base_alphas   [1] or List or [batch, no_alphas]
        :return: Tuple:
                  1) Tensor[float] Overall Loss (backprop here)
                  2) Dict[str, Tensor[float]] individual losses
        """
        high_alpha = high_alpha.repeat(q_high.shape[0], 1) if torch.numel(high_alpha) == 1 else high_alpha
        if len(base_alphas.shape) == 1:  # One Batch  (different alpha results a concated)
            no_alphas = base_alphas.shape[0]
            assert q_high.shape[0] % no_alphas == 0
            eval_len = q_high.shape[0] // no_alphas
            per_alpha_len = q_high.shape[0] // no_alphas
            assert Y.shape[0] == eval_len
            Y = Y[-eval_len:]   # In case interval prediction has offset at begging
        else:  # Multi Batch    (first different batches then different alphas after another)
            no_alphas = base_alphas.shape[1]
            batches = base_alphas.shape[0]
            assert q_high.shape[0] % (no_alphas * batches) == 0
            eval_len = q_high.shape[0] // (no_alphas * batches)
            per_alpha_len = q_high.shape[0] // no_alphas
            Y = Y[:, -eval_len:, :]     # In case interval prediction has offset at begging
            Y = Y.reshape(-1, 1)
        Y = Y.repeat(no_alphas, 1)      # Repeat for each alpha
        eps = torch.abs(calc_residuals(Y_hat, Y))
        eps_pred = torch.abs(calc_residuals(Y_hat, q_high))
        loss_mask = kwargs.get('loss_mask', None)
        loss_high = pinball_loss(eps_pred, eps, high_alpha, mask=loss_mask).sum()
        if width_loss_weight > 0:
            loss_width = width_loss_weight * width_loss(torch.neg(eps_pred), eps_pred, mask=loss_mask).sum()
        else:
            loss_width = torch.tensor([0], dtype=torch.float, device=eps_pred.device)
        return loss_high + loss_width, {'loss_high_pinball': loss_high, 'loss_width': loss_width}

    def get_loss_mse(eps, eps_predicted, **kwargs):
        loss_mask = kwargs.get('loss_mask', None)
        loss_mse = mse_loss(eps_predicted, eps, mask=loss_mask).sum()
        if width_loss_weight > 0:
            loss_width = width_loss_weight * width_loss(torch.neg(eps_predicted), eps_predicted, mask=loss_mask).sum()
        else:
            loss_width = torch.tensor([0], dtype=torch.float, device=eps_predicted.device)
        return loss_mse + loss_width, {'loss_mse': loss_mse, 'loss_width': loss_width}

    def get_loss_mixture(eps, eps_reference, weights, variance, **kwargs):
        assert eps.shape[0] == weights.shape[0]
        assert eps_reference.shape[1] == weights.shape[1]
        mix = torch.distributions.Categorical(probs=weights)
        comp = D.Normal(eps_reference, torch.clamp(variance, min=1e-5).expand_as(eps_reference))
        gmm = D.MixtureSameFamily(mix, comp)
        log_prob = gmm.log_prob(eps.T)
        log_prob = log_prob.mean()
        return -log_prob, {'loss_nll': -log_prob}

    def get_loss_epscdf(eps, eps_reference, weights, **kwargs):
        samples = eps.shape[0]
        no_references = eps_reference.shape[1]

        # 1) Sort Weights
        eps_reference, sort_idx = torch.sort(eps_reference, dim=1)
        weights = torch.gather(weights, dim=1, index=sort_idx)
        # 2) CumSum -> Alpha
        weights = torch.cumsum(weights, dim=1)
        #a = weights[:, :, -1].unsqueeze(-1).expand_as(weights)  # Not necessary because already normalized
        #weights = weights / a
        loss_pinball = pinball_loss(eps_reference.view(-1, 1), torch.repeat_interleave(eps, no_references, dim=0),
                                    weights.view(-1, 1), mask=None).mean()
        return loss_pinball, {'loss_pinball': loss_pinball}

    def get_loss_dist_stats(eps_stats, eps_stats_predicted, **kwargs):
        pass  # WIP

    if loss_mode == LOSS_MODE_ABS:
        return get_loss_abs
    elif loss_mode == LOSS_MODE_RES:
        return get_loss
    elif loss_mode == LOSS_MODE_MSE:
        return get_loss_mse
    elif loss_mode == LOSS_MODE_EPS_CDF:
        return get_loss_epscdf
    elif loss_mode == LOSS_MODE_MIX:
        return get_loss_mixture
    else:
        raise NotImplemented("Loss not available!")
