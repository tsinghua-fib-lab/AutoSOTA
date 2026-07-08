from abc import ABC, abstractmethod

import torch
from torch import nn


class EpsMemory(nn.Module, ABC):
    """
    Memory Represents epsilons and there corresponding context
    """
    @property
    @abstractmethod
    def ctx(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def ctx_chronological(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def eps(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def eps_chronological(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def step_no(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def step_no_chronological(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def empty(self):
        pass

    @abstractmethod
    def clear(self):
        pass  # Clear the memory


class FiFoMemory(EpsMemory):
    """
    FIFO Memory (not ordered!)
    Supports also a fixed (freezed) amount of entries (e.g. for prioritized calib data)
    but these all transient entries get reset for each add_freeze
    """
    def __init__(self, mep_steps, store_step_no) -> None:
        super().__init__()
        self._mem_steps = mep_steps
        self._first_transient_step = 0
        self._current_transient_step = 0
        self._fill_len = 0
        self._ctx = None
        self._eps = None
        self._step_no = None
        self._store_step_no = store_step_no

    def _init_mem(self, ctx_size, eps_size, device):
        self._ctx = torch.zeros((self._mem_steps, ctx_size), device=device)
        self._eps = torch.zeros((self._mem_steps, eps_size), device=device)
        if self._store_step_no:
            self._step_no = torch.zeros((self._mem_steps,), dtype=torch.long, device=device)
        self._first_transient_step = 0
        self._current_transient_step = 0
        self._fill_len = 0

    def clear(self):
        self._ctx = None
        self._eps = None
        self._step_no = None
        self._first_transient_step = 0
        self._current_transient_step = 0
        self._fill_len = 0

    def add_freezed(self, cxt, eps, step_no=None):
        """ Add m observations as freezed entries (Note: all transient entries get deleted)
           :param cxt: dim: [m,ctx_len]
           :param eps: dim: [m,eps_len] (eps len typically 1)
           :return:
        """
        no_of_steps, ctx_size = cxt.shape
        if self._ctx is None:
            _, eps_size = eps.shape
            self._init_mem(ctx_size, eps_size, cxt.device)

        self._ctx[self._first_transient_step:self._first_transient_step+no_of_steps] = cxt
        self._eps[self._first_transient_step:self._first_transient_step+no_of_steps] = eps
        if self._store_step_no:
            if step_no is None:
                raise ValueError("Step Number required!")
            self._step_no[self._first_transient_step:self._first_transient_step+no_of_steps] = step_no
        self._first_transient_step += no_of_steps
        self._current_transient_step = self._first_transient_step
        self._fill_len = self._first_transient_step

    def add_transient(self, cxt, eps, step_no=None):
        """ Add m observations
        :param cxt: dim: [m,ctx_len]
        :param eps: dim: [m,eps_len] (eps len typically 1)
        :return:
        """
        no_of_steps, ctx_size = cxt.shape
        if self._ctx is None:
            _, eps_size = eps.shape
            self._init_mem(ctx_size, eps_size, cxt.device)

        for i in range(no_of_steps):
            # ToDO Optimize to add as one
            if self._current_transient_step >= self._mem_steps:
                self._current_transient_step = self._first_transient_step
            self._ctx[self._current_transient_step, :] = cxt[i, :]
            self._eps[self._current_transient_step, :] = eps[i, :]
            if self._store_step_no:
                if step_no is None:
                    raise ValueError("Step Number required!")
                self._step_no[self._current_transient_step] = step_no[i]

            self._current_transient_step += 1
            if self._fill_len < self._mem_steps:
                self._fill_len += 1

    @property
    def ctx(self) -> torch.Tensor:
        if self._fill_len < self._mem_steps:
            return self._ctx[:self._fill_len]
        else:
            return self._ctx

    @property
    def ctx_chronological(self) -> torch.Tensor:
        if self._fill_len < self._mem_steps:
            return self.ctx
        else:
            return self._ctx[self._get_chrono_index()]

    @property
    def eps(self) -> torch.Tensor:
        if self._fill_len < self._mem_steps:
            return self._eps[:self._fill_len]
        else:
            return self._eps

    @property
    def eps_chronological(self) -> torch.Tensor:
        if self._fill_len < self._mem_steps:
            return self.eps
        else:
            return self._eps[self._get_chrono_index()]

    @property
    def step_no(self) -> torch.Tensor:
        if not self._store_step_no:
            raise ValueError("")
        if self._fill_len < self._mem_steps:
            return self._step_no[:self._fill_len]
        else:
            return self._step_no

    @property
    def step_no_chronological(self) -> torch.Tensor:
        if not self._store_step_no:
            raise ValueError("")
        if self._fill_len < self._mem_steps:
            return self.step_no
        else:
            return self._step_no[self._get_chrono_index()]

    @property
    def empty(self):
        return self._ctx is None

    def _get_chrono_index(self) -> torch.Tensor:
        first_step = self._current_transient_step % self._mem_steps
        indices = torch.cat((torch.arange(first_step, self._mem_steps), torch.arange(0, first_step)))
        return indices

    def __len__(self):
        return self._fill_len
