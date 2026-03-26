from torch import nn
from copy import deepcopy


class AttrScheduler:
    def __init__(self, module: nn.Module, attr_list, last_epoch=-1, verbose=False):
        self.module = module
        self.attr_list = attr_list
        self.base_attrs = [deepcopy(getattr(module, name)) for name in attr_list]
        self.last_epoch = last_epoch
        self.verbose = verbose

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_val(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_val

    def get_val(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_val(self, is_verbose, group, val, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose is True:
            if epoch is None:
                print(f'Adjusting value of group {group} to {val:.4e}.')
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting value of {group} to {val}.')

    def step(self, epoch=None):
        self._step_count += 1
        self.last_epoch += 1
        values = self.get_val()

        for i, (attr_name, value) in enumerate(zip(self.attr_list, values)):
            setattr(self.module, attr_name, value)
            # self.print_val(self.verbose, i, value, epoch)

        self._last_val = [getattr(self.module, attr_name) for attr_name in self.attr_list]


class StepAttr(AttrScheduler):
    """Decays the each specified attr by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        module (Optimizer): Wrapped module.
        attr_list (list[str]): List of attr names within module to update.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses attr = 1 for all groups
        >>> # attr = 1.     if epoch < 30
        >>> # lr = 1.5    if 30 <= epoch < 60
        >>> # lr = 2.25   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepAttr(optimizer, step_size=30, gamma=1.5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, module: nn.Module, attr_list, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(module, attr_list, last_epoch, verbose)

    def get_val(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [getattr(self.module, attr_name) for attr_name in self.attr_list]
        return [getattr(self.module, attr_name)*self.gamma for attr_name in self.attr_list]

    def _get_closed_form_attr(self):
        return [base_attr * self.gamma ** (self.last_epoch // self.step_size)
                for base_attr in self.base_attrs]


class ConstantAttr(AttrScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        module,
        attr_list,
        factor=1,
        total_iters=5,
        last_epoch=-1,
        verbose="deprecated",
    ):

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(module, attr_list, last_epoch, verbose)

    def get_val(self):
        # On first epoch multiple attr by val
        if self.last_epoch == 0:
            return [getattr(self.module, attr_name) * self.factor for attr_name in self.attr_list]

        # For all other epochs leave value as is
        if self.last_epoch != self.total_iters:
            return [getattr(self.module, attr_name) for attr_name in self.attr_list]

        # At last epoch (for multiplying) undo multiplication
        else:
            attr_list = []
            for attr_name in self.attr_list:
                temp_attr = getattr(self.module, attr_name)
                # update the attr
                temp_attr = temp_attr // self.factor if isinstance(temp_attr, int) else temp_attr / self.factor
                attr_list.append(temp_attr)
        return attr_list

    def _get_closed_form_val(self):
        return [
            base_attr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_attr in self.base_attrs
        ]
