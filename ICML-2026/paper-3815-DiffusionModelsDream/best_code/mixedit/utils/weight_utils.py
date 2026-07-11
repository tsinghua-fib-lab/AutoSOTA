import abc
import torch

_WEIGHT_FN = {}


def register_weight_fn(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _WEIGHT_FN:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _WEIGHT_FN[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_weight_fn(name):
    return _WEIGHT_FN[name]


class SchedulerWeight(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def forward(self, t):
        pass

    @abc.abstractmethod
    def cum_weight_fn(self, t):
        pass

    @property
    def norm_const(self):
        return self.cum_weight_fn(1)


@register_weight_fn(name="default")
class DefaultWeight(SchedulerWeight):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, t):
        return 1.
    
    def cum_weight_fn(self, t):
        return t
    

@register_weight_fn(name="step")
class StepWeight(SchedulerWeight):
    def __init__(self, **kwargs):
        super().__init__()
        self.left = kwargs.get("left", 0.3)
        self.right = kwargs.get("right", 0.75)#0.6)
        self.ub = kwargs.get("ub", 1)
        self.lb = kwargs.get("lb", 1e-4)

    def forward(self, t):
        return torch.where(
            (t>self.left)&(t<self.right), 
            torch.ones_like(t)*self.ub, 
            torch.ones_like(t)*self.lb
        )
    
    def cum_weight_fn(self, t):
        if isinstance(t, torch.Tensor):
            # left_flag = t <= self.left
            # mid_flag = (t>self.left)&(t < self.right)
            # right_flag = t >= self.right
            return torch.where(
                t<self.left, 
                t * self.lb, 
                torch.where(
                    t>self.right, 
                    self.left * self.lb + (self.right - self.left) * self.ub + (t - self.right) * self.lb, 
                    self.left * self.lb + (t - self.left) * self.ub
                )
            )
        else:
            return t * self.lb if t < self.left else \
                (self.left * self.lb + (self.right - self.left) * self.ub + (t - self.right) * self.lb if t>self.right else \
                    self.left * self.lb + (t - self.left) * self.ub)