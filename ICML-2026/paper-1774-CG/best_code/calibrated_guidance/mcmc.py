from pyro.infer import RandomWalkKernel


class PotentialRandomWalkKernel(RandomWalkKernel):
    def __init__(self, potential_fn, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.potential_fn = potential_fn

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        self._energy_last = self.potential_fn(self._initial_params)
