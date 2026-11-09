"""Initialize modern PyTorch bookkeeping for the original Lookahead algorithm.

The original 2021 Lookahead omitted Optimizer.__init__, which breaks zero_grad
and state_dict under torch 2.4. Ralamb updates and slow-weight interpolation are
the contestant's unchanged code. This adapter also persists the slow weights.
"""
import torch


def ranger_factory(source_module):
    class CompatibleLookahead(source_module.Lookahead):
        def __init__(self, base_optimizer, alpha=.5, k=6):
            torch.optim.Optimizer.__init__(self, base_optimizer.param_groups, base_optimizer.defaults)
            source_module.Lookahead.__init__(self, base_optimizer, alpha, k)

        def state_dict(self):
            value = super().state_dict()
            value['lookahead_slow_weights'] = [[p.detach().clone() for p in group] for group in self.slow_weights]
            value['lookahead_alpha'] = self.alpha
            value['lookahead_k'] = self.k
            return value

        def load_state_dict(self, value):
            super().load_state_dict(value)
            self.optimizer.param_groups = self.param_groups
            self.optimizer.state = self.state
            if 'lookahead_slow_weights' in value:
                for current, saved in zip(self.slow_weights, value['lookahead_slow_weights']):
                    for destination, source in zip(current, saved):
                        destination.copy_(source.to(destination.device))
            self.alpha = value.get('lookahead_alpha', self.alpha)
            self.k = value.get('lookahead_k', self.k)

    def create(parameters, alpha=.5, k=6, **kwargs):
        return CompatibleLookahead(source_module.Ralamb(parameters, **kwargs), alpha, k)

    return create
