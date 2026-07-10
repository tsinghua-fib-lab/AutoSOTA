import torch
from pc_e import PCE


class TrackedPCE(PCE):
    def __init__(self, architecture, iters, e_lr, w_lr):
        super().__init__(architecture, iters, e_lr, w_lr)

        self.log_everything = True
        self.log_errors = []
        self.log_states = []

    # TRACK ERRORS
    def minimize_error_energy(self, x, y):
        self.log_errors.clear()
        return super().minimize_error_energy(x, y)

    def E(self, x, y):
        if self.log_everything:
            with torch.no_grad():
                self.log_errors.append(self.get_states_from_errors(x))

        return super().E(x, y)

    # TRACK STATES
    def minimize_state_energy(self, x, y, iters, s_lr):
        self.log_states.clear()
        return super().minimize_state_energy(x, y, iters, s_lr)

    def E_states_only(self, x, y, states):
        if self.log_everything:
            self.log_states.append([s.detach().clone() for s in states])

        return super().E_states_only(x, y, states)
