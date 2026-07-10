from types import MethodType

import torch.nn.functional as F
from pc_e import PCE


# Define CELoss version of PCE
def use_CrossEntropyLoss(pc_module):
    """CELoss to avoid vanishing grads with state optim..."""

    # Define the new loss method using CrossEntropyLoss
    def class_loss(self, y_pred, y):
        return F.cross_entropy(y_pred, y, reduction="sum", label_smoothing=0.1)

    # Override pc_module.class_loss with the new method
    pc_module.class_loss = MethodType(class_loss, pc_module)

    return pc_module


# Define state optim version of PCE
class PC_States(PCE):
    def minimize_error_energy(self, x, y):
        # Recycle iters and e_lr for state optimization, and store final states
        self.states = super().minimize_state_energy(x, y, self.iters, self.e_lr)

    def E_local(self, x, y):
        return super().E_states_only(x, y, self.states)

    # No need to redefine forward or y_pred:
    # For prediction, they set all errors to zero and simply to the correct prediction.
    # Therefore, we only need to adapt the training procedure.


# Define backprop version of PCE
class BackpropMSE(PCE):
    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["y"]
        self.forward(x)  # sets all errors to 0
        return self.class_loss(self.y_pred(x), y) / self.batch_size


def get_pc_variant(algorithm: str, USE_CROSSENTROPY_INSTEAD_OF_MSE: bool):
    if algorithm == "EO":
        pctype = PCE
    elif algorithm == "SO":
        pctype = PC_States
    elif algorithm == "BP":
        pctype = BackpropMSE
    else:
        raise NotImplementedError("Choose one of these options: EO | SO | BP")

    def pc_maker(*args, **kwargs):
        pc = pctype(*args, **kwargs)

        if USE_CROSSENTROPY_INSTEAD_OF_MSE:
            pc = use_CrossEntropyLoss(pc)

        return pc

    return pc_maker
