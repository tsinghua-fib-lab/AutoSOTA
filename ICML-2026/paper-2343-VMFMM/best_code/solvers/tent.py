"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
Adapted from https://github.com/mariodoebler/test-time-adaptation
"""
import torch
import torch.nn as nn

import logging

from copy import deepcopy
from functools import wraps


logger = logging.getLogger(__name__)


class TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.steps = cfg.OPTIM.STEPS
        self.episodic = getattr(cfg, 'EPISODIC', False)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.img_size = (224,224)

        # configure model and optimizer
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()

        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        self.window_length = cfg.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # setup for mixed-precision or single precision
        self.mixed_precision = cfg.MIXED_PRECISION
        self.scaler = torch.amp.GradScaler() if cfg.MIXED_PRECISION else None

    def forward(self, x):
        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)

                    # if specified, reset the model after a certain amount of update steps
                    self.performed_updates += 1
                    if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                        self.reset()

                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)

                # if specified, reset the model after a certain amount of update steps
                self.performed_updates += 1
                if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                    self.reset()

        return outputs

    def loss_calculation(self, x):
        """
        Loss calculation.
        """
        raise NotImplementedError

    def forward_and_adapt(self, x):
        """
        Forward and adapt the model on a batch of data.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'AdamW':
            return torch.optim.AdamW(self.params,
                                     lr=self.cfg.OPTIM.LR,
                                     betas=(self.cfg.OPTIM.BETA, 0.999),
                                     weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        coppied_model = deepcopy(model)
        return coppied_model

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()


def forward_decorator(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs): 
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = fn(self, *args, **kwargs)
        else:
            outputs = fn(self, *args, **kwargs)
        return outputs
    return decorator



class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class Tent_solver(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # setup loss function
        self.softmax_entropy = Entropy()

    def loss_calculation(self, x, clip_prototypes): #clip prototypes are the embedded texts
        imgs_test = x
        if clip_prototypes is not None:
            encoded_x = self.model(imgs_test)
            encoded_x = encoded_x / torch.linalg.norm(encoded_x, dim = -1, keepdims = True)
            outputs = 100 * encoded_x @ clip_prototypes.T
        else:
            outputs = self.model(imgs_test)
        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x, clip_prototypes = None): #clip prototypes are the embedded texts
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs, loss = self.loss_calculation(x, clip_prototypes = clip_prototypes)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x, clip_prototypes = clip_prototypes)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs
    

    @torch.enable_grad()
    def forward(self, x, clip_prototypes = None): #just to have similar semantics to other solvers
        if self.episodic:
            self.reset()
        return self.forward_and_adapt(x, clip_prototypes = clip_prototypes)

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

#%% Loading utils
class CfgNode(object): #Dumb class to remove the dependency from yacs of the original github https://github.com/mariodoebler/test-time-adaptation
    def __init__(self,):
        return None
    def __repr__(self): #custom print function
        return ("{ " + str(", ".join([f"'{k}': {v}" for k, v in [(k, repr(v)) for (k, v) in self.__dict__.items()]])) + " }")
    
def get_default_CfgNode():
    # Default values of conf.py in 
    CFG = CfgNode()
    CFG.MODEL = CfgNode()
    CFG.MODEL.USE_CLIP = True
    CFG.EPISODIC = False
    CFG.MIXED_PRECISION = True
    
    CFG.OPTIM = CfgNode()
    CFG.OPTIM.STEPS = 1 # Number of updates per batch
    CFG.OPTIM.LR = 1e-3 # Learning rate
    CFG.OPTIM.METHOD = 'Adam' # Optimizer choices: Adam, AdamW, SGD
    CFG.OPTIM.BETA = 0.9 # Beta1 for Adam based optimizers
    CFG.OPTIM.MOMENTUM = 0.9 # Momentum
    CFG.OPTIM.DAMPENING = 0.0 # Momentum dampening
    CFG.OPTIM.NESTEROV = True # Nesterov momentum # only used for method "SGD"
    CFG.OPTIM.WD = 0.0 # L2 regularization
    
    CFG.WINDOW_LENGTH = None
    return CFG

def get_cfg(method_name, batch_size = None, num_steps = 10, lr = 1e-3, episodic = False):
    cfg = get_default_CfgNode()

    if method_name == 'tent':
        # same as https://github.com/mariodoebler/test-time-adaptation/blob/main/classification/cfgs/imagenet_others/tent.yaml
        # when lr = 2.5e-4 and num_steps = 1
        cfg.OPTIM.METHOD = 'SGD' 
        cfg.OPTIM.STEPS = num_steps
        cfg.OPTIM.BETA = 0.9
        cfg.OPTIM.LR = lr
        cfg.OPTIM.WD = 0
        cfg.OPTIM.MOMENTUM = 0.9
        cfg.OPTIM.NESTEROV = True
        cfg.EPISODIC = episodic
    else:
        raise NotImplementedError(f'Got method {method_name} but only tent is supported.')    
    return cfg


