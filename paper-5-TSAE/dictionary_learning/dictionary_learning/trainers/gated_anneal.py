"""
Implements the training scheme for a gated SAE described in https://arxiv.org/abs/2404.16014
"""

import torch as t
from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import GatedAutoEncoder
from collections import namedtuple

class GatedAnnealTrainer(SAETrainer):
    """
    Gated SAE training scheme with p-annealing.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class: type = GatedAutoEncoder,
                 lr: float = 3e-4,
                 warmup_steps: int = 1000, # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps: Optional[int] = 2000, # sparsity warmup period at start of training
                 decay_start: Optional[int] = None, # decay learning rate after this many steps
                 sparsity_function: str = 'Lp^p', # Lp or Lp^p
                 initial_sparsity_penalty: float = 1e-1, # equal to l1 penalty in standard trainer
                 anneal_start: int = 15000, # step at which to start annealing p
                 anneal_end: Optional[int] = None, # step at which to stop annealing, defaults to steps-1
                 p_start: float = 1, # starting value of p (constant throughout warmup)
                 p_end: float = 0, # annealing p_start to p_end linearly after warmup_steps, exact endpoint excluded
                 n_sparsity_updates: int | str = 10, # number of times to update the sparsity penalty, at most steps-anneal_start times
                 sparsity_queue_length: int = 10, # number of recent sparsity loss terms, onle needed for adaptive_sparsity_penalty
                 resample_steps: Optional[int] = None, # number of steps after which to resample dead neurons
                 device: Optional[str] = None,
                 seed: Optional[int] = 42,
                 wandb_name: str = 'GatedAnnealTrainer',
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        # initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.ae = dict_class(activation_dim, dict_size)
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
                
        self.lr = lr
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end if anneal_end is not None else steps
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start # p is set in self.loss()
        self.next_p = None # set in self.loss()
        self.lp_loss = None # set in self.loss()
        self.scaled_lp_loss = None # set in self.loss()
        if n_sparsity_updates == "continuous":
            self.n_sparsity_updates = self.anneal_end - anneal_start +1
        else:
            self.n_sparsity_updates = n_sparsity_updates
        self.sparsity_update_steps = t.linspace(anneal_start, self.anneal_end, self.n_sparsity_updates, dtype=int)
        self.p_values = t.linspace(p_start, p_end, self.n_sparsity_updates)
        self.p_step_count = 0
        self.sparsity_coeff = initial_sparsity_penalty # alpha
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []

        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.steps = steps
        self.logging_parameters = ['p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'sparsity_coeff']
        self.seed = seed
        self.wandb_name = wandb_name

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr, betas=(0.0, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()
            self.ae.encoder.weight[deads][:n_resample] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads][:,:n_resample] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads][:n_resample] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.
        
    def lp_norm(self, f, p):
        norm_sq = f.pow(p).sum(dim=-1)
        if self.sparsity_function == 'Lp^p':
            return norm_sq.mean()
        elif self.sparsity_function == 'Lp':
            return norm_sq.pow(1/p).mean()
        else:
            raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
        
    def loss(self, x:t.Tensor, step:int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)
        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        fs = f_gate # feature activation that we use for sparsity term
        lp_loss = self.lp_norm(fs, self.p)
        scaled_lp_loss = lp_loss * self.sparsity_coeff * sparsity_scale
        self.lp_loss = lp_loss
        self.scaled_lp_loss = scaled_lp_loss

        if self.next_p is not None:
            lp_loss_next = self.lp_norm(fs, self.next_p)
            self.sparsity_queue.append([self.lp_loss.item(), lp_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]
    
        if step in self.sparsity_update_steps:
            # check to make sure we don't update on repeat step:
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt sparsity penalty alpha
                if self.next_p is not None:
                    local_sparsity_new = t.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = t.tensor([i[1] for i in self.sparsity_queue]).mean()
                    self.sparsity_coeff = self.sparsity_coeff * (local_sparsity_new / local_sparsity_old).item()
                # Update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates-1:
                    self.next_p = self.p_values[self.p_step_count+1].item()
                else:
                    self.next_p = self.p_end
                self.p_step_count += 1

        # Update dead feature count
        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0       
            
        loss = L_recon + scaled_lp_loss + L_aux
    
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'mse_loss' : L_recon.item(),
                    'aux_loss' : L_aux.item(),
                    'loss' : loss.item(),
                    'p' : self.p,
                    'next_p' : self.next_p,
                    'lp_loss' : lp_loss.item(),
                    'sparsity_loss' : scaled_lp_loss.item(),
                    'sparsity_coeff' : self.sparsity_coeff,
                }
            )
        
    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == self.resample_steps - 1:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    # @property
    # def config(self):
    #     return {
    #         'trainer_class' : 'GatedSAETrainer',
    #         'activation_dim' : self.ae.activation_dim,
    #         'dict_size' : self.ae.dict_size,
    #         'lr' : self.lr,
    #         'l1_penalty' : self.l1_penalty,
    #         'warmup_steps' : self.warmup_steps,
    #         'device' : self.device,
    #         'wandb_name': self.wandb_name,
    #     }
        
    @property
    def config(self):
        return {
            'trainer_class' : "GatedAnnealTrainer",
            'dict_class' : "GatedAutoEncoder",
            'activation_dim' : self.activation_dim,
            'dict_size' : self.dict_size,
            'lr' : self.lr,
            'sparsity_function' : self.sparsity_function,
            'sparsity_penalty' : self.sparsity_coeff,
            'p_start' : self.p_start,
            'p_end' : self.p_end,
            'anneal_start' : self.anneal_start,
            'sparsity_queue_length' : self.sparsity_queue_length,
            'n_sparsity_updates' : self.n_sparsity_updates,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'decay_start' : self.decay_start,
            'steps' : self.steps,
            'seed' : self.seed,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name' : self.wandb_name,
        }
