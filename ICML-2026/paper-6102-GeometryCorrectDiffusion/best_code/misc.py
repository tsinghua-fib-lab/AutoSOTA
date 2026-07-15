from pathlib import Path
import numpy as np
import imageio
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from abc import ABC, abstractmethod
from piq import psnr, ssim, FID
import lpips
import prettytable
import wandb
import warnings
import tqdm
from typing import Any, Dict
import re as _re
from data import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry

def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)

def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)

def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils

def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)

def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


class Trajectory(nn.Module):
    """Class for recording and storing trajectory data."""

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj


__DIFFUSION_SCHEDULER__ = {}


def register_diffusion_scheduler(name: str):
    def wrapper(cls):
        if __DIFFUSION_SCHEDULER__.get(name, None):
            if __DIFFUSION_SCHEDULER__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DIFFUSION_SCHEDULER__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_diffusion_scheduler(name: str, **kwargs):
    if __DIFFUSION_SCHEDULER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DIFFUSION_SCHEDULER__[name](**kwargs)


class Scheduler(ABC):
    """
    Abstract base class for diffusion scheduler.

    Schedulers manage time steps, noise scales (sigma), scaling factors, and coefficients 
    used in diffusion stochastic/ordinary differential equations (SDEs/ODEs).
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps + 1 # include the initial step

    def discretize(self, time_steps):
        sigma_steps = self.get_sigma(time_steps[:-1])
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        self.sigma_steps = sigma_steps

    def tensorize(self, data):
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, list):
            return torch.tensor(data).float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        raise ValueError(f"Data type {type(data)} is not supported.") 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Noise Scheduling & Scaling Function 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_scaling(self, t):
        pass
    
    def get_sigma(self, t):
        pass
    
    def get_scaling_derivative(self, t):
        pass

    def get_sigma_derivative(self, t):
        pass

    def get_sigma_inv(self, sigma):
        pass
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Time & Sigma Range Function
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_t_min(self):
        pass

    def get_t_max(self):
        pass

    def get_discrete_time_steps(self, num_steps):
        pass

    def get_sigma_max(self):
        return self.get_sigma(self.get_t_max())

    def get_sigma_min(self):
        return self.get_sigma(self.get_t_min())
    
    def get_prior_sigma(self):
        # simga(t_max) * scaling(t_max)
        return self.get_sigma_max() * self.get_scaling(self.get_t_max())

    def summary(self):
        print('+' * 50)
        print('Diffusion Scheduler Summary')
        print('+' * 50)
        print(f"Scheduler       : {self.name}")
        print(f"Time Range      : [{self.get_t_min().item()}, {self.get_t_max().item()}]")
        print(f"Sigma Range     : [{self.get_sigma_min().item()}, {self.get_sigma_max().item()}]")
        print(f"Scaling Range   : [{self.get_scaling(self.get_t_min()).item()}, {self.get_scaling(self.get_t_max()).item()}]")
        print(f"Prior Sigma     : {self.get_prior_sigma().item()}")
        print('+' * 50)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For Iterating Over the Discretized Scheduler
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __iter__(self):
        self.pbar = tqdm.trange(self.num_steps) if self.verbose else range(self.num_steps)
        self.pbar_iter = iter(self.pbar)
        return self

    def __next__(self):
        try:
            step = next(self.pbar_iter)
            time, scaling, sigma, scaling_factor, factor = self.time_steps[step], self.scaling_steps[step], \
                self.sigma_steps[step], self.scaling_factor_steps[step], self.factor_steps[step]
            return self.pbar, time, scaling, sigma, factor, scaling_factor
        except StopIteration:
            raise StopIteration


@register_diffusion_scheduler('vp')
class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, num_steps, beta_max=20, beta_min=0.1, epsilon=1e-5, beta_type='linear'):
        super().__init__(num_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_type = beta_type
        self.epsilon = epsilon

        if beta_type == 'linear':
            self.n = 1
        elif beta_type == 'scaled_linear':
            self.n = 2
        else:
            raise NotImplementedError
        
        self.a = beta_max ** (1 / self.n) - beta_min ** (1 / self.n)
        self.b = beta_min ** (1 / self.n)

        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For VP Scheduler Only
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_beta(self, t):
        # beta(t) = (a * t + b) ^ n
        t = self.tensorize(t)
        return (self.a * t + self.b) ** self.n

    def get_beta_integrated(self, t):
        # beta_integrated(t) = [(a * t + b) ^ (n + 1) - b ^ (n + 1)] / a / (n + 1)
        t = self.tensorize(t)
        return ((self.a * t + self.b) ** (self.n + 1) - self.b ** (self.n + 1)) / self.a / (self.n + 1)

    def get_alpha(self, t):
        # alpha(t) = exp(-beta_integrated(t))
        t = self.tensorize(t)
        return torch.exp(-self.get_beta_integrated(t))

    def get_alpha_derivative(self, t):
        # alpha'(t) = -beta(t) * alpha(t)
        t = self.tensorize(t)
        return - self.get_beta(t) * self.get_alpha(t)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_scaling(self, t):
        # s(t) = sqrt(alpha(t))
        t = self.tensorize(t)
        return torch.sqrt(self.get_alpha(t))

    def get_sigma(self, t):
        # sigma(t) = sqrt(1 / alpha(t) - 1)
        t = self.tensorize(t)
        return torch.sqrt(1 / self.get_alpha(t) - 1)

    def get_scaling_derivative(self, t):
        # s'(t) = -s(t) * beta(t) / 2
        t = self.tensorize(t)
        return - self.get_scaling(t) * self.get_beta(t) / 2

    def get_sigma_derivative(self, t):
        # sigma'(t) = beta(t) / 2 / sigma(t) / alpha(t)
        t = self.tensorize(t)
        return self.get_beta(t) / 2 / self.get_sigma(t) / self.get_alpha(t)

    def get_sigma_inv(self, sigma):
        # t = {[a(n+1)log(sigma^2 + 1) + b^(n+1)]^(1/(n + 1)) - b}/a
        sigma = self.tensorize(sigma)
        return ((self.a * (self.n + 1) * torch.log(sigma ** 2 + 1) + self.b ** (self.n + 1)) ** (1 / (self.n + 1)) - self.b) / self.a

    def get_t_min(self):
        return self.tensorize(self.epsilon)
    
    def get_t_max(self):
        return self.tensorize(1)

    def get_discrete_time_steps(self, num_steps):
        return torch.linspace(1, self.epsilon, num_steps)


@register_diffusion_scheduler('ve')
class VEScheduler(Scheduler):
    """Variance Exploding Scheduler."""

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = sqrt(t)
        t = self.tensorize(t)
        return t.sqrt()

    def get_scaling(self, t):
        # s(t) = 1
        t = self.tensorize(t)
        return torch.ones_like(t)

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1 / 2 / sqrt(t)
        t = self.tensorize(t)
        return 1 / t.sqrt() / 2

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        t = self.tensorize(t)
        return torch.zeros_like(t)

    def get_sigma_inv(self, sigma):
        # t = sigma^2
        sigma = self.tensorize(sigma)
        return sigma ** 2

    def get_t_min(self):
        return self.tensorize(self.sigma_min ** 2)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max ** 2)

    def get_discrete_time_steps(self, num_steps):
        time_steps_fn = lambda r: self.sigma_max ** 2 * (self.sigma_min ** 2 / self.sigma_max ** 2) ** r
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps).float()


@register_diffusion_scheduler('edm')
class EDMScheduler(Scheduler):
    """
        EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Scheduler.
    """

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2, timestep='poly-7'):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        p = int(timestep.split('-')[1])
        self.time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = t
        return self.tensorize(t)

    def get_scaling(self, t):
        # s(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        return torch.zeros_like(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return self.tensorize(sigma)

    def get_t_min(self):
        return self.tensorize(self.sigma_min)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max)

    def get_discrete_time_steps(self, num_steps):
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([self.time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps)
    

@register_diffusion_scheduler('trigflow')
class TrigFlowScheduler(Scheduler):
    """TrigFlow (Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models) Scheduler."""
    def __init__(self, num_steps, sigma_d=1.0, sigma_max=100, sigma_min=1e-2):
        super().__init__(num_steps)
        self.sigma_d = sigma_d
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min 

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = tan(t)
        return torch.tan(self.tensorize(t))

    def get_scaling(self, t):
        # s(t) = cos(t)
        return torch.cos(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1 / cos^2(t)
        return 1 / torch.cos(self.tensorize(t)) ** 2

    def get_scaling_derivative(self, t):
        # s'(t) = -sin(t)
        return -torch.sin(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return torch.arctan(self.tensorize(sigma))

    def get_t_min(self):
        return self.get_sigma_inv(self.sigma_min)
    
    def get_t_max(self):
        return self.get_sigma_inv(self.sigma_max)

    def get_prior_sigma(self):
        return super().get_prior_sigma() * self.sigma_d
    
    def get_discrete_time_steps(self, num_steps):
        return torch.linspace(self.get_t_max().item(), self.get_t_min().item(), num_steps)


class DiffusionSDE:
    """
    Diffusion Stochastic Differential Equation (Diffusion SDE) for sampling via forward and reverse SDE processes.
    """
    def __init__(self, model, scheduler, solver='euler'):
        self.model = model
        self.scheduler = scheduler
        self.solver = solver
        self.device = next(model.parameters()).device
        if solver != 'euler':
            raise NotImplementedError
    
    def forward_sde(self, x0, t, num_steps=None, return_traj=False):
        pass

    def reverse_sde(self, x0, num_steps=None, return_traj=False):
        pass

    def get_start(self, batch_size):
        in_shape = self.model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=self.device) * self.scheduler.get_prior_sigma()
        return x_start


class Evaluator:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, eval_fn_list):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = {}
        for eval_fn in eval_fn_list:
            self.eval_fn[eval_fn.name] = eval_fn
        self.main_eval_fn_name = eval_fn_list[0].name

    def get_main_eval_fn(self):
        """
            return the first eval_fn by default
        """
        return self.eval_fn[self.main_eval_fn_name]

    def __call__(self, gt, measurement, x, reduction='mean'):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            results[eval_fn_name] = eval_fn(gt, measurement, x, reduction)
        return results

    def to_list(self, x):
        return x.cpu().detach().tolist()

    def report(self, gt, measurement, x):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, gt.shape)
        x0_flatten = gt.expand(broadcasted_shape).flatten(0, 1)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = measurement.expand((broadcasted_shape[0], *measurement.shape)).flatten(0, 1)

        for key, fn in self.eval_fn.items():
            value = fn(x0_flatten, y_flatten, x_flatten, reduction='none').reshape(broadcasted_shape[0], -1)
            result_dicts[key] = {
                'sample': self.to_list(value.permute(1, 0)),
                'mean': self.to_list(value.mean(0)),
                'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                'max': self.to_list(value.max(0)[0]),
                'min': self.to_list(value.min(0)[0]),
            }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        average, std = {}, {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            average[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
            std[key] = '{:.3f}'.format(np.std(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['mean' for _ in result_dicts.keys()])
        table.add_row(average.values())
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['std' for _ in result_dicts.keys()])
        table.add_row(std.values())

        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):
        for s in range(batch_size):
            log_dict = {key: result_dicts[key][get_eval_fn_cmp(key)][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][get_eval_fn_cmp(key)]) for key in result_dicts.keys()}
        new_log_dict = {key + '_all': value for key, value in log_dict.items()}
        wandb.log(new_log_dict)
        return


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass


__EVAL_FN__ = {}
__EVAL_FN_CMP__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        cls.name = name
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(ABC):
    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    @abstractmethod
    def __call__(self, gt, measurement, sample, reduction='none'):
        pass


@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return psnr(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return ssim(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the higher, the better

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        # self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')
        self.lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

    def evaluate_in_batch(self, gt, pred):
        batch_size = self.batch_size
        results = []
        for start in range(0, gt.shape[0], batch_size):
            res = self.lpips_fn(torch.clamp(gt[start:start + batch_size], -1.0, 1.0),
                                torch.clamp(pred[start:start + batch_size], -1.0, 1.0))
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none'):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res


def preprocess(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * 0.5 + 0.5
    images = (images - mean) / std
    return images.float()


def calculate_fid(real_loader, generated_loader, device='cuda'):
    # Load pretrained Inception model
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Modify the model to output features from the pre-logits layer
    model.to(device)
    model.eval()

    # Function to get features from a dataloader
    def get_features(loader):
        features = []
        for images in loader:
            images = preprocess(images)
            with torch.no_grad():
                images = images.to(device)
                output = model(images)
                features.append(output.cpu())
        features = torch.cat(features, dim=0)
        return features

    # Extract features
    real_features = get_features(real_loader)
    gen_features = get_features(generated_loader)

    # Compute FID score
    fid = FID()
    score = fid.compute_metric(real_features, gen_features)
    return score

def _coerce_numeric(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _coerce_numeric(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_coerce_numeric(v) for v in obj]
        return type(obj)(t)
    if isinstance(obj, str):
        s = obj.strip()
        if _re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except Exception:
                return obj
        if _re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", s):
            try:
                return float(s)
            except Exception:
                return obj
    return obj

DATA_PRESETS: Dict[str, Dict[str, Any]] = {'demo-ffhq': {'end_id': 10, 'name': 'image', 'resolution': 256, 'root': 'dataset/demo-ffhq', 'start_id': 0},
 'demo-imagenet': {'end_id': 10, 'name': 'image', 'resolution': 256, 'root': 'dataset/demo-imagenet', 'start_id': 0},
 'test-ffhq': {'end_id': 100, 'name': 'image', 'resolution': 256, 'root': 'dataset/test-ffhq', 'start_id': 0},
 'test-imagenet': {'end_id': 100, 'name': 'image', 'resolution': 256, 'root': 'dataset/test-imagenet', 'start_id': 0}}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {'ffhq256ddpm': {'model_config': {'attention_resolutions': 16,
                                  'channel_mult': '',
                                  'class_cond': False,
                                  'dropout': 0.0,
                                  'image_size': 256,
                                  'learn_sigma': True,
                                  'model_path': 'checkpoints/ffhq256.pt',
                                  'num_channels': 128,
                                  'num_head_channels': 64,
                                  'num_heads': 4,
                                  'num_heads_upsample': -1,
                                  'num_res_blocks': 1,
                                  'resblock_updown': True,
                                  'use_checkpoint': False,
                                  'use_fp16': False,
                                  'use_new_attention_order': False,
                                  'use_scale_shift_norm': True},
                 'name': 'ddpm'},
 'ffhq256ldm': {'diffusion_path': 'checkpoints/ldm_ffhq256.pt',
                'ldm_config': {'model': {'base_learning_rate': 2e-06,
                                         'params': {'channels': 3,
                                                    'cond_stage_config': '__is_unconditional__',
                                                    'first_stage_config': {'params': {'ckpt_path': None,
                                                                                      'ddconfig': {'attn_resolutions': [],
                                                                                                   'ch': 128,
                                                                                                   'ch_mult': [1, 2, 4],
                                                                                                   'double_z': False,
                                                                                                   'dropout': 0.0,
                                                                                                   'in_channels': 3,
                                                                                                   'num_res_blocks': 2,
                                                                                                   'out_ch': 3,
                                                                                                   'resolution': 256,
                                                                                                   'z_channels': 3},
                                                                                      'embed_dim': 3,
                                                                                      'lossconfig': {'target': 'torch.nn.Identity'},
                                                                                      'n_embed': 8192},
                                                                           'target': 'model.ldm.models.autoencoder.VQModelInterface'},
                                                    'first_stage_key': 'image',
                                                    'image_size': 64,
                                                    'linear_end': 0.0195,
                                                    'linear_start': 0.0015,
                                                    'log_every_t': 200,
                                                    'monitor': 'val/loss_simple_ema',
                                                    'num_timesteps_cond': 1,
                                                    'timesteps': 1000,
                                                    'unet_config': {'params': {'attention_resolutions': [8, 4, 2],
                                                                               'channel_mult': [1, 2, 3, 4],
                                                                               'image_size': 64,
                                                                               'in_channels': 3,
                                                                               'model_channels': 224,
                                                                               'num_head_channels': 32,
                                                                               'num_res_blocks': 2,
                                                                               'out_channels': 3},
                                                                    'target': 'model.ldm.modules.diffusionmodules.openaimodel.UNetModel'}},
                                         'target': 'model.ldm.models.diffusion.ddpm.LatentDiffusion'}},
                'name': 'ldm'},
 'imagenet256ddpm': {'model_config': {'attention_resolutions': '32,16,8',
                                      'channel_mult': '',
                                      'class_cond': False,
                                      'dropout': 0.0,
                                      'image_size': 256,
                                      'learn_sigma': True,
                                      'model_path': 'checkpoints/imagenet256.pt',
                                      'num_channels': 256,
                                      'num_head_channels': 64,
                                      'num_heads': 4,
                                      'num_heads_upsample': -1,
                                      'num_res_blocks': 2,
                                      'resblock_updown': True,
                                      'use_checkpoint': False,
                                      'use_fp16': False,
                                      'use_new_attention_order': False,
                                      'use_scale_shift_norm': True},
                     'name': 'ddpm'},
 'imagenet256ldm': {'diffusion_path': 'checkpoints/ldm_imagenet256.pt',
                    'ldm_config': {'model': {'base_learning_rate': 0.0001,
                                             'params': {'channels': 3,
                                                        'cond_stage_config': {'params': {'embed_dim': 512,
                                                                                         'key': 'class_label',
                                                                                         'n_classes': 1001},
                                                                              'target': 'model.ldm.modules.encoders.modules.ClassEmbedder'},
                                                        'cond_stage_key': 'class_label',
                                                        'cond_stage_trainable': True,
                                                        'conditioning_key': 'crossattn',
                                                        'first_stage_config': {'params': {'ddconfig': {'attn_resolutions': [],
                                                                                                       'ch': 128,
                                                                                                       'ch_mult': [1,
                                                                                                                   2,
                                                                                                                   4],
                                                                                                       'double_z': False,
                                                                                                       'dropout': 0.0,
                                                                                                       'in_channels': 3,
                                                                                                       'num_res_blocks': 2,
                                                                                                       'out_ch': 3,
                                                                                                       'resolution': 256,
                                                                                                       'z_channels': 3},
                                                                                          'embed_dim': 3,
                                                                                          'lossconfig': {'target': 'torch.nn.Identity'},
                                                                                          'n_embed': 8192},
                                                                               'target': 'model.ldm.models.autoencoder.VQModelInterface'},
                                                        'first_stage_key': 'image',
                                                        'image_size': 64,
                                                        'linear_end': 0.0195,
                                                        'linear_start': 0.0015,
                                                        'log_every_t': 200,
                                                        'monitor': 'val/loss',
                                                        'num_timesteps_cond': 1,
                                                        'timesteps': 1000,
                                                        'unet_config': {'params': {'attention_resolutions': [8, 4, 2],
                                                                                   'channel_mult': [1, 2, 3, 5],
                                                                                   'context_dim': 512,
                                                                                   'image_size': 64,
                                                                                   'in_channels': 3,
                                                                                   'model_channels': 192,
                                                                                   'num_heads': 1,
                                                                                   'num_res_blocks': 2,
                                                                                   'out_channels': 3,
                                                                                   'transformer_depth': 1,
                                                                                   'use_spatial_transformer': True},
                                                                        'target': 'model.ldm.modules.diffusionmodules.openaimodel.UNetModel'},
                                                        'use_ema': False},
                                             'target': 'model.ldm.models.diffusion.ddpm.LatentDiffusion'}},
                    'name': 'ldm'},
 'stable-diffusion-v1.5': {'guidance_scale': 7.5,
                           'inner_resolution': 512,
                           'model_id': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
                           'name': 'sdm',
                           'prompt': 'a natural looking image',
                           'target_resolution': 256},
 'stable-diffusion-v2.1': {'guidance_scale': 7.5,
                           'inner_resolution': 768,
                           'model_id': 'sd2-community/stable-diffusion-2-1',
                           'name': 'sdm',
                           'prompt': 'a natural looking image',
                           'target_resolution': 256}}

SAMPLER_PRESETS: Dict[str, Dict[str, Any]] = {'edm_daps': {'annealing_scheduler_config': {'name': 'edm',
                                             'num_steps': 200,
                                             'sigma_max': 100,
                                             'sigma_min': 0.1,
                                             'timestep': 'poly-7'},
              'diffusion_scheduler_config': {'name': 'edm', 'num_steps': 5, 'sigma_min': 0.01, 'timestep': 'poly-7'},
              'latent': False},
 'latent_edm_daps': {'annealing_scheduler_config': {'name': 'edm',
                                                    'num_steps': 50,
                                                    'sigma_max': 10,
                                                    'sigma_min': 0.1,
                                                    'timestep': 'poly-7'},
                     'diffusion_scheduler_config': {'name': 'edm',
                                                    'num_steps': 2,
                                                    'sigma_min': 0.001,
                                                    'timestep': 'poly-7'},
                     'latent': True},
 'sd_edm_daps': {'annealing_scheduler_config': {'name': 'edm',
                                                'num_steps': 50,
                                                'sigma_max': 10,
                                                'sigma_min': 0.1,
                                                'timestep': 'poly-7'},
                 'diffusion_scheduler_config': {'name': 'edm',
                                                'num_steps': 2,
                                                'sigma_min': 0.001,
                                                'timestep': 'poly-7'},
                 'latent': True}}

TASK_PRESETS: Dict[str, Dict[str, Any]] = {'down_sampling': {'ldm': {'mcmc_sampler_config': {'lr': 0.000135,
                                                   'lr_min_ratio': 0.023,
                                                   'mc_algo': 'hmc',
                                                   'momentum': 0.86,
                                                   'num_steps': 24,
                                                   'prior_solver': 'gaussian',
                                                   'tau': 0.01},
                           'operator': {'name': 'down_sampling', 'resolution': 256, 'scale_factor': 4, 'sigma': 0.05}},
                   'pixel': {'mcmc_sampler_config': {'lr': '1e-4',
                                                     'lr_min_ratio': 0.01,
                                                     'mc_algo': 'langevin',
                                                     'num_steps': 100,
                                                     'prior_solver': 'gaussian',
                                                     'tau': 0.01},
                             'operator': {'name': 'down_sampling',
                                          'resolution': 256,
                                          'scale_factor': 4,
                                          'sigma': 0.05}},
                   'pixel_hmc': {'mcmc_sampler_config': {'lr': 0.000104,
                                                         'lr_min_ratio': 0.012,
                                                         'mc_algo': 'hmc',
                                                         'momentum': 0.9,
                                                         'num_steps': 12,
                                                         'prior_solver': 'gaussian',
                                                         'tau': 0.01},
                                 'operator': {'name': 'down_sampling',
                                              'resolution': 256,
                                              'scale_factor': 4,
                                              'sigma': 0.05}},
                   'sd': {'mcmc_sampler_config': {'lr': '1e-4',
                                                  'lr_min_ratio': 1,
                                                  'mc_algo': 'hmc',
                                                  'momentum': 0.45,
                                                  'num_steps': 30,
                                                  'prior_solver': 'gaussian',
                                                  'tau': 0.01},
                          'operator': {'name': 'down_sampling', 'resolution': 256, 'scale_factor': 4, 'sigma': 0.01}}},
 'gaussian_blur': {'ldm': {'mcmc_sampler_config': {'lr': 2.7e-06,
                                                   'lr_min_ratio': 0.95,
                                                   'mc_algo': 'hmc',
                                                   'momentum': 0.95,
                                                   'num_steps': 35,
                                                   'prior_solver': 'gaussian',
                                                   'tau': 0.01},
                           'operator': {'intensity': 3.0, 'kernel_size': 61, 'name': 'gaussian_blur', 'sigma': 0.05}},
                   'pixel': {'mcmc_sampler_config': {'lr': '1e-4',
                                                     'lr_min_ratio': 0.01,
                                                     'mc_algo': 'langevin',
                                                     'num_steps': 100,
                                                     'prior_solver': 'gaussian',
                                                     'tau': 0.01},
                             'operator': {'intensity': 3.0, 'kernel_size': 61, 'name': 'gaussian_blur', 'sigma': 0.05}},
                   'pixel_hmc': {'mcmc_sampler_config': {'lr': 4.04e-05,
                                                         'lr_min_ratio': 0.011,
                                                         'mc_algo': 'hmc',
                                                         'momentum': 0.1,
                                                         'num_steps': 82,
                                                         'prior_solver': 'gaussian',
                                                         'tau': 0.01},
                                 'operator': {'intensity': 3.0,
                                              'kernel_size': 61,
                                              'name': 'gaussian_blur',
                                              'sigma': 0.05}},
                   'sd': {'mcmc_sampler_config': {'lr': '1e-5',
                                                  'lr_min_ratio': 1,
                                                  'mc_algo': 'hmc',
                                                  'momentum': 0.9,
                                                  'num_steps': 40,
                                                  'prior_solver': 'gaussian',
                                                  'tau': 0.01},
                          'operator': {'intensity': 3.0, 'kernel_size': 61, 'name': 'gaussian_blur', 'sigma': 0.01}}},
 'hdr': {'ldm': {'mcmc_sampler_config': {'lr': 4.52e-06,
                                         'lr_min_ratio': 0.015,
                                         'mc_algo': 'hmc',
                                         'momentum': 0.58,
                                         'num_steps': 38,
                                         'prior_solver': 'gaussian',
                                         'tau': 0.01},
                 'operator': {'name': 'high_dynamic_range', 'scale': 2, 'sigma': 0.05}},
         'pixel': {'mcmc_sampler_config': {'lr': '2e-5',
                                           'lr_min_ratio': 0.01,
                                           'mc_algo': 'langevin',
                                           'num_steps': 100,
                                           'prior_solver': 'gaussian',
                                           'tau': 0.01},
                   'operator': {'name': 'high_dynamic_range', 'scale': 2, 'sigma': 0.05}},
         'pixel_hmc': {'mcmc_sampler_config': {'lr': 2.18e-05,
                                               'lr_min_ratio': 0.013,
                                               'mc_algo': 'hmc',
                                               'momentum': 0.23,
                                               'num_steps': 100,
                                               'prior_solver': 'gaussian',
                                               'tau': 0.01},
                       'operator': {'name': 'high_dynamic_range', 'scale': 2, 'sigma': 0.05}},
         'sd': {'mcmc_sampler_config': {'lr': 1e-05,
                                        'lr_min_ratio': 1,
                                        'mc_algo': 'hmc',
                                        'momentum': 0.7,
                                        'num_steps': 45,
                                        'prior_solver': 'gaussian',
                                        'tau': 0.01},
                'operator': {'name': 'high_dynamic_range', 'scale': 2, 'sigma': 0.01}}},
 'inpainting': {'ldm': {'mcmc_sampler_config': {'lr': 9.02e-06,
                                                'lr_min_ratio': 0.13,
                                                'mc_algo': 'hmc',
                                                'momentum': 0.74,
                                                'num_steps': 15,
                                                'prior_solver': 'gaussian',
                                                'tau': 0.01},
                        'operator': {'mask_len_range': [128, 129],
                                     'mask_type': 'box',
                                     'name': 'inpainting',
                                     'resolution': 256,
                                     'sigma': 0.05}},
                'pixel': {'mcmc_sampler_config': {'lr': '5e-5',
                                                  'lr_min_ratio': 0.01,
                                                  'mc_algo': 'langevin',
                                                  'num_steps': 100,
                                                  'prior_solver': 'gaussian',
                                                  'tau': 0.01},
                          'operator': {'mask_len_range': [128, 129],
                                       'mask_type': 'box',
                                       'name': 'inpainting',
                                       'resolution': 256,
                                       'sigma': 0.05}},
                'pixel_hmc': {'mcmc_sampler_config': {'lr': 1.15e-05,
                                                      'lr_min_ratio': 0.014,
                                                      'mc_algo': 'hmc',
                                                      'momentum': 0.003,
                                                      'num_steps': 84,
                                                      'prior_solver': 'gaussian',
                                                      'tau': 0.01},
                              'operator': {'mask_len_range': [128, 129],
                                           'mask_type': 'box',
                                           'name': 'inpainting',
                                           'resolution': 256,
                                           'sigma': 0.05}},
                'sd': {'mcmc_sampler_config': {'lr': 1.11e-06,
                                               'lr_min_ratio': 0.43,
                                               'mc_algo': 'hmc',
                                               'momentum': 0.92,
                                               'num_steps': 58,
                                               'prior_solver': 'gaussian',
                                               'tau': 0.01},
                       'operator': {'mask_len_range': [128, 129],
                                    'mask_type': 'box',
                                    'name': 'inpainting',
                                    'resolution': 256,
                                    'sigma': 0.01}}},
 'inpainting_rand': {'ldm': {'mcmc_sampler_config': {'lr': 8.4e-06,
                                                     'lr_min_ratio': 0.013,
                                                     'mc_algo': 'hmc',
                                                     'momentum': 0.92,
                                                     'num_steps': 38,
                                                     'prior_solver': 'gaussian',
                                                     'tau': 0.01},
                             'operator': {'mask_prob_range': [0.7, 0.71],
                                          'mask_type': 'random',
                                          'name': 'inpainting',
                                          'resolution': 256,
                                          'sigma': 0.05}},
                     'pixel': {'mcmc_sampler_config': {'lr': '1e-4',
                                                       'lr_min_ratio': 0.01,
                                                       'mc_algo': 'langevin',
                                                       'num_steps': 100,
                                                       'prior_solver': 'gaussian',
                                                       'tau': 0.01},
                               'operator': {'mask_prob_range': [0.7, 0.71],
                                            'mask_type': 'random',
                                            'name': 'inpainting',
                                            'resolution': 256,
                                            'sigma': 0.05}},
                     'pixel_hmc': {'mcmc_sampler_config': {'lr': 3.56e-05,
                                                           'lr_min_ratio': 0.015,
                                                           'mc_algo': 'hmc',
                                                           'momentum': 0.8,
                                                           'num_steps': 11,
                                                           'prior_solver': 'gaussian',
                                                           'tau': 0.01},
                                   'operator': {'mask_prob_range': [0.7, 0.71],
                                                'mask_type': 'random',
                                                'name': 'inpainting',
                                                'resolution': 256,
                                                'sigma': 0.05}},
                     'sd': {'mcmc_sampler_config': {'lr': '3e-5',
                                                    'lr_min_ratio': 1,
                                                    'mc_algo': 'hmc',
                                                    'momentum': 0.6,
                                                    'num_steps': 15,
                                                    'prior_solver': 'gaussian',
                                                    'tau': 0.01},
                            'operator': {'mask_prob_range': [0.7, 0.71],
                                         'mask_type': 'random',
                                         'name': 'inpainting',
                                         'resolution': 256,
                                         'sigma': 0.01}}},
 'motion_blur': {'ldm': {'mcmc_sampler_config': {'lr': 2.4e-05,
                                                 'lr_min_ratio': 0.041,
                                                 'mc_algo': 'hmc',
                                                 'momentum': 0.45,
                                                 'num_steps': 49,
                                                 'prior_solver': 'gaussian',
                                                 'tau': 0.01},
                         'operator': {'intensity': 0.5, 'kernel_size': 61, 'name': 'motion_blur', 'sigma': 0.05}},
                 'pixel': {'mcmc_sampler_config': {'lr': '5e-5',
                                                   'lr_min_ratio': 0.01,
                                                   'mc_algo': 'langevin',
                                                   'num_steps': 100,
                                                   'prior_solver': 'gaussian',
                                                   'tau': 0.01},
                           'operator': {'intensity': 0.5, 'kernel_size': 61, 'name': 'motion_blur', 'sigma': 0.05}},
                 'pixel_hmc': {'mcmc_sampler_config': {'lr': 3.69e-05,
                                                       'lr_min_ratio': 0.01,
                                                       'mc_algo': 'hmc',
                                                       'momentum': 0.11,
                                                       'num_steps': 96,
                                                       'prior_solver': 'gaussian',
                                                       'tau': 0.01},
                               'operator': {'intensity': 0.5, 'kernel_size': 61, 'name': 'motion_blur', 'sigma': 0.05}},
                 'sd': {'mcmc_sampler_config': {'lr': '2e-5',
                                                'lr_min_ratio': 1,
                                                'mc_algo': 'hmc',
                                                'momentum': 0.85,
                                                'num_steps': 30,
                                                'prior_solver': 'gaussian',
                                                'tau': 0.01},
                        'operator': {'intensity': 0.5, 'kernel_size': 61, 'name': 'motion_blur', 'sigma': 0.01}}},
 'nonlinear_blur': {'ldm': {'mcmc_sampler_config': {'lr': 3.71e-06,
                                                    'lr_min_ratio': 0.024,
                                                    'mc_algo': 'hmc',
                                                    'momentum': 0.91,
                                                    'num_steps': 80,
                                                    'prior_solver': 'gaussian',
                                                    'tau': 0.01},
                            'operator': {'name': 'nonlinear_blur',
                                         'opt_yml_path': 'forward_operator/bkse/options/generate_blur/default.yml',
                                         'sigma': 0.05}},
                    'pixel': {'mcmc_sampler_config': {'lr': '1e-5',
                                                      'lr_min_ratio': 0.01,
                                                      'mc_algo': 'langevin',
                                                      'num_steps': 200,
                                                      'prior_solver': 'gaussian',
                                                      'tau': 0.01},
                              'operator': {'name': 'nonlinear_blur',
                                           'opt_yml_path': 'forward_operator/bkse/options/generate_blur/default.yml',
                                           'sigma': 0.05}},
                    'pixel_hmc': {'mcmc_sampler_config': {'lr': 1.2e-05,
                                                          'lr_min_ratio': 0.012,
                                                          'mc_algo': 'hmc',
                                                          'momentum': 0.46,
                                                          'num_steps': 85,
                                                          'prior_solver': 'gaussian',
                                                          'tau': 0.01},
                                  'operator': {'name': 'nonlinear_blur',
                                               'opt_yml_path': 'forward_operator/bkse/options/generate_blur/default.yml',
                                               'sigma': 0.05}},
                    'sd': {'mcmc_sampler_config': {'lr': '1e-5',
                                                   'lr_min_ratio': 1,
                                                   'mc_algo': 'hmc',
                                                   'momentum': 0.8,
                                                   'num_steps': 60,
                                                   'prior_solver': 'gaussian',
                                                   'tau': 0.01},
                           'operator': {'name': 'nonlinear_blur',
                                        'opt_yml_path': 'forward_operator/bkse/options/generate_blur/default.yml',
                                        'sigma': 0.01}}},
 'phase_retrieval': {'ldm': {'mcmc_sampler_config': {'lr': 2.1e-05,
                                                     'lr_min_ratio': 0.12,
                                                     'mc_algo': 'hmc',
                                                     'momentum': 0.41,
                                                     'num_steps': 65,
                                                     'prior_solver': 'gaussian',
                                                     'tau': 0.01},
                             'operator': {'name': 'phase_retrieval', 'oversample': 2.0, 'sigma': 0.05}},
                     'pixel': {'mcmc_sampler_config': {'lr': '5e-5',
                                                       'lr_min_ratio': 0.01,
                                                       'mc_algo': 'langevin',
                                                       'num_steps': 100,
                                                       'prior_solver': 'gaussian',
                                                       'tau': 0.01},
                               'operator': {'name': 'phase_retrieval', 'oversample': 2.0, 'sigma': 0.05}},
                     'pixel_hmc': {'mcmc_sampler_config': {'lr': 9.75e-06,
                                                           'lr_min_ratio': 0.019,
                                                           'mc_algo': 'hmc',
                                                           'momentum': 0.73,
                                                           'num_steps': 62,
                                                           'prior_solver': 'gaussian',
                                                           'tau': 0.01},
                                   'operator': {'name': 'phase_retrieval', 'oversample': 2.0, 'sigma': 0.05}},
                     'sd': {'mcmc_sampler_config': {'lr': 0.00011,
                                                    'lr_min_ratio': 0.027,
                                                    'mc_algo': 'hmc',
                                                    'momentum': 0.47,
                                                    'num_steps': 95,
                                                    'prior_solver': 'gaussian',
                                                    'tau': 0.01},
                            'operator': {'name': 'phase_retrieval', 'oversample': 2.0, 'sigma': 0.01}}}}


DATA_PRESETS = _coerce_numeric(DATA_PRESETS)
MODEL_PRESETS = _coerce_numeric(MODEL_PRESETS)
SAMPLER_PRESETS = _coerce_numeric(SAMPLER_PRESETS)
TASK_PRESETS = _coerce_numeric(TASK_PRESETS)


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

def _first_existing_dir(candidates):
    for c in candidates:
        try:
            if c is not None and Path(c).exists():
                return Path(c)
        except Exception:
            continue
    return Path(candidates[0])

_DEFAULT_DATASETS_ROOT = _first_existing_dir([
    _REPO_ROOT / "inverse_2d_assets" / "datasets",
    _REPO_ROOT / "inverse_2d_assets" / "dataset",
    _REPO_ROOT / "datasets",
    _REPO_ROOT / "dataset",
    _THIS_DIR / "datasets",
    _THIS_DIR / "dataset",
])

_DEFAULT_CHECKPOINTS_ROOT = _first_existing_dir([
    _REPO_ROOT / "inverse_2d_assets" / "checkpoints",
    _REPO_ROOT / "checkpoints",
    _THIS_DIR / "checkpoints",
])

def resolve_dataset_root(root: str) -> str:
    """Resolve a dataset directory path robustly.

    - If `root` is absolute or exists as-given, keep it.
    - If it starts with dataset/ or datasets/, map to the detected datasets root.
    - Otherwise, treat it as relative to the detected datasets root.
    """
    p = Path(root)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    parts = p.parts
    if parts and parts[0] in ("dataset", "datasets"):
        rel = Path(*parts[1:])
    else:
        rel = p
    cand = (_DEFAULT_DATASETS_ROOT / rel)
    return str(cand)

def resolve_checkpoint_path(path: str) -> str:
    """Resolve a checkpoint file path robustly (similar logic to datasets)."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    parts = p.parts
    if parts and parts[0] in ("checkpoint", "checkpoints", "ckpt", "ckpts"):
        rel = Path(*parts[1:])
    else:
        rel = p
    cand = (_DEFAULT_CHECKPOINTS_ROOT / rel)
    return str(cand)

def resolve_repo_path(path: str) -> str:
    """Resolve a path that is expected to live inside this code directory."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    cand = (_THIS_DIR / p)
    if cand.exists():
        return str(cand.resolve())
    cand2 = (_REPO_ROOT / p)
    if cand2.exists():
        return str(cand2.resolve())
    # Fallback: keep it relative to this dir.
    return str(cand)

# Apply path resolution to embedded presets so defaults work regardless of CWD.
for _k, _cfg in DATA_PRESETS.items():
    if isinstance(_cfg, dict) and "root" in _cfg:
        _cfg["root"] = resolve_dataset_root(str(_cfg["root"]))

for _k, _cfg in MODEL_PRESETS.items():
    if not isinstance(_cfg, dict):
        continue
    if _cfg.get("name") == "ddpm":
        mc = _cfg.get("model_config", {})
        if isinstance(mc, dict) and "model_path" in mc:
            mc["model_path"] = resolve_checkpoint_path(str(mc["model_path"]))
    if _cfg.get("name") == "ldm":
        if "diffusion_path" in _cfg:
            _cfg["diffusion_path"] = resolve_checkpoint_path(str(_cfg["diffusion_path"]))

# BKSE nonlinear blur: opt_yml_path should be relative to this code folder.
for _task_name, _groups in TASK_PRESETS.items():
    if not isinstance(_groups, dict):
        continue
    for _group_name, _sub in _groups.items():
        if not isinstance(_sub, dict):
            continue
        op = _sub.get("operator", {})
        if isinstance(op, dict) and "opt_yml_path" in op:
            op["opt_yml_path"] = resolve_repo_path(str(op["opt_yml_path"]))


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++
    # Please change the path to your dataset
    real_dataset = ImageDataset(root='dataset/test-ffhq', resolution=256, start_id=0, end_id=100)
    fake_dataset = ImageDataset(root='results/pixel/ffhq/inpainting/samples', resolution=256, start_id=0, end_id=100)

    real_loader = DataLoader(real_dataset, batch_size=100, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)

    fid_score = calculate_fid(real_loader, fake_loader)
    print(f'FID Score: {fid_score.item():.4f}')

# === End: evaluate_fid.py ===
