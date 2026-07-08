
import os
import gc
from functools import partial
import numpy as np
import torch

from devito import Function

from examples.seismic import Model, Receiver, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

from distributed import Client, LocalCluster
# distributed scheduler serializes the task and sends it to a worker. 
# We implement the task as a stateless function to avoid serialization issues.

from .base import BaseOperator
from devito import configuration
configuration['log-level'] = 'WARNING'

import ctypes
from typing import Dict


def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)

'''
2D Acoustic Wave Modeling and Inversion
---------------------------------------

'''
# Define a type to store the functional and gradient.
class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        
        return fg_pair(f, g)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

def convert2np(rec):
    '''
    Convert the data of Reciever object to numpy array.
    Args:
        - rec: Reciever object
    Returns:
        - data: numpy array
    '''
    return np.array(rec.data)


def forward_single_shot(geometry, model, save=False, dt=1.0):
    '''
    Args:
        - shot_idx: index of the shot
        - save: whether to save the wavefield
    '''
    solver_i = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs = solver_i.forward(vp=model.vp, save=save, dt=dt)[0]
    return d_obs.resample(dt)


def forward_multi_shots(model, geometry_list, client, save=False, dt=1.0, return_rec=True):
    '''
    Args:
        - velocity: velocity model, (H, W), numpy array.
        - client: dask client
        - save: whether to save the wavefield
        - dt: time step
        - return_rec: If True, return the Reciever object, else return the data of Reciever objects as list of np.ndarray.
    Returns:
        - shots: list of Reciever objects, which contains recorded data. 
    '''
    forward_single_shot_fn = partial(forward_single_shot, model=model, save=save, dt=dt)
    futures = client.map(forward_single_shot_fn, geometry_list)
    
    if return_rec:
        shots = client.gather(futures)
        return shots
    else:
        shots_tmp = client.map(convert2np, futures)
        shots_np = client.gather(shots_tmp)
        return shots_np


def gradient_single_shot(geometry, d_obs, model, fs=True):
    '''
    Compute the functional value and gradient of the functional w.r.t. the squared slowness for a single shot.
    Args:
        - model: Devito model object
        - geometry: AcquisitionGeometry object
        - d_obs: Reciever object
        - fs: whether to use free surface boundary condition
    '''
    # Devito objects for gradient and data residual
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=4)

    # Predicted data and residual
    d_pred, u0 = solver.forward(vp=model.vp, save=True)[0:2]
    residual.data[:] = d_pred.data[:] - d_obs.resample(geometry.dt).data[:][0:d_pred.data.shape[0], :]

    # Function value and gradient    
    fval = .5*np.linalg.norm(residual.data.flatten())**2
    solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad)
    
    # Convert to numpy array and remove absorbing boundaries
    nbl = model.nbl
    z_start = 0 if fs else nbl
    grad_crop = np.array(grad.data[:])[nbl:-nbl, z_start:-nbl]
    return fg_pair(fval, grad_crop)


def gradient_multi_shots(model, geometry_list, ob_recs, client, fs=True):
    '''
    Compute the functional value and gradient of the functional w.r.t. the squared slowness for all shots.
    Args:
        - model: Devito model object
        - geometry_list: list of AcquisitionGeometry objects
        - ob_recs: list of Reciever objects
        - client: dask client
        - fs: whether to use free surface boundary condition
    Returns:
        - fg.f: functional value
        - fg.g: gradient of the functional, np.ndarray
    '''
    gradient_single_shot_fn = partial(gradient_single_shot, model=model, fs=fs)
    fgi = client.map(gradient_single_shot_fn, geometry_list, ob_recs)
    fg = client.submit(sum, fgi).result()
    return fg.f, fg.g


class AcousticWave(BaseOperator):
    def __init__(self, 
                 shape,                 # Number of grid points [nx, nz] where nx is the number of grid points in x-direction and nz is the number of grid points in depth
                 spacing,               # Grid spacing in m. [dx, dz]
                 tn,                    # Final time in ms
                 f0,                    # Dominant frequency of Ricker source
                 dt,                    # Time step in ms, must be smaller than model.critical_dt for stability
                 nbl,                   # Number of obsorbing boundary layers
                 nreceivers,            # Number of receivers
                 nshots,                # Number of sources
                 src_depth=10.0,        # Depth of the sources in m
                 rec_depth=10.0,        # Depth of the receivers in m
                 fs=True,               # Free surface boundary condition
                 bcs='damp',            # Boundary condition
                 space_order=4,         # Finite difference order in space
                gc_threshold=1000,      # Threshold for garbage collection
                 **kwargs):
        super().__init__(**kwargs)
        vel_init = np.ones(shape, dtype=np.float32)
    
        self.model = Model(vp=vel_init, origin=(0, 0), 
                           shape=shape, spacing=spacing, space_order=space_order, 
                           nbl=nbl, fs=fs, bcs=bcs, dt=dt)
        self.dt = dt
        self.fs = fs                # If True, use free surface boundary condition, else use obsorbing boundary condition
        self.nshots = nshots
        self.nreceivers = nreceivers
        # Set up acquisition geometry
        src_coordinates = np.empty((nshots, 2))
        src_coordinates[:, 0] = np.linspace(spacing[0], self.model.domain_size[0], num=nshots)
        src_coordinates[:, 1] = src_depth    # Source depth

        rec_coordinates = np.empty((nreceivers, 2))
        rec_coordinates[:, 0] = np.linspace(0, self.model.domain_size[0], num=nreceivers)
        rec_coordinates[:, 1] = rec_depth    # Receiver depth

        self.geometry_list = []
        self.solver_list = []
        for i in range(nshots):
            geometry_i = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates[i, :], 0.0, tn, f0=f0, src_type='Ricker')        
            solver_i = AcousticWaveSolver(self.model, geometry_i, space_order=space_order)
            self.geometry_list.append(geometry_i)
            self.solver_list.append(solver_i)
        self.num_time_steps = geometry_i.time_axis.num
        print(f"Will record {self.num_time_steps} time steps.")
        # Set up dask client
        cluster = LocalCluster(threads_per_worker=nshots, death_timeout=120)
        self.client = Client(cluster)
        self.client.run(gc.disable)
        self.num_calls = 0      # Number of forward calls, if it is greater than 1000, restart the client.
        self.gc_threshold = gc_threshold
    
    def __call__(self, data: Dict[str, torch.Tensor], unnormalize=True):
        '''
        Args:
            inputs: single velocity model, (1, 1, H, W), torch.tensor.
        Returns:
            shots: list of Reciever objects, which contains recorded data.
        '''
        inputs = data['target']
        if unnormalize:
            inputs = self.unnormalize(inputs)
        vel_np = inputs.detach().transpose(-2,-1).cpu().numpy()[0, 0]
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        self.model.vp.data[nbl:-nbl, z_start:-nbl] = vel_np

        shots = forward_multi_shots(self.model, self.geometry_list, self.client, dt=self.dt, return_rec=True)
        return shots

    def forward(self, inputs, unnormalize=True):
        '''
        Args:
            - inputs: velocity model, (batch_size, 1, H, W), torch.tensor.
        Returns:
            - out: recorded data, (batch_size, nshots, T, num_receivers), torch.tensor.
        '''
        self.check_gc()
        if unnormalize:
            inputs = self.unnormalize(inputs)
        batch_vel_np = inputs.detach().transpose(-2, -1).cpu().numpy()
        out_np = np.empty((batch_vel_np.shape[0], self.nshots, self.num_time_steps, self.nreceivers), dtype=np.float32)
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        for i in range(batch_vel_np.shape[0]):
            self.model.vp.data[nbl:-nbl, z_start:-nbl] = batch_vel_np[i, 0]
            shots = forward_multi_shots(self.model, self.geometry_list, self.client, dt=self.dt, return_rec=False)
            shots_np = np.stack(shots, axis=0) # (nshots, T, nreceivers)
            out_np[i] = shots_np
        del shots_np
        out = torch.from_numpy(out_np).to(inputs.device) # (batch_size, nshots, T, nreceivers)
        # check NaN values
        if torch.isnan(out).any():
            raise ValueError("NaN values in the forward evaluation.")
        return out
    
    def loss(self, pred, observation, unnormalize=True):
        '''
        Compute the loss functional 1/2 ||d_obs - d_pred||^2.
        Args:
            - pred: predicted velocity model, (batchsize, 1, Z, X), torch.tensor.
            - observation: list of Reciever objects, which contains recorded data.
        Returns:
            - loss: loss functional, torch.tensor. (batch_size)
        '''
        self.check_gc()
        pred_out = self.forward(pred, unnormalize=unnormalize) # (batch_size, nshots, T, nreceivers)
        obs_out = torch.from_numpy(np.stack([convert2np(obs) for obs in observation], axis=0)).to(pred.device) # (batch_size, nshots, T, nreceivers)
        residual = pred_out - obs_out.unsqueeze(0)
        loss = 0.5 * torch.linalg.norm(residual.flatten(start_dim=1), dim=1)**2
        return loss

    def gradient(self, pred, observation, return_loss=False, unnormalize=True):
        '''
        Compute the gradient of the functional w.r.t. the velocity model where the loss functional is 1/2 ||d_obs - d_pred||^2.
        Args:
            - pred: predicted velocity model, (1, 1, Z, X), torch.tensor.
            - observation: list of Reciever objects, which contains recorded data.
            - return_loss: whether to return loss scale, bool.
        returns:
            - vel_grad: gradient of the functional w.r.t. velocity, (1, 1, Z, X), torch.tensor.
        '''
        self.check_gc()
        if unnormalize:
            pred = self.unnormalize(pred)
        pred_np = pred.detach().transpose(-2, -1).detach().cpu().numpy()
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        self.model.vp.data[nbl:-nbl, z_start:-nbl] = pred_np[0, 0]
        fval, grad_slowness = gradient_multi_shots(self.model, self.geometry_list, observation, self.client, fs=self.fs)
        # check NaN values
        if np.isnan(grad_slowness).any():
            raise ValueError("NaN values in the gradient.")
        if np.isnan(fval):
            raise ValueError("NaN values in the functional value.")
        grad_vel = - 2.0 * grad_slowness / pred_np[0, 0] ** 3       # (X, Z)
        grad_vel = torch.from_numpy(grad_vel).transpose(0, 1).unsqueeze(0).unsqueeze(0).to(pred.device)
        if unnormalize:
            grad_vel = grad_vel * self.unnorm_scale
        if return_loss:
            return grad_vel, torch.tensor(fval)
        else:
            return grad_vel

    def check_gc(self):
        if self.num_calls > self.gc_threshold:
            self.client.run(gc.collect)
            self.client.run(trim_memory)
            self.num_calls = 0
        else:
            self.num_calls += 1
    
    def close(self):
        self.client.close()