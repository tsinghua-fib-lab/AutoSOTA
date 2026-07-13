"""The CNN variants module houses convolutional encoders / decoders to be used with image data."""


## This needs a lot of cleanup, performance-wise seems ok but too much code dump in cnn_arch in models

import torch, pyro
from pyro import poutine
import pyro.distributions as dist
import torch.nn as nn
from MLP_variants import _MLPMixin, MLP, GaussianMLP
import numpy as np
import torch.nn.functional as F
from VAE_variants import (
    VAE, 
    CVAE, 
    CSVAENA, 
    CSVAE, 
    HCSVAENA, 
    HCSVAE, 
    SDIVA,
    CCVAE,
    DLVAE,
)

from typing import Literal


class Unflatten(nn.Module):
    def __init__(self, target_shape):
        nn.Module.__init__(self)
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape([x.shape[0]] + self.target_shape)


class _CNNMixin(_MLPMixin):
    @staticmethod
    def _make_conv(self, in_channels, out_channels, kernel_size):
        
        if self.ConvClass == nn.ConvTranspose2d:
            in_channels, out_channels = out_channels, in_channels

        return nn.Sequential(
                    self.ConvClass(in_channels, out_channels, kernel_size, stride=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )


    @staticmethod
    def _make_conv_updown(self, pool, in_channels, out_channels, kernel_size, padding=0, pool_output_size=None):
        
        if self.PoolClass == nn.Upsample:
            in_channels, out_channels = out_channels, in_channels

            if pool:
                return nn.Sequential(
                    self.PoolClass(pool_output_size),
                    self.ConvClass(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )

            return nn.Sequential(
                    self.ConvClass(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )

        
        if pool:
             return nn.Sequential(
                    self.ConvClass(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    self.PoolClass(2)
                )
            
                
                
        return nn.Sequential(
                    self.ConvClass(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
        
        

class _CNNEncoderMixin(_CNNMixin):
    def __init__(self):
        self.ConvClass = nn.Conv2d
        self.PoolClass = nn.MaxPool2d

class _CNNDecoderMixin(_CNNMixin): 
    def __init__(self):
        self.ConvClass = nn.ConvTranspose2d

class _CNNDecoderUpsampleMixin(_CNNMixin): 
    def __init__(self):
        self.ConvClass = nn.Conv2d
        self.PoolClass = nn.Upsample

class _CNNLayerMixin:
    
    def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2, kernel_size=3, extra_dim=0):
        self.last_layer=sum(repeats)

        i = 0

        for channel_idx, repeat in zip(range(len(channels)), repeats):

            for j in range(repeat):

                match (i,j):
                    case (0,j): # Input layer
                        setattr(self, f'layer_{i}', self._make_conv(self, in_channels, channels[channel_idx], kernel_size))

                    case (i,j) if j == 0: # Initial layer for midsections
                        setattr(self, f'layer_{i}', self._make_conv(self, channels[channel_idx-1], channels[channel_idx], kernel_size))
                    
                    case _: # Intermediate layers
                        setattr(self, f'layer_{i}', self._make_conv(self, channels[channel_idx], channels[channel_idx], kernel_size))


                i += 1

        setattr(self, f'layer_{i}', MLP((np.array(in_dims) - (sum(repeats) * (kernel_size-1))).prod() * channels[-1] + extra_dim, [hidden_dim]*num_hidden, out_dim))


    def forward(self, x, start=0, end=14):
        for i in range(start, end):
            x = getattr(self, f'layer_{i}')(x)

        return x



class _CNNSampledLayerMixin(_CNNLayerMixin):
    def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2, kernel_size=3, extra_dim=0):

        assert repeats[0] > 1 # This needs to be fixed down the line but is very minor
        
        self.output_sizes=[np.array(in_dims) // 2**i for i in range(len(repeats))]
        self.last_layer=sum(repeats)
        self.pad_size = kernel_size // 2

        i = 0

        for channel_idx, repeat in zip(range(len(channels)), repeats):

            for j in range(repeat):

                match (i,j):
                    case (0,j): # Input layer
                        if repeat == 1: # Input is also last layer!
                            setattr(self, f'layer_{i}', self._make_conv_updown(self, True, in_channels, channels[channel_idx], kernel_size, self.pad_size, pool_output_size=self.output_sizes[channel_idx].tolist()))

                        else:
                            setattr(self, f'layer_{i}', self._make_conv_updown(self, False, in_channels, channels[channel_idx], kernel_size, self.pad_size))

                    case (i,j) if j == 0: # Initial layer for midsections
                        if repeat == 1: # Initial is also last layer!
                            setattr(self, f'layer_{i}', self._make_conv_updown(self, True, channels[channel_idx-1], channels[channel_idx], kernel_size, self.pad_size, pool_output_size=self.output_sizes[channel_idx].tolist()))

                        else:
                            setattr(self, f'layer_{i}', self._make_conv_updown(self, False, channels[channel_idx-1], channels[channel_idx], kernel_size, self.pad_size))

                    case (i,j) if j == repeat-1: # Last layer for section
                        setattr(self, f'layer_{i}', self._make_conv_updown(self, True, channels[channel_idx], channels[channel_idx], kernel_size, self.pad_size, pool_output_size=self.output_sizes[channel_idx].tolist()))
                    
                    case _: # Intermediate layers
                        setattr(self, f'layer_{i}', self._make_conv_updown(self, False, channels[channel_idx], channels[channel_idx], kernel_size, self.pad_size))


                i += 1

        setattr(self, f'layer_{i}', MLP((np.array(in_dims) // 2**len(repeats)).prod() * channels[-1] + extra_dim, [hidden_dim]*num_hidden, out_dim))




class CNNEncoder(_CNNLayerMixin, _CNNEncoderMixin, nn.Module):
    def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2,  kernel_size=3, extra_dim=0):
        nn.Module.__init__(self)
        _CNNEncoderMixin.__init__(self)
        _CNNLayerMixin.__init__(self, in_dims, in_channels, out_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, kernel_size=kernel_size, extra_dim=extra_dim)

    def forward(self, x, y=None):

        x = _CNNLayerMixin.forward(self, x, end=self.last_layer)
        x = x.reshape(x.size(0), -1)

        if y is not None:
            x = torch.concatenate((x,y), dim=-1)
        
        x = getattr(self, f'layer_{self.last_layer}')(x)
        return x


class CNNSampledEncoder(_CNNSampledLayerMixin, CNNEncoder):
    def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2, kernel_size=3, extra_dim=0):
        nn.Module.__init__(self)
        _CNNEncoderMixin.__init__(self)
        _CNNSampledLayerMixin.__init__(self, in_dims, in_channels, out_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, extra_dim=extra_dim)


class GaussianCNNEncoder(CNNEncoder):
     def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2,  kernel_size=3, extra_dim=0):
         CNNEncoder.__init__(self, in_dims, in_channels, out_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, kernel_size=kernel_size, extra_dim=extra_dim)
         setattr(self, f'layer_{self.last_layer}', GaussianMLP((np.array(in_dims) - (sum(repeats) * (kernel_size-1))).prod() * channels[-1] + extra_dim, [hidden_dim]*num_hidden, out_dim))


class GaussianCNNSampledEncoder(CNNSampledEncoder):
      def __init__(self, in_dims, in_channels, out_dim, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2, kernel_size=3, extra_dim=0):
         CNNSampledEncoder.__init__(self, in_dims, in_channels, out_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, extra_dim=extra_dim)
         setattr(self, f'layer_{self.last_layer}', GaussianMLP((np.array(in_dims) // 2**len(repeats)).prod() * channels[-1] + extra_dim, [hidden_dim]*num_hidden, out_dim))
         

class CNNDecoder(_CNNLayerMixin, _CNNDecoderMixin, nn.Module):
    def __init__(self, in_dim, out_channels, out_dims, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2,  kernel_size=3, extra_dim=0):
        nn.Module.__init__(self)
        
        _CNNDecoderMixin.__init__(self)
        _CNNLayerMixin.__init__(self, out_dims, out_channels, in_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, kernel_size=kernel_size, extra_dim=extra_dim)

        new_layers = [getattr(self, f'layer_{i}') for i in range(self.last_layer,-1,-1)]

        for i in range(1,sum(repeats)+1):
            setattr(self, f'layer_{i}', new_layers[i])

        setattr(self, f'layer_0', MLP(in_dim + extra_dim, [hidden_dim]*num_hidden, (np.array(out_dims) - (sum(repeats) * (kernel_size-1))).prod() * channels[-1]))
        self.layer_0.layers = nn.Sequential(*(list(self.layer_0.layers.children()) + [Unflatten([channels[-1], out_dims[0] - (sum(repeats) * (kernel_size-1)), out_dims[1] - (sum(repeats) * (kernel_size-1))])]))

        setattr(self, f'layer_{self.last_layer}', nn.Sequential(*([list(getattr(self, f'layer_{self.last_layer}').children())[0]] + [nn.Sigmoid()])))

    def forward(self, x , y=None):
        if y is not None:
            x = torch.concatenate((x,y), dim=-1)
        
        x = self.layer_0(x)
        x = _CNNLayerMixin.forward(self, x, start=1, end=self.last_layer+1)

        return x

class CNNSampledDecoder(_CNNSampledLayerMixin, _CNNDecoderUpsampleMixin, CNNDecoder):
    def __init__(self, in_dim, out_channels, out_dims, channels=[64,128,256,512,512], repeats=[2,2,3,3,3], hidden_dim=4096, num_hidden=2, kernel_size=3, extra_dim=0):
        nn.Module.__init__(self)
        
        _CNNDecoderUpsampleMixin.__init__(self)
        _CNNSampledLayerMixin.__init__(self, out_dims, out_channels, in_dim, channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_hidden, extra_dim=extra_dim)

        new_layers = [getattr(self, f'layer_{i}') for i in range(self.last_layer,-1,-1)]

        for i in range(1,sum(repeats)+1):
            setattr(self, f'layer_{i}', new_layers[i])

        setattr(self, f'layer_0', MLP(in_dim + extra_dim, [hidden_dim]*num_hidden, (np.array(out_dims) // 2**len(repeats)).prod() * channels[-1]))
        self.layer_0.layers = nn.Sequential(*(list(self.layer_0.layers.children()) + [Unflatten([channels[-1], out_dims[0] // 2**len(repeats), out_dims[1] // 2**len(repeats)])]))

        setattr(self, f'layer_{self.last_layer}', nn.Sequential(*([list(getattr(self, f'layer_{self.last_layer}').children())[0]] + [nn.Sigmoid()])))

    


class CNNVAE(VAE):
    def __init__(
        self,
        in_dims,
        in_channels,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):
        nn.Module.__init__(self)
        VAE.__init__(
            self, in_dims[0], hidden_dim, num_layers, latent_dim, recon_weight, kl_weight
        )
        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)

            case 'conv+pool':
                self.encoder = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNSampledDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)


class CNNCVAE(CVAE):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dim: int,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):
        nn.Module.__init__(self)
        CVAE.__init__(
            self, in_dims[0], label_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=label_dim)
                self.decoder = CNNDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=label_dim)

            case 'conv+pool':
                self.encoder = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=label_dim)
                self.decoder = CNNSampledDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=label_dim)


    @staticmethod
    def _get_input_args(*args):
        return args[0], args[1]

    @staticmethod
    def _get_latent_args(z, *args):
        return z, args[1]


    def _reconstruct(self, z, *args):
        x_rec = self.decoder(*self._get_latent_args(z, *args))
        pyro.deterministic('rec', x_rec)

        with poutine.scale(None, self.recon_weight):
            pyro.factor(
                "reconstruction_loss",
                -1 * torch.nn.functional.mse_loss(x_rec, self._get_output_args(*args)).mean()
            )

    def model(self, *args):
        x, y = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
        
            z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
                x.device
            ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)
    
            with poutine.scale(None, self.kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
    
            self._reconstruct(z, *args)

    def guide(self, *args):
        x, y = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = self.encoder(x,y)
    
            with poutine.scale(None, self.kl_weight):
                pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
    



class CNNCSVAENA(CSVAENA):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',

    ):
        nn.Module.__init__(self)
        CSVAENA.__init__(
            self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
        
            case 'conv':
                self.encoder = self.encoder_z = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)


            case 'conv+pool':
                self.encoder = self.encoder_z = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNSampledDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                

        
    def guide(self, *args):
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):

            x, y = self._get_input_args(*args), self._get_label_args(*args)

            z_loc, z_scale = self.encoder_z(x)
            w_loc, w_scale = self.encoder_w(x, y)
    
            with poutine.scale(None, self.z_kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
    
            with poutine.scale(None, self.w_kl_weight):
                pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
    
            return z



class CNNCSVAE(CSVAE):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',

    ):
        nn.Module.__init__(self)
        CSVAE.__init__(
            self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight, adversarial_weight,
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
        
            case 'conv':
                self.encoder = self.encoder_z = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)

            
            case 'conv+pool':
                self.encoder = self.encoder_z = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNSampledDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                

        
    def guide(self, *args):
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):

            x, y = self._get_input_args(*args), self._get_label_args(*args)

            z_loc, z_scale = self.encoder_z(x)
            w_loc, w_scale = self.encoder_w(x, y)
    
            with poutine.scale(None, self.z_kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
    
            with poutine.scale(None, self.w_kl_weight):
                pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))


            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    self._entropy_from_encodings(z),
                    has_rsample=True,
                )
    


class CNNHCSVAENA(HCSVAENA):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):
        nn.Module.__init__(self)
        HCSVAENA.__init__(
            self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight,
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder_rho = GaussianCNNEncoder(self.in_dims, self.in_channels, self.rho_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNDecoder(self.rho_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)

            case 'conv+pool':
                self.encoder_rho = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.rho_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNSampledDecoder(self.rho_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)




class CNNHCSVAE(HCSVAE):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',

    ):
        nn.Module.__init__(self)
        HCSVAE.__init__(
            self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight, adversarial_weight
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder_rho = GaussianCNNEncoder(self.in_dims, self.in_channels, self.rho_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNDecoder(self.rho_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)    


            case 'conv+pool':
                self.encoder_rho = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.rho_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                self.decoder = CNNSampledDecoder(self.rho_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)



class CNNSDIVA(SDIVA):

    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        classifier_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):
        SDIVA.__init__(self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight, kl_weight, classifier_weight)
        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder = self.encoder_z = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.decoder = CNNDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)


            case 'conv+pool':
                self.encoder = self.encoder_z = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.decoder = CNNSampledDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)

     


class CNNCCVAE(CCVAE):
    
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        classifier_weight: float = 1.0,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):

        CCVAE.__init__(self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight , kl_weight , classifier_weight)
        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )


        match cnn_arch:
            case 'conv':
                self.encoder = self.encoder_z = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.decoder = CNNDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)


            case 'conv+pool':
                self.encoder = self.encoder_z = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, sum(self.label_dims) * self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.decoder = CNNSampledDecoder(self.latent_dim + sum(self.label_dims) * self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
    
    
        

class CNNDLVAE(DLVAE):
    def __init__(
        self,
        in_dims,
        in_channels: int,
        label_dims,
        channels = [64,128,256,512,512],
        repeats = [64,128,256,512,512],
        hidden_dim: int = 4096,
        num_layers: int = 2,
        kernel_size: int = 3,
        latent_dim: int = 10,
        w_dim: int = 2,
        recon_weight: float = 1.0,
        recon_weight_z: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        classifier_layers: int = 1,
        learnable_prior: bool = True,
        cnn_arch: Literal['conv', 'conv+pool'] = 'conv',
    ):
        nn.Module.__init__(self)
        DLVAE.__init__(
            self, in_dims[0], label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight, recon_weight_z, z_kl_weight, w_kl_weight, adversarial_weight, classifier_layers, learnable_prior,
        )

        self.in_dims, self.in_channels = (
            in_dims,
            in_channels,
        )

        match cnn_arch:
            case 'conv':
                self.encoder = GaussianCNNEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNEncoder(self.in_dims, self.in_channels, self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNDecoder(self.latent_dim + self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
               
                self.decoder_z = CNNDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
        
        
                classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
                for i in range(len(classifier_dims)):
                    setattr(
                        self,
                        f"classifiers_{i}",
                        CNNEncoder(self.in_dims, 3, classifier_dims[i], channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                    )


            case 'conv+pool':
                self.encoder = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.latent_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                
                self.encoder_w = GaussianCNNSampledEncoder(self.in_dims, self.in_channels, self.w_dim,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size, extra_dim=sum(self.label_dims))
                
                self.decoder = CNNSampledDecoder(self.latent_dim + self.w_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
               
                self.decoder_z = CNNSampledDecoder(self.latent_dim, self.in_channels, self.in_dims,  channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
        
        
                classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
                for i in range(len(classifier_dims)):
                    setattr(
                        self,
                        f"classifiers_{i}",
                        CNNSampledEncoder(self.in_dims, 3, classifier_dims[i], channels=channels, repeats=repeats, hidden_dim=hidden_dim, num_hidden=num_layers, kernel_size=kernel_size)
                    )
        


    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):

            z_loc, z_scale = self.encoder(x)
            
            with poutine.scale(None, self.z_kl_weight):
                z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
    
            w_loc, w_scale = self.encoder_w(x, y)
    
            with poutine.scale(None, self.w_kl_weight):
                w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            
            return z

    
    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder(x)
        
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        x_rec_z = self._reconstruct_z(z, *args)

        return -1 * self._adversarial_from_encodings(x_rec_z, self._get_label_args(*args))
    

    def _reconstruct(self, w, *args):
        x_rec_w = self.decoder(w)

        pyro.deterministic('rec_w', x_rec_w)

        with poutine.scale(None, self.recon_weight):
            pyro.factor(
                "reconstruction_loss_w",
                -1 * torch.nn.functional.mse_loss(x_rec_w, self._get_output_args(*args)).mean()
            ) 

    def _reconstruct_z(self, z, *args):
        x_rec_z = self.decoder_z(z)

        pyro.deterministic('rec_z', x_rec_z)

        with poutine.scale(None, self.recon_weight_z):
            pyro.factor(
                "reconstruction_loss_z",
                -1 * torch.nn.functional.mse_loss(x_rec_z, self._get_output_args(*args)).mean()
            ) 
             
        return x_rec_z