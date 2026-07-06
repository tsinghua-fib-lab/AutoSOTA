import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, jacfwd, jacrev
import numpy as np
from einops import rearrange
from findiff import FinDiff

def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)])
    return rearrange(tensor, pattern)

def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

class StencilGradientComputation(nn.Module):
    '''
    Warning: This is hard-coded for finite differences on images with 2nd order accuracy.
    '''
    def __init__(self, stencils, periodic=False, device = 'cpu'):
        super(StencilGradientComputation, self).__init__()

        # identify max kernel size
        self.max_inner_offset = 0
        self.max_offset = 0
        for key, stencil in stencils.items():
            for (i, j), value in stencil.items():
                if key == ('C', 'C'):
                    self.max_inner_offset = max(self.max_inner_offset, abs(i), abs(j))
                else:
                    self.max_offset = max(self.max_offset, abs(i), abs(j))
        self.max_inner_kernel_size = 2*self.max_inner_offset + 1 # include center and in both directions
        self.max_kernel_size = 2*self.max_offset + 1 # include center and in both directions

        self.kernels = {}
        mid_inner = self.max_inner_offset # center of the kernel
        mid = self.max_offset # center of the kernel
        for key, stencil in stencils.items():
            if key == ('C', 'C'):
                kernel = torch.zeros((1, 1, self.max_inner_kernel_size, self.max_inner_kernel_size), device=device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid_inner+i, mid_inner+j] = value
            else:
                kernel = torch.zeros((1, 1, self.max_kernel_size, self.max_kernel_size), device = device)
                self.kernels[key] = kernel
                for (i, j), value in stencil.items():
                    kernel[0, 0, mid+i, mid+j] = value
            self.kernels[key] = kernel

            self.periodic = periodic
            
    def forward(self, x):

        original_size = x.size()
        batch_size, *channels, height, width = original_size

        # flatten the channel dimensions
        x = x.view(batch_size, -1, height, width)
        channels = x.size(1)

        interior_kernel = self.kernels[('C', 'C')]
        interior_kernel = interior_kernel.repeat((channels, 1, 1, 1))
        
        if self.periodic:
            # pad the image with the opposite boundary
            padding = (self.max_inner_offset, self.max_inner_offset, self.max_inner_offset, self.max_inner_offset)
            x = F.pad(x, padding, mode='circular')
            x_grads = F.conv2d(x, interior_kernel, groups=channels)
            return x_grads.view(original_size)

        interior_conv = F.conv2d(x, interior_kernel, groups=channels)

        # manually apply boundary stencils
        # we extend the image by max_offset since kernel is centered
        x_ext = F.pad(x, (self.max_offset, self.max_offset, self.max_offset, self.max_offset), mode='constant', value=0)

        # only consider the part of x that is at the boundary for the convolution (while being consistent with the convolution kernels)
        reduced_conv_offset = 2*self.max_offset+self.max_inner_offset

        # top boundary
        top_kernel = self.kernels[('L', 'C')]
        top_kernel = top_kernel.repeat((channels, 1, 1, 1))
        top_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, :], top_kernel, groups=channels)

        # bottom boundary
        bottom_kernel = self.kernels[('H', 'C')]
        bottom_kernel = bottom_kernel.repeat((channels, 1, 1, 1))
        bottom_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, :], bottom_kernel, groups=channels)

        # left boundary
        left_kernel = self.kernels[('C', 'L')]
        left_kernel = left_kernel.repeat((channels, 1, 1, 1))
        left_conv = F.conv2d(x_ext[:, :, :, 0:reduced_conv_offset], left_kernel, groups=channels)

        # right boundary
        right_kernel = self.kernels[('C', 'H')]
        right_kernel = right_kernel.repeat((channels, 1, 1, 1))
        right_conv = F.conv2d(x_ext[:, :, :, -reduced_conv_offset:], right_kernel, groups=channels)

        # top-left corner
        tl_corner_kernel = self.kernels[('L', 'L')]
        tl_corner_kernel = tl_corner_kernel.repeat((channels, 1, 1, 1))
        tl_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, 0:reduced_conv_offset], tl_corner_kernel, groups=channels)

        # top-right corner
        tr_corner_kernel = self.kernels[('L', 'H')]
        tr_corner_kernel = tr_corner_kernel.repeat((channels, 1, 1, 1))
        tr_corner_conv = F.conv2d(x_ext[:, :, 0:reduced_conv_offset, -reduced_conv_offset:], tr_corner_kernel, groups=channels)

        # bottom-left corner
        bl_corner_kernel = self.kernels[('H', 'L')]
        bl_corner_kernel = bl_corner_kernel.repeat((channels, 1, 1, 1))
        bl_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, 0:reduced_conv_offset], bl_corner_kernel, groups=channels)

        # bottom-right corner
        br_corner_kernel = self.kernels[('H', 'H')]
        br_corner_kernel = br_corner_kernel.repeat((channels, 1, 1, 1))
        br_corner_conv = F.conv2d(x_ext[:, :, -reduced_conv_offset:, -reduced_conv_offset:], br_corner_kernel, groups=channels)

        # combine the results from interior, boundaries, and corners
        x_grads = torch.zeros_like(x)
        x_grads[:, :, self.max_inner_offset:-self.max_inner_offset, self.max_inner_offset:-self.max_inner_offset] = interior_conv
        x_grads[:, :, 0:self.max_inner_offset, :] = top_conv
        x_grads[:, :, -self.max_inner_offset:, :] = bottom_conv
        x_grads[:, :, :, 0:self.max_inner_offset] = left_conv
        x_grads[:, :, :, -self.max_inner_offset:] = right_conv
        x_grads[:, :, 0:self.max_inner_offset, 0:self.max_inner_offset] = tl_corner_conv
        x_grads[:, :, 0:self.max_inner_offset, -self.max_inner_offset:] = tr_corner_conv
        x_grads[:, :, -self.max_inner_offset:, 0:self.max_inner_offset] = bl_corner_conv
        x_grads[:, :, -self.max_inner_offset:, -self.max_inner_offset:] = br_corner_conv

        # reshape back to the original dimensions
        x_grads = x_grads.view(original_size)        
        return x_grads
    
class StencilGradients(nn.Module):
    '''
    This is hard-coded for finite differences on images with n-th order accuracy (for first and second derivatives).
    '''
    def __init__(self, d0=1, d1=1, fd_acc = 2, periodic=False, device = 'cpu'):
        super(StencilGradients, self).__init__()
        self.d_d0 = StencilGradientComputation(FinDiff(0, d0, 1, acc=fd_acc).stencil((99,99)).data, periodic, device)
        self.d_d1 = StencilGradientComputation(FinDiff(1, d1, 1, acc=fd_acc).stencil((99,99)).data, periodic, device)
        self.d_d00 = StencilGradientComputation(FinDiff(0, d0, 2, acc=fd_acc).stencil((99,99)).data, periodic, device)
        self.d_d11 = StencilGradientComputation(FinDiff(1, d1, 2, acc=fd_acc).stencil((99,99)).data, periodic, device)
        self.d_d01 = StencilGradientComputation(FinDiff((0, d0, 1), (1, d1, 1), acc=fd_acc).stencil((99, 99)).data, periodic, device)

    def forward(self, x, mode):
        if mode == 'all':
            return self.d_d0(x), self.d_d1(x), self.d_d00(x), self.d_d11(x), self.d_d01(x)
        elif mode == 'd_d0':
            return self.d_d0(x)
        elif mode == 'd_d1':
            return self.d_d1(x)
        elif mode == 'd_d00':
            return self.d_d00(x)
        elif mode == 'd_d11':
            return self.d_d11(x)
        elif mode == 'd_d01':
            return self.d_d01(x)
        else:
            raise NotImplementedError

class GradientsHelper:
    def __init__(self, d0, d1, fd_acc, periodic = False, device = 'cpu', eps=1e-6):
        """
        Class for gradient computations.
        """
        self.eps = eps

        self.stencil_gradients = StencilGradients(d0=d0, d1=d1, fd_acc = fd_acc, periodic=periodic, device = device)
        
    def compute_jacobian_num(self, func, branch_in, input, aux = False):
        """
        Numerically computes the Jacobian matrix of `func` at `input`.
        
        :param func: The function whose Jacobian is to be computed. Should take and return a torch tensor.
        :param input: The point (torch tensor) at which to compute the Jacobian.
        :param eps: Small perturbation used for finite differences.
        :return: Jacobian matrix as a torch tensor.
        """
        input = input.clone().detach().requires_grad_(False)
        input_dim = input.shape[1]
        if aux:
            jacobian = torch.zeros(*func(branch_in, input)[0].shape, input_dim, device=branch_in.device)
        else:
            jacobian = torch.zeros(*func(branch_in, input).shape, input_dim, device=branch_in.device)

        for i in range(input_dim):
            perturb = torch.zeros_like(input)
            perturb[:, i] = self.eps

            if aux:
                output_plus = func(branch_in, input + perturb)[0]
                output_minus = func(branch_in, input - perturb)[0]
            else:
                output_plus = func(branch_in, input + perturb)
                output_minus = func(branch_in, input - perturb)

            # approximate the partial derivatives using central finite differences
            jacobian[...,i] = (output_plus - output_minus) / (2 * self.eps)

        if aux:
            return jacobian, *func(branch_in, input)[1:]
        else:
            return jacobian

    def compute_hessian_num(self, func, input, branch_in):

        if self.eps < 1e-6:
            print('WARNING: Epsilon too small. Hessian computation may be unstable.')
        eps_ext = torch.full_like(input, self.eps)
        input_dim = input.shape[1]
        hessian = torch.zeros(*func(branch_in, input), input_dim, input_dim, device=input.device)
        for i in range(input.size(1)):
            for j in range(input.size(1)):
                input_perturbed_i = input.clone()
                input_perturbed_i[:, i] += eps_ext[:, i]
                input_perturbed_j = input.clone()
                input_perturbed_j[:, j] += eps_ext[:, j]
                input_perturbed_ij = input.clone()
                input_perturbed_ij[:, i] += eps_ext[:, i]
                input_perturbed_ij[:, j] += eps_ext[:, j]
                
                output_plus_i = func(branch_in, input_perturbed_i)
                output_plus_j = func(branch_in, input_perturbed_j)
                output_plus_ij = func(branch_in, input_perturbed_ij)
                output = func(branch_in, input)

                hessian[:, :, :, i, j] = (output_plus_ij - output_plus_i - output_plus_j + output) / self.eps**2

        return hessian
    
    def compute_jacobian_finite_diff(self, tensor, aux = False):
        """
        Compute the Jacobian of a tensor along two pixel axes using finite differences via torch.functional.conv2d.
        :param tensor: Input tensor, assumed to be an image.
        :param boundary_condition: Boundary condition for finite differences.
        :return: Jacobian tensor.
        IMPORTANT: This assumes a Jacobian w.r.t. two spatial dimensions, i.e. the last two dimensions of the tensor. For three, four, ... spatial dimensions, this function must be extended.
        """
        if tensor.ndim < 4:
            raise ValueError("Tensor must be at least 4-dimensional. We expect an image-based representation as input!")

        grad_axis1 = self.stencil_gradients(tensor, 'd_d0')
        grad_axis2 = self.stencil_gradients(tensor, 'd_d1')

        # concatenate new dimension before the pixel dimensions
        jacobian = torch.stack([grad_axis1, grad_axis2], dim=-3)

        if aux:
            return jacobian, tensor
        else:
            return jacobian

    def compute_jacobian_autograd(self, func, branch_in, trunk_in, aux = False, arg_grad = 1, batched = False, mode = 'rev'):
        
        if mode == 'rev':
            ag_mode = jacrev
        elif mode == 'fwd':
            ag_mode = jacfwd
        else:
            raise ValueError('Unknown differentiation mode.')
        
        if batched:
            jacobian = vmap(vmap(ag_mode(func, argnums=arg_grad, has_aux=aux), in_dims=(0, None)), in_dims=(None, 0), out_dims=1)
        else:
            jacobian = ag_mode(func, argnums=arg_grad, has_aux=aux)
            
        return jacobian(branch_in, trunk_in)

    def compute_hessian_autograd(self, func, branch_in, trunk_in, arg_grad, batched = False):
        if batched:
            batch_hessian = vmap(vmap(jacfwd(jacrev(func, argnums=arg_grad), argnums=arg_grad), in_dims=(0, None)), in_dims=(None, 0), out_dims=1)
            return batch_hessian(branch_in, trunk_in).squeeze(2,3)
        else:
            batch_hessian = jacfwd(jacrev(func, argnums=arg_grad), argnums=arg_grad)
            return batch_hessian(branch_in, trunk_in)