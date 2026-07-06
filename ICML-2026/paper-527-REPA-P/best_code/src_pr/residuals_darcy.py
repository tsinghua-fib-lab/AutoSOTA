import torch
from src_pr.grad_utils import *
import einops as ein
        
class ResidualsDarcy:
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary, reverse_d1, device = 'cpu', bcs = 'none', domain_length = 1., residual_grad_guidance = False, use_ddim_x0 = False, ddim_steps = 0):
        """
        Initialize the residual evaluation.

        :param model: The neural network model to compute the residuals for.
        :param n_steps: Number of steps for time discretization.
        """
        self.gov_eqs = 'darcy'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2

        if bcs == 'periodic':
            self.periodic = True

        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim
        
        self.reverse_d1 = reverse_d1
        # Note: we don't modify d1 here as findiff requires positive spacing
        # The reverse handling is done in the gradient calculations instead
        
        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.relu = torch.nn.ReLU()

        self.pixels_per_dim = pixels_per_dim

        # create stationary source field        
        w = 0.125
        r = 10.0
        domain_size = 1.
        # create point grid
        pixel_size = domain_size / pixels_per_dim
        start = pixel_size / 2
        end = domain_size - pixel_size / 2
        x = torch.linspace(start, end, steps=pixels_per_dim)
        y = torch.linspace(start, end, steps=pixels_per_dim)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        # compute the function values on the grid
        f_s = self.create_f_s(X, Y, w, r) # [pixels_per_dim, pixels_per_dim]
        self.f_s = generalized_image_to_b_xy_c(f_s.unsqueeze(0)).to(device) # [1, pixels_per_dim*pixels_per_dim, 1]

        if self.pixels_at_boundary:
            # adjust for boundary pixels via trapezoidal rule
            self.use_trapezoid = True
        else:
            # no need for boundary adjustments, simply take mean
            self.use_trapezoid = False

        self.device = device

        if self.use_trapezoid:
            self.trapezoidal_weights = self.create_trapezoidal_weights()

        self.residual_grad_guidance = residual_grad_guidance

        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

    def create_trapezoidal_weights(self):        
        # identify corner nodes
        trapezoidal_weights = torch.zeros((1, self.pixels_per_dim, self.pixels_per_dim))
        if self.model is not None:
            trapezoidal_weights = trapezoidal_weights.to(self.device)
        trapezoidal_weights[..., 0,0] = 1.
        trapezoidal_weights[..., 0,-1] = 1.
        trapezoidal_weights[..., -1,0] = 1.
        trapezoidal_weights[..., -1,-1] = 1.
        # identify boundary nodes
        trapezoidal_weights[..., 1:-1,0] = 2.
        trapezoidal_weights[..., 1:-1,-1] = 2.
        trapezoidal_weights[..., 0,1:-1] = 2.
        trapezoidal_weights[..., -1,1:-1] = 2.
        # identify interior nodes
        trapezoidal_weights[..., 1:-1,1:-1] = 4.
        # assert that no node is 0
        assert torch.all(trapezoidal_weights != 0)
        trapezoidal_weights *= (1./self.pixels_per_dim)**2 / 4.
        trapezoidal_weights = generalized_image_to_b_xy_c(trapezoidal_weights)
        return trapezoidal_weights

    # Define the source function using PyTorch operations
    def create_f_s(self, x, y, w = 0.125, r = 10.):
        condition1 = torch.abs(x - 0.5 * w) <= 0.5 * w
        condition2 = torch.abs(x - 1 + 0.5 * w) <= 0.5 * w
        condition3 = torch.abs(y - 0.5 * w) <= 0.5 * w
        condition4 = torch.abs(y - 1 + 0.5 * w) <= 0.5 * w

        result = torch.zeros_like(x)
        result[torch.logical_and(condition1, condition3)] = r
        result[torch.logical_and(condition2, condition4)] = -r
        return result

    def compute_residual(self, input, reduce = 'none', return_model_out = False, return_optimizer = False, return_inequality = False, sample = False, ddim_func = None, pass_through = False, return_projections = False, skip_model_call = False, given_model_output = None):

        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
        elif skip_model_call and given_model_output is not None:
            # Use the provided model output directly.
            x0_pred = given_model_output
            model_out = x0_pred
        else:
            assert len(input[0]) == 2 and isinstance(input[0], tuple), 'Input[0] must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(input[0])

            if self.residual_grad_guidance:
                assert not self.use_ddim_x0, 'Residual gradient guidance is not implemented with sample estimation for residual.'
                noisy_in.requires_grad = True
                residual_noisy_in = self.compute_residual(generalized_b_xy_c_to_image(noisy_in), pass_through = True)['residual']
                dr_dx = torch.autograd.grad(residual_noisy_in.abs().mean(), noisy_in)[0]
                if sample:
                    x0_pred = self.model.forward_with_guidance_scale(noisy_in, time, cond = dr_dx, guidance_scale = 3.) # There is no mentioning of value for the guidance scale in the paper and repo?!?
                    model_out = x0_pred
                else:
                    # Do not request projections during residual guidance.
                    x0_pred = self.model(noisy_in, time, cond = dr_dx, null_cond_prob = 0.1)
                    model_out = x0_pred
            else:
                if self.use_ddim_x0:
                    x0_pred, model_out = ddim_func(noisy_in, time, self.model, noisy_in.shape,self.ddim_steps, 0.)
                else:
                    # Request projection outputs when supported and enabled.
                    call_kwargs = {}
                    if return_projections and hasattr(self.model, 'use_projection_heads') and getattr(self.model, 'use_projection_heads'):
                        call_kwargs['return_projections'] = True
                    x0_pred = self.model(noisy_in, time, **call_kwargs)
                    model_out = x0_pred

        # Normalize model outputs, which may include projections.
        projections = None
        if isinstance(x0_pred, tuple):
            x0_pred, projections = x0_pred
        if isinstance(model_out, tuple):
            model_out = model_out[0]

        # Support direct b_xy_c inputs such as projection-head outputs.
        if x0_pred.ndim == 3:
            x0_pred = generalized_b_xy_c_to_image(x0_pred)

        assert len(x0_pred.shape) == 4, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape
        
        p = x0_pred[:, 0]
        permeability_field = x0_pred[:, 1]
        p_d0 = self.grads.stencil_gradients(p, mode='d_d0')
        p_d1 = self.grads.stencil_gradients(p, mode='d_d1')
        grad_p = torch.stack([p_d0, p_d1], dim=-3)
        p_d00 = self.grads.stencil_gradients(p, mode='d_d00')
        p_d11 = self.grads.stencil_gradients(p, mode='d_d11')
        perm_d0 = self.grads.stencil_gradients(permeability_field, mode='d_d0')
        perm_d1 = self.grads.stencil_gradients(permeability_field, mode='d_d1')
        velocity_jacobian = torch.zeros(batch_size, output_dim, self.input_dim, pixels_per_dim, pixels_per_dim, device=x0_pred.device, dtype=x0_pred.dtype)
        velocity_jacobian[:, 0, 0] = -permeability_field * p_d00 - perm_d0 * p_d0
        velocity_jacobian[:, 1, 1] = -permeability_field * p_d11 - perm_d1 * p_d1
        x0_pred = generalized_image_to_b_xy_c(x0_pred)
        grad_p = generalized_image_to_b_xy_c(grad_p)
        velocity_jacobian = generalized_image_to_b_xy_c(velocity_jacobian)
                
        # obtain equilibrium equations for residual
        eq_0 = velocity_jacobian[:,:, 0, 0] + velocity_jacobian[:, :, 1, 1] - self.f_s
        residual = eq_0

        # satisfy integral condition by definition, note that this does not change the residual since it only depends on the derivatives
        if self.use_trapezoid:
            p_int = self.trapezoidal_weights * x0_pred[..., 0].detach()
            correction = ein.reduce(p_int, 'b ... -> b 1', 'sum')
        else:
            # simple mean
            correction = ein.reduce(x0_pred[..., 0], 'b ... -> b 1', 'mean').detach()

        x0_pred_zero_p = x0_pred[:,:,0] - correction
        x0_pred_zero_p = torch.stack([x0_pred_zero_p, x0_pred[:,:,1]], dim=-1)
        x0_pred = x0_pred_zero_p
        
        # manually add BCs
        # reshape output to match image shape
        grad_p_img = generalized_b_xy_c_to_image(grad_p)
        residual_bc = torch.zeros_like(grad_p_img)
        residual_bc[:,0,0,:] = -grad_p_img[:,0,0,:] # xmin / top (acc. to matplotlib visualization)
        residual_bc[:,0,-1,:] = grad_p_img[:,0,-1,:] # xmax / bot
        if self.reverse_d1:
            residual_bc[:,1,:,0] = grad_p_img[:,1,:,0] # ymin / left
            residual_bc[:,1,:,-1] = -grad_p_img[:,1,:,-1] # ymax / right
        else:
            residual_bc[:,1,:,0] = -grad_p_img[:,1,:,0] # ymin / left
            residual_bc[:,1,:,-1] = grad_p_img[:,1,:,-1] # ymax / right

        residual_bc = generalized_image_to_b_xy_c(residual_bc)
        residual = torch.cat([eq_0.unsqueeze(-1), residual_bc], dim=-1)

        output = {}
        output['residual'] = residual

        if return_inequality:
            pass # not considered here
        if return_optimizer:
            pass # not considered here

        if return_model_out:
            output['model_out'] = model_out
        if return_projections and (projections is not None):
            output['projections'] = projections

        if reduce == 'full':
            # mean over all items in dict
            return {k: v.mean() for k, v in output.items()}
        elif reduce == 'per-batch':
            # mean over all but first dimension (batch dimension) for tensor values only
            reduced = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim > 1 and (k != 'model_out' and k != 'residual'):
                        reduced[k] = v.mean(dim=tuple(range(1, v.ndim)))
                    else:
                        reduced[k] = v
                else:
                    # keep non-tensor values (e.g., dict of projections) as-is
                    reduced[k] = v
            return reduced
        elif reduce == 'none':
            # return as-is
            return output
        else:
            raise ValueError('Unknown reduction method.')
        
    def residual_correction(self, x0_pred_in):

        # Ensure the model output is in the correct shape
        assert len(x0_pred_in.shape) == 3, 'Model output must be a tensor shaped as b_xy_c.'

        x0_pred = x0_pred_in.detach().clone()
        x0_pred.requires_grad_(True)

        residual_x0_pred = self.compute_residual(generalized_b_xy_c_to_image(x0_pred), pass_through = True)['residual']
        dr_dp = torch.autograd.grad(torch.sum(residual_x0_pred**2), x0_pred)[0][:,:,0] # residuals w.r.t. p
        
        jacobian_batch_size = 1 # reduced batch size to avoid OOM
        jacobian_vmap = vmap(jacfwd(self.compute_residual_direct, argnums=0, has_aux=False), in_dims=0, out_dims=0)
        num_batches = x0_pred.shape[0] // jacobian_batch_size + (0 if x0_pred.shape[0] % jacobian_batch_size == 0 else 1)
        jacobian_max_values = []
        for i in range(num_batches):
            batch_inputs = x0_pred[i*jacobian_batch_size:(i+1)*jacobian_batch_size]
            jacobian_vmapped = jacobian_vmap(batch_inputs).squeeze(1)[:,:,:,:,0] # NOTE we only consider residuals with respect to p
            batch_max_values = torch.max(jacobian_vmapped.reshape(len(batch_inputs), -1), dim=1)[0]
            jacobian_max_values.extend(batch_max_values.tolist())
            del jacobian_vmapped, batch_max_values
            torch.cuda.empty_cache()

        max_dr_dp = torch.tensor(jacobian_max_values).to(x0_pred.device)
        max_dr_dp = torch.clamp(max_dr_dp, max=1e12)
        correction_eps = 1.e-6 / max_dr_dp

        x0_pred_in[:,:,0] -= correction_eps.unsqueeze(1) * dr_dp.detach()

        # compute residual again based on correction
        residual_corrected = self.compute_residual(generalized_b_xy_c_to_image(x0_pred_in), pass_through = True)['residual']
        return x0_pred_in, residual_corrected
        
    # Compute the residual directly so simplify jacfwd call
    def compute_residual_direct(self, x0_output):
        # ensure the model output is in the correct shape
        if x0_output.ndim == 2:
            x0_output = x0_output.unsqueeze(0)
        assert len(x0_output.shape) == 3, 'Model output must be a tensor shaped as b_xy_c.'
        x0_output = generalized_b_xy_c_to_image(x0_output)

        p = x0_output[:, 0]
        permeability_field = x0_output[:, 1] # HACK remove gradients here?!
        p_d0 = self.grads.stencil_gradients(p, mode='d_d0')
        p_d1 = self.grads.stencil_gradients(p, mode='d_d1')
        grad_p = torch.stack([p_d0, p_d1], dim=-3)
        p_d00 = self.grads.stencil_gradients(p, mode='d_d00')
        p_d11 = self.grads.stencil_gradients(p, mode='d_d11')
        perm_d0 = self.grads.stencil_gradients(permeability_field, mode='d_d0')
        perm_d1 = self.grads.stencil_gradients(permeability_field, mode='d_d1')

        velocity_jacobian = torch.zeros_like(x0_output).unsqueeze(-3).repeat(1, 1, self.input_dim, 1, 1)
        velocity_jacobian[:, 0, 0] = -permeability_field * p_d00 - perm_d0 * p_d0
        velocity_jacobian[:, 1, 1] = -permeability_field * p_d11 - perm_d1 * p_d1
        grad_p = generalized_image_to_b_xy_c(grad_p)
        velocity_jacobian = generalized_image_to_b_xy_c(velocity_jacobian)
        
        # obtain equilibrium equations for residual
        eq_0 = velocity_jacobian[:,:, 0, 0] + velocity_jacobian[:, :, 1, 1] - self.f_s
        residual = eq_0
        
        # manually add BCs
        # reshape output to match image shape
        grad_p_img = generalized_b_xy_c_to_image(grad_p)
        # set up residual for BCs
        residual_bc = torch.zeros_like(grad_p_img)
        residual_bc[:,0,0,:] = -grad_p_img[:,0,0,:] # xmin / top (acc. to matplotlib visualization)
        residual_bc[:,0,-1,:] = grad_p_img[:,0,-1,:] # xmax / bot

        if self.reverse_d1:
            residual_bc[:,1,:,0] = grad_p_img[:,1,:,0] # ymin / left
            residual_bc[:,1,:,-1] = -grad_p_img[:,1,:,-1] # ymax / right
        else:
            residual_bc[:,1,:,0] = -grad_p_img[:,1,:,0] # ymin / left
            residual_bc[:,1,:,-1] = grad_p_img[:,1,:,-1] # ymax / right

        residual_bc = generalized_image_to_b_xy_c(residual_bc)
        residual = torch.cat([eq_0.unsqueeze(-1), residual_bc], dim=-1)
        return residual 