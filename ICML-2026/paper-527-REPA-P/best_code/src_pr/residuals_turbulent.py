import torch

from src_pr.newdata import PhysicsLoss
from src_pr.unet_new import generalized_image_to_b_xy_c, generalized_b_xy_c_to_image


class ResidualsTurbulent:
    """
    Physics-inspired residual wrapper for the turbulent channel-flow dataset.

    Unlike Darcy, this dataset does not admit a closed PDE residual from the
    available variables, so we expose a residual map composed of:

    - wall no-slip residual at y=0
    - interior smoothness residual (Laplacian)
    - optional gradient residual

    The output tensor stays image-shaped and is later converted to ``b_xy_c`` so the
    existing diffusion training path can reuse its virtual-likelihood machinery.
    """

    def __init__(
        self,
        model,
        pixels_per_dim,
        device='cpu',
        lambda_wall=0.1,
        lambda_smooth=0.01,
        lambda_gradient=0.0,
        lambda_near_wall=0.0,
        near_wall_rows=3,
        residual_grad_guidance=False,
        use_ddim_x0=False,
        ddim_steps=0,
    ):
        self.gov_eqs = 'turbulent'
        self.model = model
        self.pixels_per_dim = pixels_per_dim
        self.device = device
        self.residual_grad_guidance = residual_grad_guidance
        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

        self.physics = PhysicsLoss(
            dx=1.0,
            dy=1.0,
            lambda_wall=lambda_wall,
            lambda_smooth=lambda_smooth,
            lambda_gradient=lambda_gradient,
            near_wall_rows=near_wall_rows,
            lambda_near_wall=lambda_near_wall,
            device=device,
        )

    @staticmethod
    def _sqrt_weight(weight, device, dtype):
        return torch.tensor(max(float(weight), 0.0), device=device, dtype=dtype).sqrt()

    def _compose_residual_image(self, x0_pred):
        if x0_pred.ndim == 3:
            x0_pred = generalized_b_xy_c_to_image(x0_pred)

        if x0_pred.ndim != 4:
            raise ValueError('Model output must be [B, C, H, W] or b_xy_c.')
        if x0_pred.shape[1] != 1:
            raise ValueError(f'Turbulent branch expects one output channel, got {x0_pred.shape[1]}')

        wall = self.physics.wall_residual_map(x0_pred)
        smooth = self.physics.smoothness_residual_map(x0_pred)

        residual_parts = [
            self._sqrt_weight(self.physics.lambda_wall, x0_pred.device, x0_pred.dtype) * wall,
            self._sqrt_weight(self.physics.lambda_smooth, x0_pred.device, x0_pred.dtype) * smooth,
        ]

        if self.physics.lambda_gradient > 0:
            grad = self.physics.gradient_residual_map(x0_pred)
            residual_parts.append(
                self._sqrt_weight(self.physics.lambda_gradient, x0_pred.device, x0_pred.dtype) * grad
            )

        if self.physics.lambda_near_wall > 0:
            yd = self.physics._y_dim(x0_pred)
            near_wall = torch.zeros_like(x0_pred)
            rows = min(self.physics.near_wall_rows, x0_pred.size(yd) - 1)
            if rows > 0:
                near_wall.narrow(yd, 1, rows).copy_(x0_pred.narrow(yd, 1, rows))
            residual_parts.append(
                self._sqrt_weight(self.physics.lambda_near_wall, x0_pred.device, x0_pred.dtype) * near_wall
            )

        return torch.cat(residual_parts, dim=1)

    def compute_residual(
        self,
        input,
        reduce='none',
        return_model_out=False,
        return_optimizer=False,
        return_inequality=False,
        sample=False,
        ddim_func=None,
        pass_through=False,
        return_projections=False,
        skip_model_call=False,
        given_model_output=None,
    ):
        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
        elif skip_model_call and given_model_output is not None:
            x0_pred = given_model_output
            model_out = x0_pred
        else:
            assert len(input[0]) == 2 and isinstance(input[0], tuple), (
                'Input[0] must be a tuple consisting of noisy signal and time.'
            )
            noisy_in, time = iter(input[0])

            if self.residual_grad_guidance:
                raise NotImplementedError('Residual gradient guidance is not implemented for turbulent data.')

            if self.use_ddim_x0:
                x0_pred, model_out = ddim_func(noisy_in, time, self.model, noisy_in.shape, self.ddim_steps, 0.)
            else:
                call_kwargs = {}
                if return_projections and hasattr(self.model, 'use_projection_heads') and getattr(self.model, 'use_projection_heads'):
                    call_kwargs['return_projections'] = True
                x0_pred = self.model(noisy_in, time, **call_kwargs)
                model_out = x0_pred

        projections = None
        if isinstance(x0_pred, tuple):
            x0_pred, projections = x0_pred
        if isinstance(model_out, tuple):
            model_out = model_out[0]

        residual_img = self._compose_residual_image(x0_pred)
        residual = generalized_image_to_b_xy_c(residual_img)

        output = {'residual': residual}
        if return_model_out:
            output['model_out'] = model_out
        if return_projections and (projections is not None):
            output['projections'] = projections
        if return_inequality:
            pass
        if return_optimizer:
            pass

        if reduce == 'full':
            return {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        if reduce == 'per-batch':
            reduced = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim > 1 and (k != 'model_out' and k != 'residual'):
                        reduced[k] = v.mean(dim=tuple(range(1, v.ndim)))
                    else:
                        reduced[k] = v
                else:
                    reduced[k] = v
            return reduced
        if reduce == 'none':
            return output
        raise ValueError('Unknown reduction method.')
