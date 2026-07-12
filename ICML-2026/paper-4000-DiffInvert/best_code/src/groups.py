# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,invalid-name
from typing import Tuple
import torch
from torch import nn, Tensor
import math

from .utils.custom_grid_sample import grid_sample_zeros
from .utils.custom_interp1d import interp1d


class Group(nn.Module):

    def forward(self):
        print("Forward method is not implemented for groups")
        raise NotImplementedError

    @property
    def identity(self) -> Tensor:
        raise NotImplementedError

    def inverse(self, g: Tensor) -> Tensor:
        raise NotImplementedError

    def compose(self, g: Tensor, h: Tensor) -> Tensor:
        raise NotImplementedError


class ConnectedMatrixLieGroup(Group):
    basis: Tensor

    def __init__(self, basis: Tensor):
        """
        Arguments:
            basis: [num_generators, degree, degree] 
                or [num_generators_per_component, num_components, degree, degree]
        """
        super().__init__()

        self.num_generators_per_component = basis.shape[0]

        self.num_components = 1 if basis.ndim == 3 else basis.shape[1]

        self.num_generators = self.num_generators_per_component * self.num_components

        self.degree = basis.shape[-1]

        self.register_buffer('basis', basis.view(self.num_generators, self.degree, self.degree))

        assert basis.shape in [
            (self.num_generators, self.degree, self.degree),
            (self.num_generators_per_component, self.num_components, self.degree, self.degree)
        ], f"Invalid basis shape {basis.shape}"

        print(f"Initialized group with {self.num_generators} generators, "
              f"{self.num_components} components, degree {self.degree}")

    @property
    def device(self) -> torch.device:
        return self.basis.device

    @property
    def identity(self) -> Tensor:
        """
        Return:
            identity: [1, degree, degree] or [1, num_components, degree, degree]
        """
        eye = torch.eye(self.degree, device=self.device)
        if self.num_components == 1:
            return eye[None]
        return eye[None, None].repeat(1, self.num_components, 1, 1)

    def inverse(self, g: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, degree, degree] or [bsize, num_components, degree, degree]
        """
        return torch.inverse(g)

    def compose(self, g: Tensor, h: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, degree, degree] or [bsize, num_components, degree, degree]
            h: [bsize, degree, degree] or [bsize, num_components, degree, degree]
        """
        if self.num_components == 1:
            return torch.matmul(g, h)
        return torch.einsum('bcij,bcjk->bcik', g, h)

    def exp(self, coeff: Tensor) -> Tensor:
        """
        Arguments:
            coeff: [bsize, num_generators]
        Return:
            g: [bsize, degree, degree] or [bsize, num_components, degree, degree]
        """
        if self.num_components == 1:
            v = torch.einsum('bg,gij->bij', coeff, self.basis.to(coeff.dtype))
            g = torch.matrix_exp(v)
            return g
        bsize = coeff.shape[0]
        coeff = coeff.view(bsize, self.num_generators_per_component, self.num_components)
        basis = self.basis.view(self.num_generators_per_component, self.num_components, self.degree, self.degree)
        v = torch.einsum('bgc,gcij->bcij', coeff, basis.to(coeff.dtype))
        g = torch.matrix_exp(v)
        return g

    def act(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, degree, degree] or [bsize, num_components, degree, degree]
            x: [bsize, ...]
        """
        raise NotImplementedError

    @property
    def adjoint_trace(self) -> Tensor:
        """
        Return:
            adjoint_trace: [num_generators,]
        """
        raise NotImplementedError

    def random_coeff(self, num_samples: int) -> Tensor:
        """
        Randomized coeffs from the Lie algebra (not necessarily Haar)
        Return:
            coeff: [num_samples, num_generators]
        """
        return torch.randn(num_samples, self.num_generators, device=self.device)


def gen_hom_coords(device: torch.device, size: int) -> Tensor:
    """
    Homogenous coordinates for 2D points.
    Return:
        hom_coords: [size, size, 3]
    """
    coord_vectors = [
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device)
    ]
    euc_coords = torch.stack(torch.meshgrid(coord_vectors), 2)
    hom_coords = torch.cat((euc_coords, torch.ones([size, size, 1], device=device)), dim=2)
    return hom_coords


class Dummy(ConnectedMatrixLieGroup):
    """Dummy group that does nothing."""

    def __init__(self):
        basis = torch.empty(1, 2, 2)
        super().__init__(basis)


class ImageAffine(ConnectedMatrixLieGroup):
    """Connected component of the affine group in 2D."""

    def __init__(self, device='cpu'):
        basis = torch.zeros(6, 3, 3).to(device)
        basis[0] = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        basis[1] = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        basis[2] = torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])
        basis[3] = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
        basis[4] = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        basis[5] = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]])
        # apply scaling
        self.scale = torch.tensor([0.084, 0.084, 0.421, 0.084, 0.084, 0.421]).to(device)
        basis = self.scale[:, None, None] * basis
        super().__init__(basis)

        self.ad = torch.zeros(6, 6, 6).to(device)
        for i in range(6):
            for j in range(6):
                self.ad[i, :, j] = torch.flatten(torch.matmul(self.basis[i], self.basis[j]) - torch.matmul(self.basis[j], self.basis[i]))[:6]

    @property
    def adjoint_trace(self) -> Tensor:
        # under scaled structure constants
        return torch.tensor([0.084, 0., 0., 0., 0.084, 0.], device=self.device)

    def act_warp(self, g: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return:
            x_transformed: [bsize, c, h, w]
            warp_grid: [bsize, h, w, 2]
        """
        assert x.shape[2] == x.shape[3], "input image must be square"
        hom_coords = gen_hom_coords(device=self.device, size=x.shape[2])
        warp_grid = torch.einsum('bij,xyj->byxi', g, hom_coords)[..., :2]
        x_transformed = grid_sample_zeros(x, warp_grid)
        return x_transformed, warp_grid

    def act(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return:
            x_transformed: [bsize, c, h, w]
        """
        x_transformed, _ = self.act_warp(g, x)
        return x_transformed

    def log_det_jacobian(self, g: Tensor, x: Tensor) -> Tensor:  # pylint: disable=unused-argument
        """
        Log determinant of the Jacobian of the action on coordinates.
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return: [bsize,]
        """
        b, c, h, w = x.shape
        a = g[:, :2, :2]
        logdet = torch.slogdet(a)[1]
        logdet = logdet.view(b, 1, 1).expand(b, h, w)
        return logdet
    
    def dvol(self, v, precision=10):
        """
        Computes dvol(v), where v is a batch of Lie algebra elements
        """
        adv = torch.tensordot(v, self.ad, ([-1], [0]))

        e = torch.zeros_like(adv)
        for i in range(precision):
            e = e + (1 / math.factorial(i+1)) * torch.matrix_power( -1. * adv, i)

        return torch.abs(torch.det(e))


class ImageHomography(ConnectedMatrixLieGroup):
    """Homography group of 2D image transformations."""

    def __init__(self, device='cpu'):
        basis = torch.zeros(8, 3, 3).to(device)
        e1 = torch.tensor([[1., 0., 0.]]).T
        e2 = torch.tensor([[0., 1., 0.]]).T
        e3 = torch.tensor([[0., 0., 1.]]).T
        basis[0] = torch.matmul(e1, e1.T) - (1 / 3) * torch.eye(3)
        basis[1] = torch.matmul(e1, e2.T)
        basis[2] = torch.matmul(e1, e3.T)
        basis[3] = torch.matmul(e2, e1.T)
        basis[4] = torch.matmul(e2, e2.T) - (1 / 3) * torch.eye(3)
        basis[5] = torch.matmul(e2, e3.T)
        basis[6] = torch.matmul(e3, e1.T)
        basis[7] = torch.matmul(e3, e2.T)
        # apply scaling
        self.scale = torch.tensor([0.15, 0.35, 0.5, 0.35, 0.15, 0.5, 0.15, 0.15]).to(device)
        basis = self.scale[:, None, None] * basis
        super().__init__(basis)

        _basis = []
        for i in range(8):
            _basis.append(basis[i].flatten())
        basis_mtx = torch.stack(_basis).T
        basis_pinv = torch.linalg.pinv(basis_mtx)

        self.ad = torch.zeros(8, 8, 8).to(device)
        for i in range(8):
            for j in range(8):
                self.ad[i, :, j] = basis_pinv @ torch.flatten(torch.matmul(basis[i], basis[j]) - torch.matmul(basis[j], basis[i]))

    @property
    def adjoint_trace(self) -> Tensor:
        return torch.zeros(8, device=self.device)

    def act_warp(self, g: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return:
            x_transformed: [bsize, c, h, w]
            warp_grid: [bsize, h, w, 2]
        """
        assert x.shape[2] == x.shape[3], "input image must be square"
        hom_coords = gen_hom_coords(device=self.device, size=x.shape[2])
        pre_warp_grid = torch.einsum('bij,xyj->byxi', g, hom_coords)
        warp_grid = pre_warp_grid[..., :2] / pre_warp_grid[..., 2:3].clamp_min(1e-12)
        x_transformed = grid_sample_zeros(x, warp_grid)
        return x_transformed, warp_grid

    def act(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return:
            x_transformed: [bsize, c, h, w]
        """
        x_transformed, _ = self.act_warp(g, x)
        return x_transformed

    def log_det_jacobian(self, g: Tensor, x: Tensor) -> Tensor:  # pylint: disable=unused-argument
        """
        Log determinant of the Jacobian of the action on coordinates.
        Arguments:
            g: [bsize, 3, 3]
            x: [bsize, c, h, w]
        Return: [bsize,]
        """
        assert x.shape[2] == x.shape[3], "input image must be square"
        hom_coords = gen_hom_coords(device=self.device, size=x.shape[2])
        pre_warp_grid = torch.einsum('bij,xyj->byxi', g, hom_coords)
        denom = pre_warp_grid[..., 2:, None].clamp_min(1e-12)
        a = g[:, None, None, :2, :2] / denom
        a -= (g[:, None, None, 2:, :2] * pre_warp_grid[..., :2, None]) / denom.pow(2)
        # return torch.slogdet(a)[1].sum((1, 2))
        return torch.slogdet(a)[1]
    
    def dvol(self, v, precision=10):
        """
        Computes dvol(v), where v is a batch of Lie algebra elements
        """
        adv = torch.tensordot(v, self.ad, ([-1], [0]))

        e = torch.zeros_like(adv)
        for i in range(precision):
            e = e + (1 / math.factorial(i+1)) * torch.matrix_power( -1. * adv, i)

        return torch.abs(torch.det(e))


class Heat1D(Group):
    """1D Heat equation symmetry group (connected component at identity)."""
    def __init__(self, nu: float, sensors_x: Tensor):
        """
        global parameterization
        g = (alpha, beta, gamma, delta, lambda_0, lambda_1, log_sigma)
        arbitrary constants with:
            alpha * delta - beta * gamma = 1
            log_sigma = log(sigma), sigma > 0
        """
        super().__init__()
        assert nu > 0. and nu != 1., "nu should be a positive constant not equal to 1"
        assert sensors_x.ndim == 1
        self.nu = nu
        self.num_generators = 6
        self._dim = 7
        self.sensors_x = sensors_x

        # special linear basis
        sl_basis = torch.zeros(3, 2, 2)
        sl_basis[0] = torch.tensor([[0., 1.], [0., 0.]])
        sl_basis[1] = torch.tensor([[1., 0.], [0., -1.]])
        sl_basis[2] = torch.tensor([[0., 0.], [1., 0.]])
        # apply scaling
        scale = torch.tensor([0.02, 0.02, 0.02])
        sl_basis = scale[:, None, None] * sl_basis
        self.sl_basis = sl_basis.to(self.device).to(torch.float64)

    @property
    def device(self) -> torch.device:
        return self.sensors_x.device

    @property
    def identity(self) -> Tensor:
        return torch.tensor([1., 0., 0., 1., 0., 0., 0.], device=self.device, dtype=torch.float64)

    @property
    def adjoint_trace(self) -> Tensor:
        return torch.zeros(6, device=self.device, dtype=torch.float64)

    def inverse(self, g: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, dim]
        """
        assert g.dtype == torch.float64
        alpha, beta, gamma, delta, lambda_0, lambda_1, log_sigma = g.unbind(1)

        lambda_0_inv = beta * lambda_1 - alpha * lambda_0
        lambda_1_inv = -delta * lambda_1 + gamma * lambda_0

        sigma_inv = -log_sigma
        sigma_inv -= 0.25 * lambda_0 * lambda_1
        sigma_inv -= 0.25 * lambda_0_inv * lambda_1_inv

        g_inv = torch.stack([
            delta, -beta, -gamma, alpha,
            lambda_0_inv, lambda_1_inv, sigma_inv
        ], dim=1)
        return g_inv

    def _to_mat(self, alpha: Tensor, beta: Tensor, gamma: Tensor, delta: Tensor) -> Tensor:
        return torch.stack([
            torch.stack([alpha, beta], dim=1),
            torch.stack([gamma, delta], dim=1)
        ], dim=1)

    def _from_mat(self, mat: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = mat[:, 0, 0]
        beta = mat[:, 0, 1]
        gamma = mat[:, 1, 0]
        delta = mat[:, 1, 1]
        return alpha, beta, gamma, delta

    def compose(self, g: Tensor, h: Tensor) -> Tensor:
        """
        # https://arxiv.org/abs/2410.02698
        Arguments:
            g: [bsize, dim]
            h: [bsize, dim]
        """
        assert g.dtype == torch.float64
        assert h.dtype == torch.float64

        alpha_g, beta_g, gamma_g, delta_g, lambda_0_g, lambda_1_g, log_sigma_g = g.unbind(1)
        alpha_h, beta_h, gamma_h, delta_h, lambda_0_h, lambda_1_h, log_sigma_h = h.unbind(1)

        # alpha, beta, gamma, delta
        mat_g = self._to_mat(alpha_g, beta_g, gamma_g, delta_g)
        mat_h = self._to_mat(alpha_h, beta_h, gamma_h, delta_h)
        prod = torch.einsum('bij,bjk->bik', mat_g, mat_h)
        alpha_gh, beta_gh, gamma_gh, delta_gh = self._from_mat(prod)

        # lambda_0, lambda_1, log_sigma
        tmp1 = beta_h * lambda_1_g + delta_h * lambda_0_g
        tmp2 = alpha_h * lambda_1_g + gamma_h * lambda_0_g

        lambda_0_gh = lambda_0_h + tmp1
        lambda_1_gh = lambda_1_h + tmp2

        log_sigma_gh = log_sigma_g + log_sigma_h
        log_sigma_gh += 0.25 * (lambda_0_g * lambda_1_g - tmp1 * tmp2)
        log_sigma_gh -= 0.5 * lambda_0_h * tmp2

        gh = torch.stack([
            alpha_gh, beta_gh, gamma_gh, delta_gh,
            lambda_0_gh, lambda_1_gh, log_sigma_gh
        ], dim=1)
        return gh

    def exp(self, coeff: Tensor) -> Tensor:
        """
        exp map from a lie algebra to a lie group
        Arguments:
            coeff: [bsize, 6] (sl_coeff [bsize, 3], heis_coeff [bsize, 3])
        Return:
            g: [bsize, dim]
        """
        assert coeff.dtype == torch.float64
        assert coeff.shape[1] == self.num_generators
        sl_coeff = coeff[:, :3]
        heis_coeff = coeff[:, 3:]

        # SL(2, R) part
        sl_mat = torch.einsum('bk,kij->bij', sl_coeff, self.sl_basis)
        sl_exp = torch.matrix_exp(sl_mat)
        alpha, beta, gamma, delta = self._from_mat(sl_exp)

        # H(1, R) part
        lambda0, lambda1, log_sigma = heis_coeff.unbind(1)

        # apply semi-direct product rule
        new_lambda0 = beta * lambda1 + delta * lambda0
        new_lambda1 = alpha * lambda1 + gamma * lambda0
        new_log_sigma = log_sigma + 0.25 * (lambda0 * lambda1 - new_lambda0 * new_lambda1)

        g = torch.stack([
            alpha, beta, gamma, delta,
            new_lambda0, new_lambda1, new_log_sigma
        ], dim=1)
        return g

    def random_coeff(self, num_samples: int) -> Tensor:
        return torch.randn(num_samples, self.num_generators,
                           device=self.device, dtype=torch.float64)

    def resample_to_sensors(self, jet: Tensor) -> Tensor:
        """
        Resample signal in jet to the sensor locations.
        Args:
            jet: [bsize, 3, points] (x, t, u)
        Return:
            u_resampled: [bsize, sensors] (u,)
        """
        assert jet.dtype == torch.float64
        assert jet.shape[1] == 3

        x, t, u = jet.unbind(1)

        assert (t[:, 1:] - t[:, :-1]).abs().max().item() < 1e-5, "t should be constant"

        u_resampled = interp1d(x, u, self.sensors_x)
        assert isinstance(u_resampled, Tensor)

        return u_resampled

    def act(self, g: Tensor, jet: Tensor) -> Tensor:
        """
        Args:
            g: [bsize, dim]
            jet: [bsize, 2, points] (x, t) or [bsize, 3, points] (x, t, u)
        Return:
            jet_transformed: [bsize, 2, points] or [bsize, 3, points]
        """
        assert g.dtype == torch.float64
        assert jet.dtype == torch.float64
        assert jet.shape[1] in (2, 3)

        alpha, beta, gamma, delta, lambda_0, lambda_1, log_sigma = g.split(1, dim=1)

        if jet.shape[1] == 3:
            x, t, u = jet.unbind(1)

            tmp1 = gamma * t + delta
            tmp2 = x + lambda_1 * t + lambda_0

            x_hat = tmp2 / tmp1.clamp_min(1e-12)
            t_hat = (alpha * t + beta) / tmp1.clamp_min(1e-12)

            tmp3 = gamma * (tmp2 ** 2) / (4 * self.nu * tmp1).clamp_min(1e-12)
            tmp4 = lambda_1 * x / (2 * self.nu)
            tmp5 = (lambda_1 ** 2) * t / (4 * self.nu)

            u_hat = tmp1.abs().sqrt() * u * torch.exp(log_sigma + tmp3 - tmp4 - tmp5)

            return torch.stack([x_hat, t_hat, u_hat], dim=1)

        x, t = jet.unbind(1)

        tmp1 = gamma * t + delta
        tmp2 = x + lambda_1 * t + lambda_0

        x_hat = tmp2 / tmp1.clamp_min(1e-12)
        t_hat = (alpha * t + beta) / tmp1.clamp_min(1e-12)

        return torch.stack([x_hat, t_hat], dim=1)


class Burgers1D(Group):
    """1D Burgers equation symmetry group."""
    def __init__(self, sensors_x: Tensor):
        super().__init__()
        assert sensors_x.ndim == 1
        self.num_generators = 5
        self._dim = 6
        self.sensors_x = sensors_x

        # special linear basis
        sl_basis = torch.zeros(3, 2, 2)
        sl_basis[0] = torch.tensor([[0., 1.], [0., 0.]])
        sl_basis[1] = torch.tensor([[1., 0.], [0., -1.]])
        sl_basis[2] = torch.tensor([[0., 0.], [1., 0.]])
        # apply scaling
        scale = torch.tensor([0.1, 0.1, 0.1])
        sl_basis = scale[:, None, None] * sl_basis
        self.sl_basis = sl_basis.to(self.device).to(torch.float64)

    @property
    def device(self) -> torch.device:
        return self.sensors_x.device

    @property
    def identity(self) -> Tensor:
        return torch.tensor([1., 0., 0., 1., 0., 0.], device=self.device, dtype=torch.float64)

    @property
    def adjoint_trace(self) -> Tensor:
        return torch.zeros(6, device=self.device, dtype=torch.float64)

    def inverse(self, g: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, dim]
        """
        assert g.dtype == torch.float64
        alpha, beta, gamma, delta, lambda_0, lambda_1 = g.unbind(1)

        lambda_0_inv = beta * lambda_1 - alpha * lambda_0
        lambda_1_inv = -delta * lambda_1 + gamma * lambda_0

        g_inv = torch.stack([
            delta, -beta, -gamma, alpha,
            lambda_0_inv, lambda_1_inv
        ], dim=1)
        return g_inv

    def _to_mat(self, alpha: Tensor, beta: Tensor, gamma: Tensor, delta: Tensor) -> Tensor:
        return torch.stack([
            torch.stack([alpha, beta], dim=1),
            torch.stack([gamma, delta], dim=1)
        ], dim=1)

    def _from_mat(self, mat: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = mat[:, 0, 0]
        beta = mat[:, 0, 1]
        gamma = mat[:, 1, 0]
        delta = mat[:, 1, 1]
        return alpha, beta, gamma, delta

    def compose(self, g: Tensor, h: Tensor) -> Tensor:
        """
        Arguments:
            g: [bsize, dim]
            h: [bsize, dim]
        """
        assert g.dtype == torch.float64
        assert h.dtype == torch.float64

        alpha_g, beta_g, gamma_g, delta_g, lambda_0_g, lambda_1_g = g.unbind(1)
        alpha_h, beta_h, gamma_h, delta_h, lambda_0_h, lambda_1_h = h.unbind(1)

        # alpha, beta, gamma, delta
        mat_g = self._to_mat(alpha_g, beta_g, gamma_g, delta_g)
        mat_h = self._to_mat(alpha_h, beta_h, gamma_h, delta_h)
        prod = torch.einsum('bij,bjk->bik', mat_g, mat_h)
        alpha_gh, beta_gh, gamma_gh, delta_gh = self._from_mat(prod)

        # lambda_0, lambda_1
        lambda_0_gh = lambda_0_h + (beta_h * lambda_1_g + delta_h * lambda_0_g)
        lambda_1_gh = lambda_1_h + (alpha_h * lambda_1_g + gamma_h * lambda_0_g)

        return torch.stack([
            alpha_gh, beta_gh, gamma_gh, delta_gh,
            lambda_0_gh, lambda_1_gh
        ], dim=1)

    def exp(self, coeff: Tensor) -> Tensor:
        """
        exp map from a lie algebra to a lie group
        Arguments:
            coeff: [bsize, 5] (sl_coeff [bsize, 3], R2_coeff [bsize, 2])
        Return:
            g: [bsize, dim]
        """
        assert coeff.dtype == torch.float64
        assert coeff.shape[1] == self.num_generators

        sl_coeff = coeff[:, :3]
        r2_coeff = coeff[:, 3:]

        # SL(2, R) part
        sl_mat = torch.einsum('bk,kij->bij', sl_coeff, self.sl_basis)
        sl_exp = torch.matrix_exp(sl_mat)
        alpha, beta, gamma, delta = self._from_mat(sl_exp)

        # R2 part
        lambda0, lambda1 = r2_coeff.unbind(1)

        # apply semi-direct product rule
        new_lambda0 = beta * lambda1 + delta * lambda0
        new_lambda1 = alpha * lambda1 + gamma * lambda0

        g = torch.stack([
            alpha, beta, gamma, delta,
            new_lambda0, new_lambda1
        ], dim=1)
        return g

    def random_coeff(self, num_samples: int) -> Tensor:
        return torch.randn(num_samples, self.num_generators,
                           device=self.device, dtype=torch.float64)

    def resample_to_sensors(self, jet: Tensor) -> Tensor:
        """
        Resample signal in jet to the sensor locations.
        Args:
            jet: [bsize, 3, points] (x, t, u)
        Return:
            u_resampled: [bsize, sensors] (u,)
        """
        assert jet.dtype == torch.float64
        assert jet.shape[1] == 3

        x, t, u = jet.unbind(1)

        assert (t[:, 1:] - t[:, :-1]).abs().max().item() < 1e-5, "t should be constant"

        u_resampled = interp1d(x, u, self.sensors_x)
        assert isinstance(u_resampled, Tensor)

        return u_resampled

    def act(self, g: Tensor, jet: Tensor) -> Tensor:
        """
        Args:
            g: [bsize, dim]
            jet: [bsize, 2, points] (x, t) or [bsize, 3, points] (x, t, u)
        Return:
            jet_transformed: [bsize, 2, points] or [bsize, 3, points]
        """
        assert g.dtype == torch.float64
        assert jet.dtype == torch.float64
        assert jet.shape[1] in (2, 3)

        alpha, beta, gamma, delta, lambda_0, lambda_1 = g.split(1, dim=1)

        if jet.shape[1] == 3:
            x, t, u = jet.unbind(1)

            tmp1 = gamma * t + delta
            tmp2 = x + lambda_1 * t + lambda_0

            x_hat = tmp2 / tmp1.clamp_min(1e-12)
            t_hat = (alpha * t + beta) / tmp1.clamp_min(1e-12)

            tmp3 = -gamma * x + lambda_1 * delta - lambda_0 * gamma

            u_hat = tmp1 * u + tmp3

            return torch.stack([x_hat, t_hat, u_hat], dim=1)

        x, t = jet.unbind(1)

        tmp1 = gamma * t + delta
        tmp2 = x + lambda_1 * t + lambda_0

        x_hat = tmp2 / tmp1.clamp_min(1e-12)
        t_hat = (alpha * t + beta) / tmp1.clamp_min(1e-12)
        return torch.stack([x_hat, t_hat], dim=1)
