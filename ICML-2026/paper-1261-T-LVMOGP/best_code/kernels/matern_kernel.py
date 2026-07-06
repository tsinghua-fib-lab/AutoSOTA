from typing import Optional
import math
import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


__all__ = ["MyMaternKernel"]


class MyMaternKernel(Kernel):
    """
    My (Multi-Output) Matern kernel, equipped with outputscale attribute and additive delta kernel.
    Putting independent Matern kernels for each output. When doing forward, select along output dimension (-3).
    """
    has_lengthscale = True

    # override
    def __init__(
        self, multi_output:bool, has_outputscale:bool=False, delta_const:float=0., nu: Optional[float]=2.5, **kwargs
    ):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MyMaternKernel, self).__init__(**kwargs)
        self.nu = nu
        self.multi_output = multi_output
        assert delta_const >= 0
        self.delta_const = delta_const  # 0 means no delta kernel
        if self.multi_output:
            assert len(self.batch_shape) > 0, \
                f"Batch shape for output dim must be specified, but get self.batch_shape={self.batch_shape}."

        self.has_outputscale = has_outputscale

        if has_outputscale:
            outputscale = torch.zeros(self.batch_shape)
            self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))

            outputscale_constraint = Positive()
            self.register_constraint("raw_outputscale", outputscale_constraint)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    # override
    def forward(self,
        x1: Tensor, x2: Tensor, output_idx:Tensor=None, diag=False, **params
    ) -> Tensor:
        # multi_output case:
        # x1, x2: [..., 1 or P, n, D_X]
        # output_idx: [P], the indices of outputs to be selected

        # NOT multi_output:
        # x1, x2: [..., n, D_X], output_idx=None

        # check
        if diag: assert torch.equal(x1, x2)

        # prepare
        if self.multi_output:
            total_P = self.lengthscale.size(-3)
            if output_idx is None:
                output_idx = torch.arange(total_P, device=x1.device)

            assert x1.size(-3) == 1 or x1.size(-3) == len(output_idx), \
                f"Expect output dim of x1 to be 1 or {len(output_idx)}, but get {x1.size(-3)}."

            assert x2.size(-3) == 1 or x2.size(-3) == len(output_idx), \
                f"Expect output dim of x2 to be 1 or {len(output_idx)}, but get {x2.size(-3)}."

            ls = self.lengthscale.index_select(-3, output_idx)
            x1_ = x1.div(ls)  # [..., P, n1, D_X]
            x2_ = x2.div(ls)  # [..., P, n2, D_X]

        else:
            assert output_idx is None, "output_idx MUST be None for non multi-output case."
            x1_ = x1.div(self.lengthscale)  # [..., n1, D_X]
            x2_ = x2.div(self.lengthscale)  # [..., n2, D_X]

        # main computation
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)

        K = constant_component * exp_component

        # warp up with outputscale
        if self.has_outputscale:
            if self.multi_output:
                # select along output dimension (-1)
                outputscale = self.outputscale.index_select(-1, output_idx)
            else:
                outputscale = self.outputscale

            if diag:
                outputscale = outputscale.unsqueeze(-1)  # [..., (P), 1]
                K = outputscale * K
            else:
                outputscale = outputscale.view(*outputscale.shape, 1, 1)  # [..., (P), 1, 1]
                K = outputscale * K

        # add delta_const if specified, by default it is 0.
        if self.delta_const != 0:
            K = K + self.delta_const * (distance.detach() == 0)

        return K

    # override
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Call method is not implemented for MyMaternKernel. Use forward instead.")


if __name__ == "__main__":
    import torch
    from gpytorch.kernels import MaternKernel, ScaleKernel
    torch.set_default_dtype(torch.float64)

    D_X, P, batch_shape = 2, 5, (3, )
    n1, n2 = 10, 20

    p_select = torch.tensor([1, 2, 4])

    x1 = torch.randn(*batch_shape, P, n1, D_X)
    x2 = torch.randn(*batch_shape, P, n2, D_X)

    for nu in [0.5, 1.5, 2.5]:
        _pytorch_kernel_with_P = MaternKernel(ard_num_dims=D_X, nu=nu, batch_shape=torch.Size(batch_shape + (P,)))
        pytorch_kernel_with_P = ScaleKernel(_pytorch_kernel_with_P, batch_shape=torch.Size(batch_shape + (P,)))

        my_kernel_with_P = MyMaternKernel(multi_output=True, has_outputscale=True, delta_const=0, nu=nu, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,)))

        kernel1_Kx1x1 = pytorch_kernel_with_P(x1, x1, diag=True).to_dense()
        kernel2_kx1x1 = my_kernel_with_P.forward(x1, x1, output_idx=None, diag=True)

        kernel1_kx1x2 = pytorch_kernel_with_P(x1, x2, diag=False).to_dense()
        kernel2_kx1x2 = my_kernel_with_P.forward(x1, x2, output_idx=None, diag=False)

        kernel1_kx1x2_select = torch.index_select(kernel1_kx1x2, -3, p_select)
        kernel2_kx1x2_select = my_kernel_with_P.forward(
            x1.index_select(-3, p_select),
            x2.index_select(-3, p_select),
            output_idx=p_select, diag=False
        )

        assert torch.equal(kernel1_Kx1x1, kernel2_kx1x1)
        # print(torch.abs(kernel1_kx1x2 - kernel2_kx1x2).max().item())
        # TODO: why not exactly equal?
        assert torch.allclose(kernel1_kx1x2, kernel2_kx1x2)
        assert torch.allclose(kernel1_kx1x2_select, kernel2_kx1x2_select)

        my_kernel_with_delta = MyMaternKernel(
            multi_output=True, has_outputscale=True, delta_const=1e-5, nu=nu, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,))
        )
        kernel2_kx1x1_delta = my_kernel_with_delta.forward(x1, x1, output_idx=None, diag=True)
        assert torch.equal(kernel2_kx1x1_delta, kernel1_Kx1x1 + 1e-5 * torch.ones(n1))


        print("All checks passed!")