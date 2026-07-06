import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


__all__ = ["MyRBFKernel"]


# def postprocess_rbf(dist_mat):
#     return dist_mat.div_(-2).exp_()

def postprocess_rbf(dist_mat):
    return dist_mat.div(-2).exp()


class MyRBFKernel(Kernel):
    """
    My (Multi-Output) RBF kernel, equipped with outputscale attribute and additive delta kernel.
    Putting independent RBF kernels for each output. When doing forward, select along output dimension (-3).
    """
    has_lengthscale = True

    # override
    def __init__(
        self, multi_output:bool, has_outputscale:bool=False, delta_const:float=0., *args, **kwargs
    ):
        super(MyRBFKernel, self).__init__(*args, **kwargs)
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
            x1_ = x1.div(ls) # [..., P, n1, D_X]
            x2_ = x2.div(ls) # [..., P, n2, D_X]

        else:
            assert output_idx is None, "output_idx MUST be None for non multi-output case."
            x1_ = x1.div(self.lengthscale) # [..., n1, D_X]
            x2_ = x2.div(self.lengthscale) # [..., n2, D_X]

        # main computation
        covar_dist = self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
        K = postprocess_rbf(covar_dist)

        # wrap up with outputscale
        if self.has_outputscale:
            if self.multi_output:
                # select along output dimension (-1)
                outputscale = self.outputscale.index_select(-1, output_idx)
            else:
                outputscale = self.outputscale

            if diag:
                outputscale = outputscale.unsqueeze(-1) # [..., (P), 1]
                K = outputscale * K
            else:
                outputscale = outputscale.view(*outputscale.shape, 1, 1)  # [..., (P), 1, 1]
                K = outputscale * K

        # add delta_const if specified, by default it is 0.
        if self.delta_const != 0:
            K = K + self.delta_const * (covar_dist.detach() == 0)

        return K

    # override
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Call method is not implemented for MyRBFKernel. Use forward instead.")


if __name__ == "__main__":
    import torch
    from gpytorch.kernels import RBFKernel, ScaleKernel

    # hyperparameters and dataset
    D_X, P, batch_shape = 2, 5, ()

    p_select = torch.tensor([1, 2, 4])

    num_select_outputs = len(p_select)

    n1, n2, n3, n4, n5, n6 = 10, 20, 30, 40, 50, 60

    X1 = torch.randn(*batch_shape, num_select_outputs, n1, D_X)
    X2 = torch.randn(*batch_shape, num_select_outputs, n2, D_X)
    X3 = torch.randn(*batch_shape, 1, n3, D_X)
    X4 = torch.randn(*batch_shape, 1, n4, D_X)
    X5 = torch.randn(*batch_shape, n5, D_X)
    X6 = torch.randn(*batch_shape, n6, D_X)

    # Multi-output, no outputscale

    my_kernel1 = MyRBFKernel(
        multi_output=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,))
    )

    print("my_kernel1.raw_lengthscale", my_kernel1.raw_lengthscale.shape)

    res1 = my_kernel1.forward(X1, X2, diag=False, output_idx=p_select)
    assert res1.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1, n2])

    res2 = my_kernel1.forward(X1, X1, diag=True, output_idx=p_select)
    assert res2.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1])

    res3 = my_kernel1.forward(X1, X3, diag=False, output_idx=p_select)
    assert res3.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1, n3])

    res4 = my_kernel1.forward(X4, X4, diag=True, output_idx=None)  # all outputs
    assert res4.shape == torch.Size(batch_shape) + torch.Size([P, n4])

    # Non multi-output, no outputscale

    my_kernel2 = MyRBFKernel(
        multi_output=False, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape)
    )
    RBF_kernel = RBFKernel(
        ard_num_dims=D_X, batch_shape=torch.Size(batch_shape)
    )

    res5 = my_kernel2.forward(X5, X6, diag=False)
    reference_res5 = RBF_kernel(X5, X6, diag=False).to_dense()
    assert res5.shape == torch.Size(batch_shape) + torch.Size([n5, n6])
    assert torch.equal(res5, reference_res5)

    res6 = my_kernel2.forward(X5, X5, diag=True)
    reference_res6 = RBF_kernel(X5, X5, diag=True).to_dense()
    assert res6.shape == torch.Size(batch_shape) + torch.Size([n5])
    assert torch.equal(res6, reference_res6)

    # Multi-output, with outputscale

    my_kernel3 = MyRBFKernel(
        multi_output=True, has_outputscale=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,))
    )

    print("my_kernel3.raw_outputscale", my_kernel3.raw_outputscale.shape)

    res7 = my_kernel3.forward(X1, X2, diag=False, output_idx=p_select)
    assert res7.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1, n2])

    res8 = my_kernel3.forward(X1, X1, diag=True, output_idx=p_select)
    assert res8.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1])

    res9 = my_kernel3.forward(X1, X3, diag=False, output_idx=p_select)
    assert res9.shape == torch.Size(batch_shape) + torch.Size([num_select_outputs, n1, n3])

    res10 = my_kernel3.forward(X4, X4, diag=False, output_idx=None)  # all outputs
    assert res10.shape == torch.Size(batch_shape) + torch.Size([P, n4, n4])

    # Non multi-output, with outputscale

    my_kernel4 = MyRBFKernel(
        multi_output=False, has_outputscale=True, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape)
    )

    Scale_RBF_kernel = ScaleKernel(
        RBFKernel(ard_num_dims=D_X, batch_shape=torch.Size(batch_shape))
    )

    res11 = my_kernel4.forward(X5, X6, diag=False)
    reference_res11 = Scale_RBF_kernel(X5, X6, diag=False).to_dense()
    assert res11.shape == torch.Size(batch_shape) + torch.Size([n5, n6])
    assert torch.equal(res11, reference_res11)

    res12 = my_kernel4.forward(X5, X5, diag=True)
    reference_res12 = Scale_RBF_kernel(X5, X5, diag=True).to_dense()
    assert res12.shape == torch.Size(batch_shape) + torch.Size([n5])
    assert torch.equal(res12, reference_res12)

    # check delta_const
    my_kernel3_delta = MyRBFKernel(
        multi_output=True, has_outputscale=True, delta_const=1e-5, ard_num_dims=D_X, batch_shape=torch.Size(batch_shape + (P,))
    )  # compare with results from my_kernel3

    res7_delta = my_kernel3_delta.forward(X1, X2, diag=False, output_idx=p_select)
    assert torch.equal(res7, res7_delta)

    res8_delta = my_kernel3_delta.forward(X1, X1, diag=True, output_idx=p_select)
    assert torch.equal(res8 + 1e-5, res8_delta)






