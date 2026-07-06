import torch
import numpy as np
import scipy.linalg as slin


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply


def main():
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(trace_expm, input)

    input = torch.tensor([[1, 2], [3, 4.]], requires_grad=True)
    tre = trace_expm(input)
    f = 0.5 * tre * tre
    print('f\n', f.item())
    f.backward()
    print('grad\n', input.grad)


if __name__ == '__main__':
    main()
