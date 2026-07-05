"""LRP (Layer-wise Relevance Propagation) autograd rules for AttnLRP.

This module implements the core autograd functions needed for AttnLRP:
- stop_gradient: Stop gradient flow completely
- divide_gradient: Divide gradient by a factor (Uniform Rule from Eq. 7)
- identity_rule_implicit: Handle non-linear activations (Identity Rule from Eq. 9)

Reference:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers (ICML 2024)
    https://arxiv.org/abs/2402.05602
"""

import torch
from torch.autograd import Function


def stop_gradient(input: torch.Tensor) -> torch.Tensor:
    """Stop the gradient from flowing through the input tensor.

    This is used in RMSNorm/LayerNorm to stop gradient flow through
    the variance calculation, implementing the identity rule.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor

    Returns
    -------
    torch.Tensor
        The detached tensor (same values, no gradient)
    """
    return input.detach()


def divide_gradient(input: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Divide the gradient by a factor during backpropagation.

    Implements the Uniform Rule (Equation 7 from the AttnLRP paper).
    Used after matmul or element-wise multiplication operations.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor
    factor : int
        The factor to divide the gradient by. Default is 2.

    Returns
    -------
    torch.Tensor
        The same tensor with modified backward gradient
    """
    return DivideGradientFn.apply(input, factor)


def identity_rule_implicit(fn, input: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Apply the identity rule implicitly through the Gradient*Input framework.

    Implements the Identity Rule (Equation 9 from the AttnLRP paper).
    Used for element-wise non-linear activation functions.

    The backward pass computes: gradient = (output / input) * out_relevance
    which, when multiplied by input in the final relevance computation,
    gives the identity rule: relevance = output * out_relevance / output = out_relevance

    Parameters
    ----------
    fn : callable
        The non-linear function to apply (e.g., SiLU, GELU)
    input : torch.Tensor
        The input tensor
    epsilon : float
        Small constant for numerical stability in division

    Returns
    -------
    torch.Tensor
        The output of fn(input) with modified backward pass
    """
    return IdentityRuleImplicitFn.apply(fn, input, epsilon)


class DivideGradientFn(Function):
    """Autograd Function that divides gradient by a constant factor.

    Forward pass: identity (returns input unchanged)
    Backward pass: divides gradient by factor

    This implements the Uniform Rule for element-wise multiplication
    and part of the handling for matrix multiplication.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, factor: int = 2) -> torch.Tensor:
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output / ctx.factor, None


class IdentityRuleImplicitFn(Function):
    """Autograd Function implementing the identity rule for non-linear activations.

    Forward pass: computes fn(input) and saves output/input ratio
    Backward pass: multiplies gradient by saved ratio

    This is more efficient than explicit LRP computation because it
    leverages the Gradient*Input framework.
    """

    @staticmethod
    def forward(ctx, fn, input: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        output = fn(input)
        if input.requires_grad:
            # Save output/input for backward
            # Adding epsilon prevents division by zero
            ctx.save_for_backward(output / (input + epsilon))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gradient is scaled by output/input ratio
        # When multiplied by input later, this gives the identity rule
        ratio = ctx.saved_tensors[0]
        gradient = ratio * grad_output
        return None, gradient, None
