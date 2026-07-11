import torch
import math

def lambert_w_circ_exp(log_y, iterations = 5):
    """
    Helper function that computes log_y --> (W o exp)(log_y) with high numerical accuracy for a large range of inputs.
    The Lambert W function is the solution of the equation w*exp(w) = y for y > -1/e. Uses logarithmic recursion introduced in 
    Lajos Loczi, Guaranteed- and high-precision evaluation of the Lambert  function
    Applied Mathematics and Computation
    Volume 433, 15 November 2022
    https://www.sciencedirect.com/science/article/pii/S0096300322004805

    Args:
        log_y (torch.Tensor): Input tensor containing the logarithm of the argument for the Lambert W function.
        iterations (int): Number of iterations for the recursive computation. Default is 5.
        
    Returns:
        torch.Tensor: The computed values of the Lambert W function applied to exp(log_y).
    """

    # Initialize beta_n depending on y>e or y<=e
    beta_n = torch.where(log_y > 1., log_y - torch.log(log_y), (log_y - 1.).exp())
    
    # Logarithmic recursion introduced in 
    # Lajos Loczi, Guaranteed- and high-precision evaluation of the Lambert  function
    # Applied Mathematics and Computation
    # Volume 433, 15 November 2022
    # https://www.sciencedirect.com/science/article/pii/S0096300322004805
    for i in range(iterations):
        beta_n = beta_n/(1 + beta_n)*(1 + log_y - torch.log(beta_n))

    return beta_n


def lambert_w_sec(y, iterations = 5):
    """
    Helper function that computes the secondary branch of the Lambert W function
    y --> (W_{-1}(y) with high numerical accuracy.
    The Lambert W function W_{-1}(y) is one of the solutions of the equation w*exp(w) = y for 0 > y > -1/e.

    Uses logarithmic recursion introduced in 
    Lajos Loczi, Guaranteed- and high-precision evaluation of the Lambert  function
    Applied Mathematics and Computation
    Volume 433, 15 November 2022
    https://www.sciencedirect.com/science/article/pii/S0096300322004805
    """

    assert torch.all(-1/math.e < y < 0.), "y must be in the range (-1/e, 0)"

    beta_n = torch.where(y > -0.25, torch.log(-y) - torch.log(-torch.log(-y)), -1 - math.sqrt(2)*torch.sqrt(1 + math.e*y))
    
    for i in range(iterations):
        beta_n = beta_n/(1 + beta_n)*(1 + torch.log(y/beta_n))

    return beta_n


if __name__ == "__main__":
    log_y = torch.linspace(-10, 50, 20000)

    # Compute outputs
    fast_out = lambert_w_circ_exp(log_y, iterations=2)
    true_out = lambert_w_circ_exp(log_y, iterations=5)   # high-accuracy reference

    # Compute absolute and relative errors
    abs_err = (fast_out - true_out).abs()
    rel_err = abs_err / (true_out.abs() + 1e-30)

    print("==============================================")
    print("ACCURACY COMPARISON: fast vs iterative (20 iters)")
    print("==============================================")
    print(f"Max absolute error:     {abs_err.max().item():.3e}")
    print(f"Mean absolute error:    {abs_err.mean().item():.3e}")
    print(f"Median absolute error:  {abs_err.median().item():.3e}")
    print()
    print(f"Max relative error:     {rel_err.max().item():.3e}")
    print(f"Mean relative error:    {rel_err.mean().item():.3e}")
    print(f"Median relative error:  {rel_err.median().item():.3e}")