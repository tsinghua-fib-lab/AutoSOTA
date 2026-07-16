import torch
import numpy as np
from torch.autograd.functional import hessian
from extraction import calc_extraction_loss, get_kkt_loss, evaluate_extraction, viz_nns
from common_utils.common import now
from tqdm import tqdm 
from scipy.optimize import minimize_scalar

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def compute_min_eig_Tj_hvp(args, X, y, lambda_, model, method="Haim",model_init=None):
    """
    Inputs:
        X: [N, C, H, W] or [N, d]
        y: [N] or [N,1]
        lambda_: [N,1]
        model
    Outputs:
        min_eigs: [N]
        min_vecs: [N, d]
    """
    N = X.shape[0]
    X_flat = X.reshape(N, -1)  # [N, d]
    d = X_flat.shape[1]

    y = y.reshape(-1)
    lambda_ = lambda_.reshape(-1)

    device = X.device
    min_eigs = torch.zeros(N, device=device)
    min_vecs = torch.zeros(N, d, device=device)

    def f_grad(args, x, model,l,y,method="Haim",model_init=None, graph=True):
        l=l.squeeze()
        values = model(x).squeeze(1)
        if method == 'Haim':
            if args.output_dim > 1: # multiclass
                phi_yi = values.gather(1, y.view(-1, 1)).squeeze()
                values_copy = values.clone()
                values_copy = values_copy.scatter(1, y.view(-1, 1), -torch.inf)
                second_best = values_copy.max(dim=1)[0].squeeze()
                l_margins = (phi_yi - second_best) * l
                output = l_margins
            else: # binary classification
                # all three shape should be (n)
                output = values * l * y
            grad = torch.autograd.grad(
                outputs=output,
                inputs=model.parameters(),
                grad_outputs=torch.ones_like(output, requires_grad=False, device=output.device).div(args.extraction_data_amount),
                create_graph=graph,
                retain_graph=graph,
            )
        elif method == 'Loo':
            output_i = model_init(x).squeeze(1)*l
            para_i = list(model_init.parameters())
            grad_i = torch.autograd.grad(
                outputs=output_i,
                inputs=para_i,
                grad_outputs=torch.ones_like(output_i, requires_grad=False, device=output_i.device).div(args.extraction_data_amount),
                create_graph=graph,
                retain_graph=graph,
            )
            output_f = values * l
            para_f = list(model.parameters())
            grad_f = torch.autograd.grad(
                outputs=output_f,
                inputs=para_f,
                grad_outputs=torch.ones_like(output_f, requires_grad=False, device=output_f.device).div(args.extraction_data_amount),
                create_graph=graph,
                retain_graph=graph,
            )
            grad = [(grad_i[i] + grad_f[i])/2 for i in range(len(grad_i))]
        else:
            raise ValueError("Unknown method: {}".format(method))
        return grad

    # flatten model params theta and detach (no grad)
    theta = torch.cat([p.flatten() for p in model.parameters()]).detach()

    # compute r = theta - sum_i lambda_i * y_i * grad_theta(Φ(x_i))
    r = theta.clone()
    model.eval()
    params = list(model.parameters())
    values = model(X).squeeze()
    grad = f_grad(args,X,model,lambda_,y,method=method,model_init=model_init,graph=False)
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grad])
    r = r - grad_flat

    # compute f_x(z) = dot( grad_theta_fn(z), r )
    def f_dot_r(x_single,lambdaj,yj=1):
        # x_single: [d] (flattened) or [1, C, H, W]
        x_input = x_single.unsqueeze(0)
        grads = f_grad(args,x_input,model,lambdaj,yj,method=method,model_init=model_init)
        grads_flat = torch.cat([g.contiguous().view(-1) for g in grads])*args.extraction_data_amount
        # scalar
        val = torch.dot(grads_flat, r)
        return val

    from torch.autograd.functional import hvp

    def Tj_matvec(x_single, lambdaj, yj, v):
        """
        x_single: tensor shape [d], requires_grad=True
        v: tensor shape [d], same device as x_single
        return: T_j @ v  (shape [d])
        """
        f = lambda z: f_dot_r(z,lambdaj,yj)
        _, hv = hvp(f, (x_single,), (v,), create_graph=False)
        hv = hv[0]
        Tdotv = -2.0 * hv
        return Tdotv

    for j in tqdm(range(N)):
        xj = X_flat[j].detach()
        xj = xj.to(device).detach().requires_grad_(True)  # ensure grad enabled for hvp
        lambdaj = lambda_[j:j+1]
        yj = y[j:j+1]

        # init v (in x-space)
        v = torch.randn(d, device=device)
        v = v / (v.norm() + 1e-12)

        # Lanczos algorithm to find min eigenvalue/vector
        def lanczos_min_eig(hvp, d, m=20):
            """
            hvp: function v -> H v
            d: dimension
            m: Krylov dimension
            """
            Q = []
            alpha = []
            beta = []
            q = torch.randn(d, device=device)
            q = q / q.norm()
            Q.append(q)

            for j in range(m):
                z = hvp(Q[-1])
                a = torch.dot(Q[-1], z)
                alpha.append(a)
                z = z - a * Q[-1]
                if j > 0:
                    z = z - beta[-1] * Q[-2]
                b = z.norm()
                beta.append(b)
                if b < 1e-6:
                    break
                Q.append(z / b)

            # Build tridiagonal matrix T
            T = torch.zeros(len(alpha), len(alpha), device=device)
            for i in range(len(alpha)):
                T[i, i] = alpha[i]
                if i + 1 < len(alpha):
                    T[i, i + 1] = beta[i]
                    T[i + 1, i] = beta[i]

            w, v = torch.linalg.eigh(T)
            idx = torch.argmin(w)
            eigval = w[idx]
            eigvec = sum(v[i, idx] * Q[i] for i in range(len(v)))

            return eigval, eigvec

        eigval, eigvec = lanczos_min_eig(
            hvp=lambda v: Tj_matvec(xj, lambdaj, yj, v.to(device)),
            d=d
        )
        min_eigs[j] = eigval
        min_vecs[j] = eigvec
        torch.cuda.empty_cache()

    return min_eigs, min_vecs



def select_split_indices(args, w_min, V, growth_rate=0.2):
    """
    select indices of samples to split based on their minimum eigenvalues
    """
    negative_ratio = (w_min < 0).float().mean().item()
    print(f"Negative eigenvalue ratio: {negative_ratio:.2%}")

    if negative_ratio < growth_rate:
        split_index = torch.nonzero(w_min < 0, as_tuple=False).squeeze(1)
        print(f"Negative eigenvalue ratio is below the growth rate; splitting all negative-eigenvalue samples ({split_index.numel()} total)")
        values, order = torch.sort(w_min)  
    else:
        growth_rate = 0.5
        n = w_min.numel()
        k = int(torch.ceil(torch.tensor(n * growth_rate)).item())
        values, order = torch.sort(w_min)       
        candidate_idx = order[:k]
        mask = values[:k] < -0.1
        filtered_idx = candidate_idx[mask]
        max_keep = 150
        split_index = filtered_idx[:max_keep]

    if split_index.numel() > 0:
        print("min eigenvalue:", w_min[order[:5]].detach().cpu().numpy())
        all_eigs = w_min.detach().cpu().numpy()
        selected_eigs = w_min[split_index].detach().cpu().numpy()

        print("=== All eigenvalues ===")
        print("count :", all_eigs.size)
        print("min   :", all_eigs.min())
        print("max   :", all_eigs.max())
        print("mean  :", all_eigs.mean())
        print("median:", np.median(all_eigs))
        print("std   :", all_eigs.std())
        quantiles = np.percentile(all_eigs, [1,5,25,50,75,95,99])
        print("quantiles 1/5/25/50/75/95/99%:", quantiles)

        print("\n=== Selected eigenvalues ===")
        print("count :", selected_eigs.size)
        print("min   :", selected_eigs.min())
        print("max   :", selected_eigs.max())
        print("mean  :", selected_eigs.mean())
        print("median:", np.median(selected_eigs))
        print("std   :", selected_eigs.std())
        quant_sel = np.percentile(selected_eigs, [1,25,50,75,99])
        print("quantiles 1/25/50/75/99%:", quant_sel)

        print("\nFive smallest eigenvalues overall:", np.sort(all_eigs)[:5])
        print("Five largest eigenvalues overall :", np.sort(all_eigs)[-5:])


    return split_index


def sample_splitting(args, model, A, epsilon, growth_rate, x0, y0, ds_mean, method="Haim",model_init=None):
    """
    Perform sample splitting based on minimum eigenvalues and eigenvectors.
    """
    num_samples = len(A)
    x = torch.stack([a[0] for a in A])
    y = torch.tensor([a[1] for a in A], device=x.device)
    l = torch.stack([a[2] for a in A])
    sample_id = torch.tensor([a[5] for a in A], device=x.device)
    
    w_min, V = compute_min_eig_Tj_hvp(args, x, y, l, model,method=method,model_init=model_init)
    selected_idx = select_split_indices(args, w_min, V, growth_rate)
    
    def f(eps):
        new_A = []
        for idx, (xj, yj, lj, source,parent_id, sample_id) in enumerate(A):
            if idx in selected_idx:
                u_j = V[idx]
                u_j = u_j.view_as(xj)
                x_plus = xj + eps * u_j
                x_minus = xj - eps * u_j
                lambda_half = lj / 2
                new_A.extend([
                    (x_plus, yj.clone(), lambda_half),
                    (x_minus, yj.clone(), lambda_half)
                ])
            else:
                new_A.append((xj, yj, lj))
        x = torch.stack([a[0] for a in new_A]).detach().requires_grad_(True)
        y = torch.tensor([a[1] for a in new_A], device=x.device)
        l = torch.stack([a[2] for a in new_A]).detach().requires_grad_(True)
        values = model(x).squeeze()
        loss, kkt_loss, loss_verify = calc_extraction_loss(args, l, model, values, x, y)
        kkt_loss = get_kkt_loss(args, model(x).squeeze(), l, y, model).item()
        return loss.item()
    eps_min, eps_max = 1e-5, epsilon
    print(now(), "begin optimizing epsilon")
    print("eps_min:",eps_min)
    res = minimize_scalar(f, bounds=(eps_min, eps_max), method='bounded')
    print(now())
    if res.success:
        optimal_eps = res.x
        print(f"Optimal epsilon found: {optimal_eps}")
    else:
        optimal_eps = epsilon / 2
        print(f"Optimization failed, using default epsilon: {optimal_eps}")

    new_id = int(sample_id.max().item()) + 1
    new_A = []
    for idx, (xj, yj, lj, source, parent_id, sample_id) in enumerate(A):
        if idx in selected_idx:
            u_j = V[idx]
            u_j = u_j.view_as(xj)
            x_plus = xj + optimal_eps * u_j
            x_minus = xj - optimal_eps * u_j
            lambda_half = lj / 2
            new_A.extend([
                (x_plus, yj.clone(), lambda_half, source, sample_id, new_id),
                (x_minus, yj.clone(), lambda_half, source,  sample_id, new_id+1)
            ])
            new_id += 2
        else:
            new_A.append((xj, yj, lj, source, parent_id, sample_id))
    print(len(new_A), "samples after splitting")
    return new_A
