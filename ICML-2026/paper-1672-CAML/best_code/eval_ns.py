import numpy as np
import torch

from backbone.mlp import PINN
from backbone.pinnsformer import PINNsFormer
from backbone.piratenet import PirateNet
from benchmark.ns.boundary_ns import BoundaryNS
from benchmark.ns.pde_ns import NS
from caml import CAML
from pinn import PINNLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# ============================================================
# Configuration

# mlp, piratenet, pinnsformer
backbone = 'mlp'
# caml, pinn
loss = 'caml'

lr = 1e-3
w_res = 1.0
w_bc = 100.0
t_d = 25
t_r = 50
K_init = 10
K_few = 2
t_c = 1000

min_epochs = 6000
max_epochs = 20000
target_l2 = 1e-2


# ============================================================


def analytical_solution(x, y, U=(1.0, 1.0)):
    pi = np.pi
    u = pi * np.sin(pi * x) * np.cos(pi * y)
    v = -pi * np.cos(pi * x) * np.sin(pi * y)
    p = np.sin(2.0 * pi * x) * np.sin(2.0 * pi * y)

    Ux, Uy = U
    if (Ux != 0.0) or (Uy != 0.0):
        u = u + Ux
        v = v + Uy

    return u, v, p


def sample_points(device, n_interior=2000, n_boundary=200, L=1.0):
    xi = torch.rand(n_interior, 1, device=device) * L
    yi = torch.rand(n_interior, 1, device=device) * L
    interior = torch.cat([xi, yi], dim=1)

    xb = torch.rand(n_boundary, 1, device=device) * L
    yb = torch.zeros(n_boundary, 1, device=device)
    bottom = torch.cat([xb, yb], dim=1)

    xl = torch.zeros(n_boundary, 1, device=device)
    yl = torch.rand(n_boundary, 1, device=device) * L
    left = torch.cat([xl, yl], dim=1)

    xr = torch.ones(n_boundary, 1, device=device) * L
    yr = torch.rand(n_boundary, 1, device=device) * L
    right = torch.cat([xr, yr], dim=1)

    xt = torch.rand(n_boundary, 1, device=device) * L
    yt = torch.ones(n_boundary, 1, device=device) * L
    top = torch.cat([xt, yt], dim=1)

    boundary = torch.cat([bottom, left, right, top], dim=0)
    mesh = torch.cat([interior, boundary], dim=0)
    mask = torch.zeros(mesh.shape[0], device=device, dtype=torch.bool)
    mask[n_interior:] = True

    Nbc = boundary.shape[0]
    alpha = torch.ones(Nbc, 2, device=device)
    beta = torch.zeros(Nbc, 2, device=device)
    normal = torch.zeros(Nbc, 2, device=device)

    return mesh, mask, alpha, beta, normal


def eval_l2(mesh, pred, c):
    xy = mesh.detach().cpu().numpy()
    x = xy[:, 0]
    y = xy[:, 1]

    u_gt, v_gt, _ = analytical_solution(x, y)
    gt_uv = np.stack([u_gt, v_gt], axis=1)
    gt_t = torch.from_numpy(gt_uv).to(pred.device, dtype=pred.dtype)

    if c is None:
        pred_uv = pred.detach()[:, :2]
    else:
        pred_uv = pred.detach()[:, :2] + c[:2]

    num = torch.linalg.norm(pred_uv - gt_t)
    den = torch.linalg.norm(gt_t)
    return float((num / (den + 1e-12)).detach().cpu())


def get_model(model_name):
    if model_name == 'mlp':
        model = PINN(output_dim=3).to(device)
    elif model_name == 'piratenet':
        model = PirateNet(input_dim=2, out_dim=3).to(device)
    else:
        model = PINNsFormer(d_out=3, d_model=32, d_hidden=32, N=2, heads=1).to(device)
    return model


def get_loss_fn(loss_name, mask, pde, bc, w_res, w_bc, t_d, t_r):
    if loss_name == 'caml':
        loss_fn = CAML(
            mask=mask,
            pde=pde,
            boundary_condition=bc,
            w_res=w_res,
            w_bc=w_bc,
            td=t_d,
            tr=t_r,
            linear=False,
            K_init=K_init,
            K_few=K_few,
            tc=t_c
        ).to(device)
    elif loss_name == 'pinn':
        loss_fn = PINNLoss(
            mask=mask,
            pde=pde,
            boundary_condition=bc,
            w_res=w_res,
            w_bc=w_bc
        ).to(device)
    else:
        raise NotImplementedError(f"Loss function '{loss_name}' is not implemented.")
    return loss_fn


def _flat_grad_list(grads, params):
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs)


def get_cos_sim(model, pde, bc, mesh, pred, mask, eps=1e-12):
    r = pde(mesh, pred)
    pde_loss = (r ** 2).mean()

    s = bc(mesh, pred, mask)
    bc_loss = (s ** 2).mean()

    params = [p for p in model.parameters() if p.requires_grad]
    g_res = torch.autograd.grad(pde_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    g_bc = torch.autograd.grad(bc_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    v_res = _flat_grad_list(g_res, params)
    v_bc = _flat_grad_list(g_bc, params)

    dot = torch.dot(v_res, v_bc)
    n1 = torch.linalg.norm(v_res)
    n2 = torch.linalg.norm(v_bc)
    cos = dot / (n1 * n2 + eps)

    cos_num = 0
    if cos > 0:
        cos_num = 1
    return cos_num


def run_one_seed(
        seed,
        min_epochs=6000,
        max_epochs=20000,
        target_l2=1e-3
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = get_model(backbone)
    mesh, mask, alpha, beta, normal = sample_points(device=device)
    n = mesh.shape[0]
    n_bc = normal.shape[0]

    pde = NS(Re=500).to(device)
    bc = BoundaryNS(n=n_bc, normal_vector=normal).to(device)
    loss_fn = get_loss_fn(loss, mask, pde, bc, w_res, w_bc, t_d, t_r)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cos_num = 0
    l2_at_min = None
    reach_epoch = None
    l2_at_reach = None

    for t in range(max_epochs):
        opt.zero_grad()
        mesh = mesh.requires_grad_(True)
        model = model.train()
        pred = model(mesh)
        out = loss_fn(mesh=mesh, pred=pred, mask=mask, t=t)

        if t <= min_epochs:
            cos_num += get_cos_sim(model, pde, bc, mesh, pred, mask)

        pde.reset()
        bc.reset()
        total = out["total"]
        total.backward()

        opt.step()
        epoch = t + 1

        c = loss_fn.get_c() if hasattr(loss_fn, "get_c") else 0.0
        l2_now = eval_l2(mesh, pred, c)

        if epoch == min_epochs:
            l2_at_min = l2_now
            print(f"[seed {seed}] L2@{min_epochs} recorded = {l2_at_min:.6e}")

        if l2_now <= target_l2 and reach_epoch is None:
            reach_epoch = epoch
            l2_at_reach = l2_now
            print(f"[seed {seed}] Reach L2={l2_now:.6e} at epoch {reach_epoch}")

        if epoch >= min_epochs and l2_now <= target_l2:
            break

        if epoch % 100 == 0:
            print(
                f"[seed {seed}] Epoch {epoch}/{max_epochs} | "
                f"Loss {float(total.detach().cpu()):.6f} | "
                f"PDE {float(out['pde'].detach().cpu()):.6f} | "
                f"BC {float(out['boundary_condition'].detach().cpu()):.6f} | "
                f"L2 {l2_now:.6e} | "
                f"cos_sim {float(cos_num / epoch):.6e}"
            )

    if l2_at_min is None:
        c = loss_fn.get_c() if hasattr(loss_fn, "get_c") else 0.0
        l2_at_min = eval_l2(mesh, pred, c)

    if reach_epoch is None:
        c = loss_fn.get_c() if hasattr(loss_fn, "get_c") else 0.0
        l2_at_reach = eval_l2(mesh, pred, c)
        reach_epoch = -1

    return l2_at_min, l2_at_reach, reach_epoch, cos_num / min_epochs


def main():
    seeds = [999 + i for i in range(5)]

    l2_min_list = []
    l2_final_list = []
    reach_epoch_list = []
    cos_sum_list = []

    for s in seeds:
        l2_min, l2_reach, ep_reach, cs = run_one_seed(
            s,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            target_l2=target_l2
        )

        l2_min_list.append(l2_min)
        l2_final_list.append(l2_reach)
        if ep_reach != -1:
            reach_epoch_list.append(ep_reach)
        cos_sum_list.append(cs)

        print(
            f"[seed {s}] done | "
            f"L2@6000={l2_min:.6e} | "
            f"reach_ep={ep_reach} | "
            f"L2@reach={l2_reach:.6e} | "
            f"CosSum={cs:.6e}"
        )
        print('--------------------------------------------------------------------------')

    l2_6000_arr = np.array(l2_min_list, dtype=np.float64)
    l2_final_arr = np.array(l2_final_list, dtype=np.float64)
    reach_ep_arr = np.array(reach_epoch_list, dtype=np.int64)

    print("\n========== 5-seed summary ==========")
    print("Seeds:", seeds)
    print(f"L2@min mean = {float(l2_6000_arr.mean()):.6e}")
    print(f"L2@min std  = {float(l2_6000_arr.std()):.6e}")
    print(f"Reach epoch mean = {float(reach_ep_arr.mean()):.2f}")
    print(f"Reach epoch std  = {float(reach_ep_arr.std()):.2f}")
    print(f"L2@reach mean = {float(l2_final_arr.mean()):.6e}")
    print(f"L2@reach std  = {float(l2_final_arr.std()):.6e}")
    print(f"Cosine similarity sum (rate) = {float(np.mean(cos_sum_list)):.6e}")

    print("All L2@min:", l2_min_list)
    print("All reach_epoch:", reach_epoch_list)
    print("All L2@reach:", l2_final_list)
    print("All CosSum:", cos_sum_list)


if __name__ == "__main__":
    main()
