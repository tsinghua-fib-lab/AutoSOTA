"""Hermite-NGP image reconstruction from gradient or Laplacian supervision.

Reconstructs the camera image from its gradient field (`--loss grad`, default)
or Laplacian (`--loss lap`). Field dumps (`u`, `ux`, `uy`, `lap`) are written
at 4 training checkpoints (init, ~25%, ~75%, final) for visualization.

Usage:
    python examples/image_recon.py --out results/image_recon_256
    python examples/image_recon.py --out out --loss lap --hs 18
"""
import sys, os, time, json, argparse
import torch, torch.nn as nn, numpy as np
from PIL import Image
import skimage.data

# Make `from hermite_ngp.*` work regardless of cwd by adding the repo root
# (parent of this examples/ dir) to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = 'cuda'

from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA
import hermite_mlp_cuda_v2


# -----------------------------------------------------------------------------
# Custom autograd Function for SIREN MLP with analytic 2nd derivatives
# (copied verbatim from grad_push_best.py / image_lap_camera.py)
# -----------------------------------------------------------------------------
class HL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, dx, dy, dxx, dyy, w, b, o, a):
        out = hermite_mlp_cuda_v2.forward(
            h.contiguous(), dx.contiguous(), dy.contiguous(),
            dxx.contiguous(), dyy.contiguous(),
            w.contiguous(), b.contiguous(), o, a
        )
        ctx.save_for_backward(h, dx, dy, dxx, dyy, w, out[5], out[6], out[7])
        ctx.omega = o
        ctx.act = a
        return out[0], out[1], out[2], out[3], out[4]

    @staticmethod
    def backward(ctx, gh, gdx, gdy, gdxx, gdyy):
        h, dx, dy, dxx, dyy, w, z, dzx, dzy = ctx.saved_tensors
        o = ctx.omega
        o2 = o * o
        if ctx.act:
            s = torch.sin(o * z); c = torch.cos(o * z)
            ap = o * c; app = -o2 * s
        else:
            ap = torch.ones_like(z); app = torch.zeros_like(z)
        gz = gh * ap + gdx * app * dzx + gdy * app * dzy
        gw = (gz.T @ h
              + (gdx * ap).T @ dx + (gdy * ap).T @ dy
              + (gdxx * ap).T @ dxx + (gdyy * ap).T @ dyy
              + (gdxx * app * dzx).T @ dx + (gdyy * app * dzy).T @ dy)
        gb = gz.sum(0)
        gh_  = gz @ w + (gdxx * app * dzx) @ w + (gdyy * app * dzy) @ w
        gdx_ = (gdx * ap) @ w + (gdxx * app * dzx) @ w
        gdy_ = (gdy * ap) @ w + (gdyy * app * dzy) @ w
        gdxx_ = (gdxx * ap) @ w
        gdyy_ = (gdyy * ap) @ w
        return gh_, gdx_, gdy_, gdxx_, gdyy_, gw, gb, None, None


class M(nn.Module):
    def __init__(self, hs, levels, scale, hidden, layers, omega):
        super().__init__()
        self.om = omega
        ed = levels * 2
        self.enc = HermiteHashEncodingCUDA(
            n_input_dims=2, n_levels=levels, n_features_per_level=2,
            log2_hashmap_size_1=hs, log2_hashmap_size_2=hs, log2_hashmap_size_3=hs,
            base_resolution=4, per_level_scale=scale,
        ).to(device)
        self.lays = nn.ModuleList([nn.Linear(ed, hidden)] +
                                  [nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.out = nn.Linear(hidden, 1)
        self.to(device)

    def forward(self, x):
        h = self.enc(x)
        for l in self.lays:
            h = torch.sin(self.om * l(h))
        return self.out(h)

    def fwd_d(self, x):
        e, dx, dy, dxx, dyy = self.enc.forward_with_second_derivatives_cuda(x)
        h, hx, hy, hxx, hyy = e, dx, dy, dxx, dyy
        for l in self.lays:
            h, hx, hy, hxx, hyy = HL.apply(
                h, hx, hy, hxx, hyy, l.weight, l.bias, self.om, True
            )
        u, ux, uy, uxx, uyy = HL.apply(
            h, hx, hy, hxx, hyy, self.out.weight, self.out.bias, self.om, False
        )
        return u, ux, uy, uxx + uyy


def dump_full(model, coords, H, W, outdir, tag):
    """Evaluate model on full HxW grid (in chunks to avoid OOM) and save raw arrays."""
    chunk = 65536
    u_chunks, ux_chunks, uy_chunks, lap_chunks = [], [], [], []
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            sub = coords[i:i + chunk]
            u, ux, uy, lap = model.fwd_d(sub)
            u_chunks.append(u.squeeze(-1).cpu().numpy())
            ux_chunks.append(ux.squeeze(-1).cpu().numpy())
            uy_chunks.append(uy.squeeze(-1).cpu().numpy())
            lap_chunks.append(lap.squeeze(-1).cpu().numpy())
    u  = np.concatenate(u_chunks).reshape(H, W)
    ux = np.concatenate(ux_chunks).reshape(H, W)
    uy = np.concatenate(uy_chunks).reshape(H, W)
    lap = np.concatenate(lap_chunks).reshape(H, W)
    np.savez(os.path.join(outdir, f'{tag}.npz'), u=u, ux=ux, uy=uy, lap=lap)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--loss', type=str, choices=['grad', 'lap'], required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--seed', type=int, default=7)
    # Architecture
    p.add_argument('--hs', type=int, default=None)
    p.add_argument('--omega', type=float, default=None)
    p.add_argument('--levels', type=int, default=8)
    p.add_argument('--scale', type=float, default=2.0)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    # Training
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bcw', type=float, default=None)
    p.add_argument('--sched', type=str, default=None)  # 'step' | 'cosine'
    p.add_argument('--step-size', type=int, default=40000)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--checkpoints', type=str, default=None,
                   help='comma-separated list of iterations to dump viz; default depends on loss')
    p.add_argument('--image-res', type=int, default=256,
                   help='Camera image resolution (256 or 512). 512 matches Kairanda paper config.')
    p.add_argument('--batch', type=int, default=65536,
                   help='Random collocation batch size per iter')
    args = p.parse_args()

    # Defaults per loss type (paper-validated configs)
    if args.loss == 'grad':
        if args.hs is None:    args.hs = 16
        if args.omega is None: args.omega = 2.0
        if args.epochs is None:args.epochs = 200000
        if args.bcw is None:   args.bcw = 10.0
        if args.sched is None: args.sched = 'step'
    else:  # lap
        if args.hs is None:    args.hs = 18
        if args.omega is None: args.omega = 0.5
        if args.epochs is None:args.epochs = 50000
        if args.bcw is None:   args.bcw = 5000.0
        if args.sched is None: args.sched = 'cosine'

    if args.checkpoints is None:
        # 4 dump points: init, ~25%, ~75%, final
        ck = [0, args.epochs // 4, (args.epochs * 3) // 4, args.epochs]
    else:
        ck = sorted(set(int(x) for x in args.checkpoints.split(',')))
    print(f'Will dump viz at iters: {ck}', flush=True)

    os.makedirs(args.out, exist_ok=True)

    # --- Load image at requested resolution ---
    if args.image_res == 512:
        # skimage.data.camera() is natively 512x512
        img = np.array(skimage.data.camera())
    else:
        img = np.array(Image.fromarray(skimage.data.camera()).resize(
            (args.image_res, args.image_res)))
    img_t = torch.tensor(img, dtype=torch.float32, device=device) / 255.0
    H, W = img_t.shape
    SIZE = H
    print(f'Image: {H}x{W}', flush=True)

    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    pixels = img_t.flatten()
    N = coords.shape[0]

    # --- GT gradient (Sobel) and Laplacian (5pt stencil), with size scaling
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                            device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                            device=device).view(1, 1, 3, 3) / 8.0
    lap_k   = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32,
                            device=device).view(1, 1, 3, 3)
    padded = torch.nn.functional.pad(img_t.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    gt_dx  = torch.nn.functional.conv2d(padded, sobel_x)[0, 0].flatten() * SIZE
    gt_dy  = torch.nn.functional.conv2d(padded, sobel_y)[0, 0].flatten() * SIZE
    gt_lap = torch.nn.functional.conv2d(padded, lap_k)[0, 0].flatten() * (SIZE ** 2)

    # Save GT once
    if not os.path.exists(os.path.join(args.out, 'gt.npz')):
        gt_grad = torch.stack([gt_dx, gt_dy], dim=-1).reshape(H, W, 2).cpu().numpy()
        np.savez(os.path.join(args.out, 'gt.npz'),
                 u=img_t.cpu().numpy(),
                 grad=gt_grad,
                 lap=gt_lap.reshape(H, W).cpu().numpy())
        print(f'Saved GT -> gt.npz', flush=True)

    # Boundary indices (used for BC loss when supervising on derivatives only)
    bnd = set()
    for i in range(H):
        bnd.add(i * W); bnd.add(i * W + W - 1)
    for j in range(W):
        bnd.add(j); bnd.add((H - 1) * W + j)
    bnd = torch.tensor(list(bnd), device=device)

    # --- Model ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = M(args.hs, args.levels, args.scale, args.hidden, args.layers, args.omega)
    npar = sum(p.numel() for p in model.parameters())
    tag = (f'hermite_{args.loss}_h{args.hs}_om{args.omega}_lvl{args.levels}_'
           f'sc{args.scale}_lay{args.layers}_hid{args.hidden}_lr{args.lr}_'
           f'{args.sched}_ep{args.epochs}_seed{args.seed}')
    print(f'Starting {tag} | params={npar:,}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.sched == 'step':
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.sched == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    else:
        raise ValueError(args.sched)

    # --- Initial dump (iter 0)
    if 0 in ck:
        dump_full(model, coords, H, W, args.out, f'iter_{0:07d}')
        print(f'  dumped init at iter 0', flush=True)
        ck = [c for c in ck if c != 0]

    psnr_history = []
    best_psnr = 0.0
    best_state = None
    t0 = time.perf_counter()

    for it in range(args.epochs):
        idx = torch.randint(0, N, (65536,), device=device)
        u, ux, uy, lap = model.fwd_d(coords[idx])

        if args.loss == 'grad':
            data_loss = ((ux.squeeze() - gt_dx[idx]) ** 2 +
                         (uy.squeeze() - gt_dy[idx]) ** 2).mean()
        else:  # lap
            data_loss = ((lap.squeeze() - gt_lap[idx]) ** 2).mean()

        bc_loss = ((model(coords[bnd]).squeeze() - pixels[bnd]) ** 2).mean()
        loss = data_loss + args.bcw * bc_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        # Periodic PSNR check + dump if scheduled
        if (it + 1) % 5000 == 0 or (it + 1) in ck:
            with torch.no_grad():
                p_full = model(coords).squeeze()
                mse = ((p_full - pixels) ** 2).mean()
                psnr = (-10 * torch.log10(mse)).item()
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elapsed = time.perf_counter() - t0
            psnr_history.append({'iter': it + 1, 'psnr': psnr, 'best': best_psnr,
                                 'time_s': elapsed})
            print(f'  iter {it + 1:>7}: PSNR={psnr:.2f} best={best_psnr:.2f} t={elapsed:.0f}s',
                  flush=True)

        if (it + 1) in ck:
            dump_full(model, coords, H, W, args.out, f'iter_{it + 1:07d}')
            print(f'  dumped viz at iter {it + 1}', flush=True)

    total_time = time.perf_counter() - t0
    print(f'DONE {tag} | best PSNR={best_psnr:.2f} | total {total_time:.0f}s', flush=True)

    # Save metadata
    meta = {
        'method': 'hermite_ngp',
        'loss_type': args.loss,
        'image': 'skimage.camera_256x256',
        'hash_size': args.hs,
        'omega': args.omega,
        'levels': args.levels,
        'scale': args.scale,
        'hidden': args.hidden,
        'layers': args.layers,
        'lr': args.lr,
        'bcw': args.bcw,
        'sched': args.sched,
        'epochs': args.epochs,
        'seed': args.seed,
        'best_psnr': best_psnr,
        'total_time_s': total_time,
        'n_params': npar,
        'psnr_history': psnr_history,
        'checkpoints': ck,
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save model state for reuse
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out, 'model.pth'))
    print(f"Model params saved (best-PSNR snapshot): {os.path.join(args.out, 'model.pth')}")


if __name__ == '__main__':
    main()
