import torch
import torchvision

import wandb
from common_utils.common import now
from CreateModel import Flatten
from evaluations import get_evaluation_score_dssim, viz_nns, get_evaluation_score_psnr


def l2_dist(x, y):
    """x, y should be of shape [batch, D]"""
    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    squared_dists = xx + yy - 2 * xy
    dists = torch.sqrt(squared_dists.clamp(min=0.0))
    return dists


def diversity_loss(x, min_dist):
    flat_x = Flatten()(x)
    D = l2_dist(flat_x, flat_x)
    D.fill_diagonal_(torch.inf)
    nn_dist = D.min(dim=1).values
    relevant_nns = nn_dist[nn_dist < min_dist]
    if relevant_nns.shape[0] > 0:
        return relevant_nns.mul(-20).sigmoid().mean()
    else:
        return torch.tensor(0)


# def send_input_data(args, model, x0, y0):
#     if not args.wandb_active: return
#     _, c, h, w = x0.shape
#     n_weights = model.layers[0].weight.shape[0]
#     w = model.layers[0].weight.reshape(n_weights, c, h, w)
#     w_nns, _ = viz_nns(w.data, x0, max_per_nn=2)
#     w_viz = torchvision.utils.make_grid(w_nns[:100], normalize=False, nrow=20)
#     wandb.log({
#         "weights_of_first_layer": wandb.Image(w_viz),
#     })


def get_trainable_params(args, x0, opt_method="SGD",method="Haim"):
    n, c, h, w = x0.shape
    x = torch.randn(args.extraction_data_amount, c, h, w).to(args.device) * args.extraction_init_scale
    x.requires_grad_(True)
    l = torch.rand(args.extraction_data_amount, 1).to(args.device)
    if method == "Loo":
        l = (torch.rand(args.extraction_data_amount, 1) * 2 - 1).to(args.device)
    l.requires_grad_(True)
    if opt_method == "Adam":
        opt_x = torch.optim.Adam([x], lr=args.extraction_lr)
        opt_l = torch.optim.Adam([l], lr=args.extraction_lambda_lr)
    else:
        opt_x = torch.optim.SGD([x], lr=args.extraction_lr, momentum=0.9)
        opt_l = torch.optim.SGD([l], lr=args.extraction_lambda_lr, momentum=0.9)
    return l, opt_l, opt_x, x


def get_kkt_loss(args, values, l, y, model, method="Haim",model_init=None,x=None):
    l = l.squeeze()
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
            assert values.dim() == 1
            assert l.dim() == 1
            assert y.dim() == 1
            assert values.shape == l.shape == y.shape
            output = values * l * y
        para = model.parameters()
        grad = torch.autograd.grad(
            outputs=output,
            inputs=model.parameters(),
            grad_outputs=torch.ones_like(output, requires_grad=False, device=output.device).div(args.extraction_data_amount),
            create_graph=True,
            retain_graph=True,
        )
    elif method == 'Loo':
        output_i = model_init(x).squeeze(1)*l
        para_i = list(model_init.parameters())
        grad_i = torch.autograd.grad(
            outputs=output_i,
            inputs=para_i,
            grad_outputs=torch.ones_like(output_i, requires_grad=False, device=output_i.device).div(args.extraction_data_amount),
            create_graph=True,
            retain_graph=True,
        )
        output_f = values * l
        para_f = list(model.parameters())
        grad_f = torch.autograd.grad(
            outputs=output_f,
            inputs=para_f,
            grad_outputs=torch.ones_like(output_f, requires_grad=False, device=output_f.device).div(args.extraction_data_amount),
            create_graph=True,
            retain_graph=True,
        )
        grad = [(grad_i[i] + grad_f[i])/2 for i in range(len(grad_i))]
        para = [para_f[i] - para_i[i] for i in range(len(para_i))]
        # compare_parameters(para_i, para_f, grad_i, grad_f, name="LOO Method Comparison")
    
    kkt_loss = 0
    for i, (p, grad) in enumerate(zip(para, grad)):
        assert p.shape == grad.shape
        l = (p.detach().data - grad).pow(2).sum()
        kkt_loss += l
    return kkt_loss

def compare_parameters(para_i, para_f, grad_i, grad_f, name="Comparison"):
    print(f"\n=== {name} ===")
    
    for idx, (p_i, p_f, g_i, g_f) in enumerate(zip(para_i, para_f, grad_i, grad_f)):
        print(f"\nLayer {idx}:")
        
        # Compute summary statistics
        stats = {
            'p_i_range': f"[{p_i.min():.3e}, {p_i.max():.3e}],p_i[0][0]={p_i.view(-1)[0]:.3e}",
            'p_f_range': f"[{p_f.min():.3e}, {p_f.max():.3e}],p_f[0][0]={p_f.view(-1)[0]:.3e}",
            'p_diff_range': f"[{(p_f-p_i).min():.3e}, {(p_f-p_i).max():.3e}],p_diff[0][0]={(p_f - p_i).view(-1)[0]:.3e}",
            'g_i_range': f"[{g_i.min():.3e}, {g_i.max():.3e}],g_i[0][0]={g_i.view(-1)[0]:.3e}",
            'g_f_range': f"[{g_f.min():.3e}, {g_f.max():.3e}],g_f[0][0]={g_f.view(-1)[0]:.3e}",
            'p_i_norm': f"{p_i.norm():.3e}",
            'p_f_norm': f"{p_f.norm():.3e}",
            'p_diff_norm': f"{(p_f-p_i).norm():.3e}",
            'g_i_norm': f"{g_i.norm():.3e}",
            'g_f_norm': f"{g_f.norm():.3e}",
        }
        
        for key, value in stats.items():
            print(f"  {key:15}: {value}")

def get_verify_loss(args, x, l, method="Haim"):
    loss_verify = 0
    if method == 'Haim':
        loss_verify += 1 * (x - 1).relu().pow(2).sum()
        loss_verify += 1 * (-1 - x).relu().pow(2).sum()
        loss_verify += 5 * (-l + args.extraction_min_lambda).relu().pow(2).sum()
    elif method == 'Loo':
        loss_verify += 1 * (x - 1).relu().pow(2).sum()
        loss_verify += 1 * (-1 - x).relu().pow(2).sum()
        loss_verify += 5 * (-l.abs() + args.extraction_min_lambda).relu().pow(2).sum()

    return loss_verify


def calc_extraction_loss(args, l, model, values, x, y, method="Haim",model_init=None):
    kkt_loss, loss_verify = torch.tensor(0), torch.tensor(0)
    if args.extraction_loss_type == 'kkt':
        kkt_loss = get_kkt_loss(args, values, l, y, model, method=method,model_init=model_init,x=x)
        loss_verify = get_verify_loss(args, x, l, method=method)
        if method == 'Loo' and args.extraction_kkt_scale != 1.0:
            loss = kkt_loss * args.extraction_kkt_scale + loss_verify
        else:
            loss = kkt_loss + loss_verify

    elif args.extraction_loss_type == 'naive':
        loss_naive = -(values[y == 1].mean() - values[y == -1].mean())
        loss_verify = loss_verify.to(args.device).to(torch.get_default_dtype())
        loss_verify += (x - 1).relu().pow(2).sum()
        loss_verify += (-1 - x).relu().pow(2).sum()

        loss = loss_naive + loss_verify
    else:
        raise ValueError(f'unknown args.extraction_loss_type={args.extraction_loss_type}')

    return loss, kkt_loss, loss_verify

def save_epoch_ssims_csv(epoch, ssims_vector, filename='ssims_data.csv'):
    ssims_vector = ssims_vector.cpu().numpy()[:50]  # Keep the first 50 SSIM values
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            headers = ['epoch'] + [f'max_ssim_{i+1}' for i in range(len(ssims_vector))]
            writer.writerow(headers)
        
        row = [epoch] + list(ssims_vector)
        writer.writerow(row)
    
    print(f"Epoch {epoch} has been saved.")

import os, csv
def evaluate_extraction(args, epoch, loss_extract, loss_verify, x, x0, y0, ds_mean, top_k=10):
    x_grad = x.grad.clone().data
    x = x.clone().data
    if args.wandb_active:
        wandb.log({
            "extraction epoch": epoch,
            "loss extract": loss_extract,
            "loss verify": loss_verify,
        })

    xx = x.data.clone()
    yy = x0.clone()
    # metric = 'ncc'
    metric = 'l2'
    # if args.dataset == 'mnist':
    #     metric = 'l2'

    qq, _ = viz_nns(xx, yy, max_per_nn=4, metric=metric)
    extraction_grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=10)
    _, v = viz_nns(xx, yy, max_per_nn=1, metric=metric)
    l2_dist = v
    extraction_score = v[:top_k].mean().item()
    print(v[:20])

    xx += ds_mean
    yy += ds_mean
    qq, _ = viz_nns(xx, yy, max_per_nn=4, metric=metric)
    extraction_grid_with_mean = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=10)
    _, v = viz_nns(xx, yy, max_per_nn=1, metric=metric)
    extraction_score_with_mean = v[:top_k].mean().item()

    # SSIM EVALUATION
    xx = x.data.clone()
    yy = x0.clone()
    dssim_score, dssim_grid, dssims, binary_x = get_evaluation_score_dssim(xx, yy, ds_mean, vote=None, show=False,top=top_k)
    print(dssims[:10])
    csv_file = os.path.join(args.output_dir, f"topSSIM_split_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}.csv")
    ssims = 1 - 2 * dssims
    if args.dataset == 'mnist':
        ssims = l2_dist
    save_epoch_ssims_csv(epoch, ssims, filename=csv_file)

    # PSNR EVALUATION
    xx = x.data.clone()
    yy = x0.clone()
    psnr_score, psnr_grid = get_evaluation_score_psnr(xx, yy, ds_mean, show=False,top=top_k)

    if args.wandb_active:
        wandb.log({
            "extraction": wandb.Image(extraction_grid),
            "extraction score": extraction_score,
            "extraction with mean": wandb.Image(extraction_grid_with_mean),
            "extraction score with mean": extraction_score_with_mean,
            "dssim score": dssim_score,
            "extraction dssim": wandb.Image(dssim_grid),
            "psnr score": psnr_score,
            "extraction psnr": wandb.Image(psnr_grid),
        })

    print(f'{now()} T={epoch} ; Losses: extract={loss_extract.item():5.10g} verify={loss_verify.item():5.5g} grads={x_grad.abs().mean()} Extraction-Score={extraction_score} Extraction-DSSIM={dssim_score} Extraction-PSNR={psnr_score}')

    return extraction_score, dssim_score, psnr_score, binary_x
